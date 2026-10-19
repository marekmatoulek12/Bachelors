import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, pi

# ==========================================
# 1. ZDE JE VAŠE TŘÍDA EKF Z BLUEPRINTS_3
# ==========================================
class VelocityEKF:
    def __init__(self):
        # Stavy: [vx, vy]
        self.x = np.zeros(2)
        self.P = np.eye(2) * 1.0
        
        self.Q = np.eye(2) * 0.05       # Šum modelu (IMU)
        self.R_wheels = np.diag([0.1, 100]) # Šum kol (když nekloužou)
        
        # Zvýšená matice R_gps pro potlačení šumu z derivace
        self.R_gps = np.eye(2) * 20.0    

    def predict(self, dt: float, ax: float, ay: float, yaw_rate: float):
        if dt <= 0: return
        vx_new = self.x[0] + (ax + yaw_rate * self.x[1]) * dt
        vy_new = self.x[1] + (ay - yaw_rate * self.x[0]) * dt
        F = np.array([[1.0, yaw_rate * dt], [-yaw_rate * dt, 1.0]])
        self.x = np.array([vx_new, vy_new])
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z_vx: float, z_vy: float, R_matrix: np.ndarray, is_gps: bool = False):
        z = np.array([z_vx, z_vy])
        H = np.eye(2)
        
        # y je "Inovace" - rozdíl mezi tím, co jsme čekali (self.x) a co jsme změřili (z)
        y = z - (H @ self.x)
        
        # OCHRANA PROTI NESMYSLŮM (GATING)
        # Pokud toto měření pochází z GPS a rozdíl oproti naší predikci z IMU 
        # je větší než např. 1.5 m/s, je to jasný šum satelitu. Měření zahodíme.
        if is_gps:
            if abs(y[0]) > 1.5 or abs(y[1]) > 1.5:
                # Záměrně neděláme nic, ponecháme stav čistě z IMU
                return
                
        S = H @ self.P @ H.T + R_matrix
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ H) @ self.P

# ==========================================
# 2. SIMULÁTOR (OFFLINE TESTBENCH)
# ==========================================
def run_ekf_simulation(input_csv_path: str, output_csv_path: str):
    print(f"Načítám log: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    
    # Vytvoříme přesnou kopii pro výstupní CSV se stejnou strukturou
    df_out = df.copy()
    
    ekf = VelocityEKF()
    
    # Konstanty auta
    front_axle_len = 0.16
    wheel_radius = 0.033
    
    # Pomocné proměnné pro simulaci asynchronních zpráv
    last_time = df['timestamp'].iloc[0]
    last_gps_time = None
    last_gps_x, last_gps_y = None, None
    
    # Pomocná pole čistě pro účely vykreslení grafu (neuloží se do CSV)
    gps_vx_list = np.full(len(df), np.nan)
    gps_vy_list = np.full(len(df), np.nan)

    for index, row in df.iterrows():
        current_time = row['timestamp']
        dt = current_time - last_time
        if dt <= 0: dt = 0.001
        
        # --- 1. PREDICT (Simulace IMU Callbacku) ---
        ax = row['accel_x']
        ay = row['accel_y']
        yaw_rate = row['yaw_rate']
        if not np.isnan(ax) and not np.isnan(ay) and not np.isnan(yaw_rate):
            ekf.predict(dt, ax, ay, yaw_rate)
            
        # --- 2. UPDATE KOLA (Simulace Encoder Callbacku) ---
        fl_rpm = row['fl_rpm']
        fr_rpm = row['fr_rpm']
        steer = row['cmd_steer_rad']
        
        # Simulujeme update pouze pokud máme data z kol
        if not np.isnan(fl_rpm) and not np.isnan(fr_rpm) and not np.isnan(steer):
            speed = ((fl_rpm + fr_rpm) / 2.0) * (2 * pi * wheel_radius) / 60.0
            z_vx_wheels = cos(steer) * speed
            z_vy_wheels = sin(steer) * speed - (yaw_rate * front_axle_len)
            
            is_slipping = (abs(ay) > 4.0) or (abs(yaw_rate) > 2.0)
            current_R_w = ekf.R_wheels * 1000.0 if is_slipping else ekf.R_wheels
            ekf.update(z_vx_wheels, z_vy_wheels, current_R_w)

        # --- 3. UPDATE GPS (Simulace GPS Callbacku) ---
        curr_gps_x = row['gps_x']
        curr_gps_y = row['gps_y']
        curr_gps_yaw = row['gps_yaw']
        
        if not np.isnan(curr_gps_x):
            # GPS se aktualizuje mnohem pomaleji než IMU. 
            # Update spustíme jen když se pozice v CSV změní.
            if last_gps_x is None:
                last_gps_x, last_gps_y, last_gps_time = curr_gps_x, curr_gps_y, current_time
            elif curr_gps_x != last_gps_x or curr_gps_y != last_gps_y:
                dt_gps = current_time - last_gps_time
                if dt_gps > 0:
                    v_X_global = (curr_gps_x - last_gps_x) / dt_gps
                    v_Y_global = (curr_gps_y - last_gps_y) / dt_gps
                    
                    c, s = cos(-curr_gps_yaw), sin(-curr_gps_yaw)
                    z_vx_gps = c * v_X_global - s * v_Y_global
                    z_vy_gps = s * v_X_global + c * v_Y_global
                    
                    # Uložení do pole pro graf
                    gps_vx_list[index] = z_vx_gps
                    gps_vy_list[index] = z_vy_gps
                    
                    # Update EKF daty z GPS (s použitím gatingu)
                    ekf.update(z_vx_gps, z_vy_gps, ekf.R_gps, is_gps=True)
                    
                last_gps_x, last_gps_y, last_gps_time = curr_gps_x, curr_gps_y, current_time

        # ==============================================================
        # PŘEPIS PŮVODNÍCH HODNOT NOVÝMI ODHADY EKF PŘÍMO DO KOPIE DAT
        # ==============================================================
        df_out.at[index, 'vx'] = ekf.x[0]
        df_out.at[index, 'vy'] = ekf.x[1]
        
        last_time = current_time

    # ==========================================
    # 3. ULOŽENÍ A VYKRESLENÍ
    # ==========================================
    
    # Uložíme kopii dat s vyhlazenými rychlostmi do stejného formátu (bez indexu)
    df_out.to_csv(output_csv_path, index=False)
    print(f"Hotovo. Vyhlazená data uložena do: {output_csv_path}")
    
    # --- Graf 1 a 2: Rychlosti ---
    # Porovnáváme původní data (df) s novými daty (df_out)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Testbench: Porovnání EKF odhadu rychlostí (Vx, Vy)")
    
    ax1.plot(df['timestamp'], df['vx'], label='Původní Vx (z auta)', color='red', alpha=0.5)
    ax1.plot(df_out['timestamp'], df_out['vx'], label='Nové EKF Vx (Simulátor)', color='blue', linewidth=2)
    ax1.scatter(df['timestamp'], gps_vx_list, label='Šum derivované GPS Vx', color='orange', s=10)
    ax1.set_ylabel("Rychlost Vx [m/s]")
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(df['timestamp'], df['vy'], label='Původní Vy (z auta)', color='red', alpha=0.5)
    ax2.plot(df_out['timestamp'], df_out['vy'], label='Nové EKF Vy (Simulátor)', color='green', linewidth=2)
    ax2.scatter(df['timestamp'], gps_vy_list, label='Šum derivované GPS Vy', color='orange', s=10)
    ax2.set_ylabel("Rychlost Vy [m/s]")
    ax2.set_xlabel("Čas [s]")
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# ==========================================
# SPUŠTĚNÍ
# ==========================================
if __name__ == "__main__":
    # ZDE ZMĚŇTE NÁZEV SOUBORU ZA VÁŠ LOG
    input_file = "ramp_speed_test_log_1792163567.csv" 
    
    # Automatické vytvoření jména pro vyhlazený soubor
    output_file = "simulated_output.csv"
    
    run_ekf_simulation(input_file, output_file)