import os
import sys
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# --- KONFIGURACE ---
YAW_RATE_COL = 'yaw_rate'          # Název sloupce z reálných logů
VELOCITY_COL = 'velocity'          # Název sloupce z reálných logů pro rychlost
STEADY_STATE_SAMPLES = 100         # Počet posledních vzorků pro výpočet ustáleného stavu

def find_repo_root():
    current = os.path.abspath(os.path.dirname(__file__))
    while current != '/' and 'Model_lib_and_sim' not in os.listdir(current):
        parent = os.path.dirname(current)
        if parent == current:
            return os.path.abspath(os.path.dirname(__file__))
        current = parent
    return current

repo_root = find_repo_root()
model_lib_path = os.path.join(repo_root, 'Model_lib_and_sim')
if model_lib_path not in sys.path:
    sys.path.append(model_lib_path)

try:
    from simulation import Simulation
    SIMULATION_AVAILABLE = True
except ImportError:
    print("VAROVÁNÍ: Nepodařilo se importovat 'Simulation'. Spouštím v testovacím (fallback) režimu.")
    SIMULATION_AVAILABLE = False

def run_sim_to_steady_state(sim_steer_angle_deg, velocity):
    """
    Inicializuje a spustí simulaci s předgenerovanými poli `reference_vx` a `reference_df`.
    """
    sim_steer_rad = np.deg2rad(sim_steer_angle_deg)
    
    if not SIMULATION_AVAILABLE:
        # Fallback pro otestování skriptu bez reálné simulace
        wheelbase = 0.3
        return (velocity / wheelbase) * np.tan(sim_steer_rad) * 0.8

    sim_step = 0.001
    acc_time = 2000       # 2s rovně
    steering_time = 2000  # 2s nájezd řízení
    test_time = 4000      # 4s ustálený stav 
    sample_count = acc_time + steering_time + test_time
    
    reference_vx = []
    reference_df = []
    
    # Generování požadavků pro simulaci
    for i in range(sample_count):
        reference_vx.append(velocity)  # Simulujeme konstantní reálnou rychlostí
        if i < acc_time:
            reference_df.append(0.0)
        elif i < acc_time + steering_time:
            # Plynulý nárůst na požadovaný úhel
            reference_df.append(sim_steer_rad * (i - acc_time) / steering_time)
        else:
            # Udržování úhlu
            reference_df.append(sim_steer_rad)

    # Vytvoření a spuštění simulace přesně dle vaší ukázky
    sim = Simulation(
        reference_vx=reference_vx,
        reference_df=reference_df,
        sim_length=None,
        sim_step=sim_step
    )
    sim.run()
    
    # --- EXTRAKCE YAW RATE ---
    # Musíme získat ustálenou hodnotu yaw_rate ze simulace.
    try:
        # Příklad A: Data jsou v sim.logger.yaw_rate
        if hasattr(sim, 'logger') and hasattr(sim.logger, 'yaw_rate'):
            steady_yaw = np.mean(sim.logger.yaw_rate[-1000:]) 
        
        # Příklad B: Data jsou ve stavové matici sim.logger.states 
        elif hasattr(sim, 'logger') and hasattr(sim.logger, 'states'):
            steady_yaw = np.mean(sim.logger.states[-1000:, 2])
            
        # Příklad C: Data jsou v pandas DataFrame sim.logger.df
        elif hasattr(sim, 'logger') and hasattr(sim.logger, 'df'):
            steady_yaw = sim.logger.df['yaw_rate'].tail(1000).mean()
            
        else:
            raise AttributeError("Nenalezen standardní atribut pro yaw_rate v sim.logger")
            
        return steady_yaw
        
    except Exception as e:
        print(f"Chyba při čtení dat ze simulace: {e}")
        return 0.0

def optimize_steering_angle(target_yaw_rate, velocity):
    def objective(steer_deg):
        sim_yaw = run_sim_to_steady_state(steer_deg, velocity)
        return (sim_yaw - target_yaw_rate)**2

    # Hledáme optimální úhel (od 0 do 45 stupňů)
    res = minimize_scalar(objective, bounds=(-45, 45), method='bounded')
    if res.success:
        return res.x
    else:
        return np.nan

def calibrate_log(file_path, A, B):
    STEERING_CMD_COL = "cmd_steer_rad"
    """
    Načte zadaný .csv soubor, přidá sloupec real_cmd_steer obsahující data
    převedená lineární funkcí Ax + B a uloží jej jako calibrated_{original_name}.csv
    do složky Model_lib_and_sim/Identification.
    """
    if not os.path.exists(file_path):
        print(f"Chyba: Soubor {file_path} neexistuje.")
        return

    df = pd.read_csv(file_path)
    
    # Detekce sloupce s povelem řízení
    cmd_col = None
    for col in df.columns:
        if col.lower() in [STEERING_CMD_COL.lower(), 'cmd_steer', 'steering', 'steer', 'delta', 'df']:
            cmd_col = col
            break
            
    if not cmd_col:
        candidates = [c for c in df.columns if 'steer' in c.lower() or 'deg' in c.lower() or 'df' in c.lower()]
        if candidates:
            cmd_col = candidates[0]
        else:
            print(f"Chyba: V souboru {file_path} nebyl nalezen sloupec pro povel řízení.")
            return

    # Přepočet lineární funkcí Ax + B a uložení do nového sloupce
    df['real_steer_rad'] = df[cmd_col] * A + B * np.pi/180 # predpoklad, ze puvodni odvozeni je ze stupnu, tedy je nutne prepocitat na radiany

    # Výstupní složka Model_lib_and_sim/Identification/Calibrated
    target_dir = os.path.join(repo_root, 'Model_lib_and_sim', 'Identification', 'Calibrated')
    os.makedirs(target_dir, exist_ok=True)

    original_name = os.path.basename(file_path)
    output_filename = f"calibrated_{original_name}"
    output_path = os.path.join(target_dir, output_filename)

    df.to_csv(output_path, index=False)
    print(f"Kalibrovaný soubor uložen do: {output_path}")

def main():
    search_pattern = os.path.join(repo_root, '**', 'rst_*_log_*.csv')
    log_files = glob.glob(search_pattern, recursive=True)
    
    if not log_files:
        search_pattern = os.path.join(repo_root, '**', 'rst*.csv')
        log_files = glob.glob(search_pattern, recursive=True)

    if not log_files:
        print(f"Nenalezeny žádné CSV soubory rst_*_log_* v {repo_root}")
        return

    print(f"Nalezeno {len(log_files)} logů. Analyzuji data...")
    results = []

    for file in log_files:
        match = re.search(r'rst_(-?\d+)', file)
        if not match: continue
        cmd_angle = float(match.group(1))

        try:
            df = pd.read_csv(file)
            yaw_col = YAW_RATE_COL if YAW_RATE_COL in df.columns else [c for c in df.columns if 'yaw' in c.lower()][0]
            vel_col = VELOCITY_COL if VELOCITY_COL in df.columns else [c for c in df.columns if 'vel' in c.lower() or 'speed' in c.lower() or 'vx' in c.lower()][0]

            steady_df = df.tail(STEADY_STATE_SAMPLES)
            real_yaw_rate = steady_df[yaw_col].mean()
            real_velocity = steady_df[vel_col].mean()
            
            print(f"Zpracován Log: {os.path.basename(file)} | Cmd: {cmd_angle}° | V_x: {real_velocity:.2f} m/s | Yaw: {real_yaw_rate:.3f} rad/s")

            results.append({
                'cmd_angle': cmd_angle,
                'real_yaw_rate': real_yaw_rate,
                'real_velocity': real_velocity
            })
        except Exception as e:
            print(f"Přeskočeno {os.path.basename(file)}: chybí sloupce nebo nelze načíst ({e})")

    if not results: return

    results_df = pd.DataFrame(results)

    # 1. Odstranění NaN hodnot (pokud se nějaké načetly)
    results_df = results_df.dropna()

    # 2. Filtrace anomálií: Ponecháme jen testy, kde byla rychlost mezi 0.5 a 1.5 m/s
    results_df = results_df[(results_df['real_velocity'] > 0.5) & (results_df['real_velocity'] < 1.5)]

    # 3. Sloučení pomocí mediánu (medián je mnohem odolnější vůči případným dalším extrémům než průměr)
    results_df = results_df.sort_values(by='cmd_angle').groupby('cmd_angle').median().reset_index()

    print("\nOptimalizuji úhly v simulaci (toto chvíli potrvá)...")
    sim_angles = []
    for idx, row in results_df.iterrows():
        opt_angle = optimize_steering_angle(row['real_yaw_rate'], row['real_velocity'])
        sim_angles.append(opt_angle)
        print(f"  Povel {row['cmd_angle']}° -> Simulační úhel {opt_angle:.2f}°")

    results_df['sim_angle'] = sim_angles

    coeffs = np.polyfit(results_df['cmd_angle'], results_df['sim_angle'], 1)
    
    print("\n=== VÝSLEDEK KALIBRACE ===")
    print(f"rovnice: Fyzický simulační úhel = {coeffs[0]:.4f} * povel + {coeffs[1]:.4f}")
    
    plt.figure(figsize=(8, 5))
    plt.scatter(results_df['cmd_angle'], results_df['sim_angle'], color='r', zorder=5)
    x_vals = np.linspace(0, max(results_df['cmd_angle']) + 5, 50)
    plt.plot(x_vals, np.poly1d(coeffs)(x_vals), 'b-', label=f'Fit: y = {coeffs[0]:.4f}x + {coeffs[1]:.4f}')
    
    plt.title('Kalibrace: Povel vs Fyzický úhel v simulaci')
    plt.xlabel('Povel z logu (stupně)')
    plt.ylabel('Hledaný fyzický úhel simulace (stupně)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(repo_root, 'Model_lib_and_sim/Identification/steering_calibration_result.png')
    plt.savefig(plot_path)
    print(f"Graf uložen do: {plot_path}")

if __name__ == '__main__':
    # main()..
    file_path = os.path.join(repo_root, 'log_files', '18.10.2026', 'rst_20_log_1792330050.csv')
    calibrate_log(file_path, 0.7448, -1.5184)