import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.signal import sosfiltfilt, butter
import matplotlib.pyplot as plt

class VehicleParams:
    front_wheel_radius: float = 0.033
    rear_wheel_radius: float = 0.033
    front_axle_len: float = 0.16
    rear_axle_len: float = 0.14
    mass: float = 2.5
    moment_of_inertia: float = 0.027
    gravity: float = 9.81
    
    # Extra parametry pro Dual-Track model
    front_track_width: float = 0.16
    rear_track_width: float = 0.16
    cg_height: float = 0.05  # Výška těžiště


class BaseIdentification:
    """Základní třída zajišťující načtení a přípravu (filtraci) dat."""
    
    def __init__(self, data: pd.DataFrame, start_from: float = 1.0) -> None:
        # Oříznutí dat podle relativního času (odřízne začátek)
        if start_from > 0.0:
            first_timestamp = data["timestamp"].iloc[0]
            # Ponecháme pouze řádky, jejichž timestamp je větší nebo roven času startu + posunu
            data = data[data["timestamp"] >= (first_timestamp + start_from)]
            
        self.timestamp = data["timestamp"].to_numpy()
        self.cmd_rpm = data["cmd_rpm"].to_numpy()
        self.cmd_steer_rad = data["cmd_steer_rad"].to_numpy() # cmd_steer_rad
        self.gps_x = data["gps_x"].to_numpy()
        self.gps_y = data["gps_y"].to_numpy()
        self.gps_yaw = data["gps_yaw"].to_numpy()
        self.vx = data["vx"].to_numpy()
        self.vy = data["vy"].to_numpy()
        self.yaw_rate = data["yaw_rate"].to_numpy()
        self.accel_x = data["accel_x"].to_numpy()
        self.accel_y = data["accel_y"].to_numpy()
        # self.accel_z = data["accel_z"].to_numpy()
        self.rl_rpm = data["rl_rpm"].to_numpy()
        self.rr_rpm = data["rr_rpm"].to_numpy()
        self.fl_rpm = data["fl_rpm"].to_numpy()
        self.fr_rpm = data["fr_rpm"].to_numpy()

        self.N = len(self.timestamp)
        self.car = VehicleParams()

    def prepare_data(self):
        dt = np.median(np.diff(self.timestamp))  # předpoklad: timestamp je v sekundách
        fs = 1.0 / dt
        cutoff_hz = 5  # příklad; dolaď podle rychlosti změn a šumu

        sos = butter(4, cutoff_hz, btype="lowpass", fs=fs, output="sos")
        
        self.yaw_rate_filtered = sosfiltfilt(sos=sos, x=self.yaw_rate)
        self.yaw_accel = np.gradient(self.yaw_rate_filtered, self.timestamp)
        self.yaw_accel_filtered = sosfiltfilt(sos=sos, x=self.yaw_accel)

        self.accel_x_filtered = sosfiltfilt(sos=sos, x=self.accel_x)
        self.accel_y_filtered = sosfiltfilt(sos=sos, x=self.accel_y)

        self.vx_filtered = sosfiltfilt(sos=sos, x=self.vx)
        self.vy_filtered = sosfiltfilt(sos=sos, x=self.vy)

        self.rl_rpm_filtered = sosfiltfilt(sos=sos, x=self.rl_rpm)
        self.rr_rpm_filtered = sosfiltfilt(sos=sos, x=self.rr_rpm)

    def plot_prepared_data(self, title_prefix="", arrow_step=2):
        if not hasattr(self, "yaw_rate_filtered"):
            raise AttributeError("Prepare the data first")

        time = self.timestamp - self.timestamp[0]
        fig, (yaw_rate_ax, yaw_rate_acc_ax, accel_x_ax, accel_y_ax, vx_ax, vy_ax) = plt.subplots(
            6, 1, sharex=True
        )
        fig.suptitle(f"{title_prefix} Raw vs Filtered Data", fontsize=14)
        
        yaw_rate_ax.plot(time, self.yaw_rate, label="Raw", alpha=0.5)
        yaw_rate_ax.plot(time, self.yaw_rate_filtered, label="Filtered")
        yaw_rate_ax.set_ylabel("Yaw rate")
        yaw_rate_ax.grid(True)
        yaw_rate_ax.legend()

        yaw_rate_acc_ax.plot(time, self.yaw_accel, label="Raw", alpha=0.5)
        yaw_rate_acc_ax.plot(time, self.yaw_accel_filtered, label="Filtered")
        yaw_rate_acc_ax.set_ylabel("Yaw accel")
        yaw_rate_acc_ax.grid(True)
        yaw_rate_acc_ax.legend()

        accel_x_ax.plot(time, self.accel_x, label="Raw", alpha=0.5)
        accel_x_ax.plot(time, self.accel_x_filtered, label="Filtered")
        accel_x_ax.set_ylabel("Accel x")
        accel_x_ax.grid(True)
        
        accel_y_ax.plot(time, self.accel_y, label="Raw", alpha=0.5)
        accel_y_ax.plot(time, self.accel_y_filtered, label="Filtered")
        accel_y_ax.set_ylabel("Accel y")
        accel_y_ax.grid(True)

        vx_ax.plot(time, self.vx, label="Raw", alpha=0.5)
        vx_ax.plot(time, self.vx_filtered, label="Filtered")
        vx_ax.set_ylabel("Vx")
        vx_ax.grid(True)
        vx_ax.legend()

        vy_ax.plot(time, self.vy, label="Raw", alpha=0.5)
        vy_ax.plot(time, self.vy_filtered, label="Filtered")
        vy_ax.set_ylabel("Vy")
        vy_ax.set_xlabel("Time [s]")
        vy_ax.grid(True)
        vy_ax.legend()

        fig.tight_layout()

        fig_gps, gps_ax = plt.subplots()
        gps_ax.plot(self.gps_x, self.gps_y, label="GPS trajectory", color="blue")

        indices = np.arange(0, len(self.gps_yaw), arrow_step)
        gps_ax.quiver(
            self.gps_x[indices],
            self.gps_y[indices],
            np.cos(self.gps_yaw[indices]),
            np.sin(self.gps_yaw[indices]),
            color="red",
            scale=30,
            width=0.0035,
            label="GPS yaw",
            zorder=5,
        )
        gps_ax.set_title(f"{title_prefix} GPS Trajectory and Heading")
        gps_ax.set_xlabel("x [m]")
        gps_ax.set_ylabel("y [m]")
        gps_ax.grid(True)
        gps_ax.axis("equal")
        gps_ax.legend()
        fig_gps.tight_layout()

        fig_rpm, rpm_ax = plt.subplots()
        rpm_ax.plot(time, self.fl_rpm, label="Front left")
        rpm_ax.plot(time, self.fr_rpm, label="Front right")
        rpm_ax.plot(time, self.rl_rpm, label="Rear left")
        rpm_ax.plot(time, self.rr_rpm, label="Rear right")
        rpm_ax.set_title(f"{title_prefix} Wheel RPM")
        rpm_ax.set_xlabel("Time [s]")
        rpm_ax.set_ylabel("RPM")
        rpm_ax.grid(True)
        rpm_ax.legend()
        fig_rpm.tight_layout()

    def steer_yaw_csv(self, output_path: str = "steer_yaw_analysis.csv"):
        # Složení sloupců za sebe do slovníku
        data_dict = {
            "timestamp": self.timestamp,
            "cmd_steer_rad": self.cmd_steer_rad,  # případně self.cmd_steer_rad podle toho, co používáte
            "yaw_rate": self.yaw_rate,
            "yaw_rate_filtered": self.yaw_rate_filtered,
        }

        # Vytvoření Pandas DataFrame
        df_to_save = pd.DataFrame(data_dict)

        # Uložení do CSV formátu (bez indexu řádků)
        df_to_save.to_csv(output_path, index=False)
        print(f"Data byla úspěšně uložena do: {output_path}")

    @staticmethod
    def rpm2rad_s(rpm: float | np.ndarray) -> float | np.ndarray:
        return rpm * (2.0 * np.pi / 60.0)

    @staticmethod
    def cmd2real_steer(cmd: float) -> float:
        pass    

class SingleTrackIdentification(BaseIdentification):
    """Původní bicyklový model (Single-Track) - zjednodušený přístup."""
    
    def slip_variables_calculation(self):
        vx = self.vx_filtered
        vy = self.vy_filtered
        yaw_rate = self.yaw_rate_filtered
        
        rl_rad_s = self.rpm2rad_s(self.rl_rpm_filtered)
        rr_rad_s = self.rpm2rad_s(self.rr_rpm_filtered)

        r_rad_s = (rl_rad_s + rr_rad_s) / 2
        v_wheel_r = self.car.rear_wheel_radius * r_rad_s

        vx_min = 0.1
        denom_r = np.maximum(np.maximum(v_wheel_r, vx), vx_min)

        self.lamda_r = (v_wheel_r - vx) / denom_r

        df = self.cmd_steer_rad
        s_df, c_df = np.sin(df), np.cos(df)
        
        self.alpha_f = -np.arctan2(
            (vy + self.car.front_axle_len * yaw_rate) * c_df - vx * s_df,
            np.abs((vy + self.car.front_axle_len * yaw_rate) * s_df + vx * c_df)
        )
        self.alpha_r = -np.arctan2((vy - self.car.rear_axle_len * yaw_rate), np.abs(vx))

    def wheel_forces_calculation(self):
        df = self.cmd_steer_rad
        L = self.car.front_axle_len + self.car.rear_axle_len
        
        self.Fyf = (self.car.mass * self.accel_y_filtered * self.car.rear_axle_len 
                    + self.car.moment_of_inertia * self.yaw_accel_filtered) / (np.cos(df) * L)
        self.Fyr = (self.car.mass * self.accel_y_filtered * self.car.front_axle_len 
                    - self.car.moment_of_inertia * self.yaw_accel_filtered) / L
        self.Fxr = self.car.mass * self.accel_x_filtered + self.Fyf * np.sin(df)

        g = self.car.gravity
        Fz_total = self.car.mass * g
        Fz_rear_axle = Fz_total * (self.car.front_axle_len / L)
        Fz_front_axle = Fz_total - Fz_rear_axle


        self.Fyf_koeff = self.Fyf / Fz_front_axle
        self.Fyr_koeff = self.Fyr / Fz_rear_axle
        self.Fxr_koeff = self.Fxr / Fz_rear_axle

    def plot_wheel_forces(self):
        time = self.timestamp - self.timestamp[0]
        fig, (fyf_ax, fxr_ax, fyr_ax) = plt.subplots(3, 1, sharex=True)
        fig.suptitle("Single-Track: Celkové síly na nápravách", fontsize=14)

        fyf_ax.plot(time, self.Fyf, color="blue")
        fyf_ax.set_ylabel("Fyf [N]")
        fyf_ax.grid(True)

        fxr_ax.plot(time, self.Fxr, color="green")
        fxr_ax.set_ylabel("Fxr [N]")
        fxr_ax.grid(True)

        fyr_ax.plot(time, self.Fyr, color="red")
        fyr_ax.set_ylabel("Fyr [N]")
        fyr_ax.set_xlabel("Time [s]")
        fyr_ax.grid(True)
        fig.tight_layout()

    def plot_slip_variables(self):
        time = self.timestamp - self.timestamp[0]
        fig, (alpha_f_ax, alpha_r_ax, lambda_ax) = plt.subplots(3, 1, sharex=True)
        fig.suptitle("Single-Track: Úhly skluzu a Slip Ratios", fontsize=14)

        alpha_f_ax.plot(time, self.alpha_f, label="alpha_f")
        alpha_f_ax.set_ylabel("alpha f [rad]")
        alpha_f_ax.grid(True)

        alpha_r_ax.plot(time, self.alpha_r, label="alpha_r")
        alpha_r_ax.set_ylabel("alpha r [rad]")
        alpha_r_ax.grid(True)

        lambda_ax.plot(time, self.lamda_r, label="lambda_rl")
        lambda_ax.set_ylabel("Slip ratio")
        lambda_ax.set_xlabel("Time [s]")
        lambda_ax.legend()
        lambda_ax.grid(True)
        fig.tight_layout()

    def plot_tire_models(self):
        if not all(hasattr(self, name) for name in ("alpha_f", "alpha_r", "lamda_r", "Fyf_koeff", "Fyr_koeff", "Fxr_koeff")):
            raise AttributeError("Calculate slip variables and wheel forces first.")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Single-Track: Identifikace pneumatik (inverzní dynamika)", fontsize=14)

        # 1. Přední náprava: Laterální síla
        axes[0].scatter(self.alpha_f, self.Fyf_koeff, s=5, alpha=0.3, color='blue')
        axes[0].set_xlim([-1.0, 1.0])
        axes[0].set_title("Přední náprava: Fyf vs αf")
        axes[0].set_xlabel("Skluzový úhel αf [rad]")
        axes[0].set_ylabel("Laterální síla Fyf koeff [-]")
        axes[0].grid(True)

        # 2. Zadní náprava: Laterální síla
        axes[1].scatter(self.alpha_r, self.Fyr_koeff, s=5, alpha=0.3, color='red')
        axes[1].set_xlim([-1.0, 1.0])
        axes[1].set_title("Zadní náprava: Fyr vs αr")
        axes[1].set_xlabel("Skluzový úhel αr [rad]")
        axes[1].set_ylabel("Laterální síla Fyr koeff [-]")
        axes[1].grid(True)

        # 3. Zadní náprava: Podélná síla
        axes[2].scatter(self.lamda_r, self.Fxr_koeff, s=5, alpha=0.3, color='green')
        axes[2].set_xlim([-1.0, 1.0])
        axes[2].set_title("Zadní náprava: Fxr vs λ_avg")
        axes[2].set_xlabel("Průměrný skluz λ_avg [-]")
        axes[2].set_ylabel("Podélná síla Fxr koeff [-]")
        axes[2].grid(True)

        fig.tight_layout()

    def plot_combined_slip_forces(self):
        if not all(hasattr(self, name) for name in ("alpha_f", "alpha_r", "lamda_r", "Fyf_koeff", "Fyr_koeff", "Fxr_koeff")):
            raise AttributeError("Calculate slip variables and wheel forces first.")
        
        # Vytvoření 3D figure
        fig = plt.figure(figsize=(8, 12))
        ax1 = fig.add_subplot(2, 1, 1, projection='3d')
        sc1 = ax1.scatter(self.alpha_r, self.lamda_r, self.Fyr_koeff, c=self.Fyr_koeff, cmap='viridis', s=10, alpha=0.5)
        ax1.set_title("3D Identifikace pneumatiky (Kombinovaný skluz)")
        ax1.set_xlabel("Skluzový úhel α [rad]")
        ax1.set_ylabel("Podélný skluz λ [-]")
        ax1.set_zlabel("Fyr koeff [-]")
        # fig.colorbar(sc1, ax=ax1, label="Síla μ")

        ax2 = fig.add_subplot(2, 1, 2, projection='3d')
        sc2 = ax2.scatter(self.alpha_r, self.lamda_r, self.Fxr_koeff, c=self.Fxr_koeff, cmap='viridis', s=10, alpha=0.5)
        ax2.set_title("3D Identifikace pneumatiky (Kombinovaný skluz)")
        ax2.set_xlabel("Skluzový úhel α [rad]")
        ax2.set_ylabel("Podélný skluz λ [-]")
        ax2.set_zlabel("Fxr koeff [-]")
        # fig.colorbar(sc2, ax=ax2, label="Síla μ")

        fig.tight_layout()

class DualTrackIdentification(BaseIdentification):
    """Rozšířený dvoustopý model s Torque Vectoringem a přenosem hmotnosti."""
    
    def slip_variables_calculation(self):
        vx = self.vx_filtered
        vy = self.vy_filtered
        yaw_rate = self.yaw_rate_filtered
        
        # Korekce rychlosti pro vnější a vnitřní kolo
        half_track = self.car.rear_track_width / 2.0
        vx_rl = vx - yaw_rate * half_track
        vx_rr = vx + yaw_rate * half_track
        
        rl_rad_s = self.rpm2rad_s(self.rl_rpm_filtered)
        rr_rad_s = self.rpm2rad_s(self.rr_rpm_filtered)
        v_wheel_rl = rl_rad_s * self.car.rear_wheel_radius
        v_wheel_rr = rr_rad_s * self.car.rear_wheel_radius

        vx_min = 0.1
        denom_rl = np.maximum(np.maximum(v_wheel_rl, vx_rl), vx_min)
        denom_rr = np.maximum(np.maximum(v_wheel_rr, vx_rr), vx_min)

        self.lamda_rl = (v_wheel_rl - vx_rl) / denom_rl
        self.lamda_rr = (v_wheel_rr - vx_rr) / denom_rr

        df = self.cmd_steer_rad
        s_df, c_df = np.sin(df), np.cos(df)
        
        self.alpha_f = -np.arctan2(
            (vy + self.car.front_axle_len * yaw_rate) * c_df - vx * s_df,
            np.abs((vy + self.car.front_axle_len * yaw_rate) * s_df + vx * c_df)
        )
        self.alpha_r = -np.arctan2((vy - self.car.rear_axle_len * yaw_rate), np.abs(vx))

    def wheel_forces_calculation(self):
        df_rad = self.cmd_steer_rad
        L = self.car.front_axle_len + self.car.rear_axle_len
        
        self.Fyf = (self.car.mass * self.accel_y_filtered * self.car.rear_axle_len 
                    + self.car.moment_of_inertia * self.yaw_accel_filtered) / (np.cos(df_rad) * L)
        self.Fyr = (self.car.mass * self.accel_y_filtered * self.car.front_axle_len 
                    - self.car.moment_of_inertia * self.yaw_accel_filtered) / L
        self.Fxr = self.car.mass * self.accel_x_filtered + self.Fyf * np.sin(df_rad)

        g = self.car.gravity
        Fz_total = self.car.mass * g
        
        Fz_rear_axle = Fz_total * (self.car.front_axle_len / L) + \
                       self.car.mass * self.accel_x_filtered * (self.car.cg_height / L)
        
        delta_Fz_y_rear = Fz_rear_axle * (self.accel_y_filtered / g) * (self.car.cg_height / self.car.rear_track_width)

        self.Fz_rl = np.maximum((Fz_rear_axle / 2.0) - delta_Fz_y_rear, 0.001)
        self.Fz_rr = np.maximum((Fz_rear_axle / 2.0) + delta_Fz_y_rear, 0.001)
        sum_Fz_rear = self.Fz_rl + self.Fz_rr

        self.Fy_rl = self.Fyr * (self.Fz_rl / sum_Fz_rear)
        self.Fy_rr = self.Fyr * (self.Fz_rr / sum_Fz_rear)

        W_rl = np.abs(self.lamda_rl) * self.Fz_rl
        W_rr = np.abs(self.lamda_rr) * self.Fz_rr
        sum_W = np.maximum(W_rl + W_rr, 1e-6)
        
        self.Fx_rl = self.Fxr * (W_rl / sum_W)
        self.Fx_rr = self.Fxr * (W_rr / sum_W)

        # Fx by slo jeste odhadovat z momentove rovnice pro kola
        # - znam uhlove zrychleni -> derivace otacek, ktere mam
        # - nevim, jestli znam torque, ale jestli jsem schopny posilat pozadavek na proud, asi bych mel

        # normalizace sil pro omezeni sumu v pacejkovi
        self.Fyf_koeff = self.Fyf / (Fz_total - Fz_rear_axle)
        self.Fy_rl_koeff = self.Fy_rl / self.Fz_rl
        self.Fy_rr_koeff = self.Fy_rr / self.Fz_rr
        self.Fx_rl_koeff = self.Fx_rl / self.Fz_rl
        self.Fx_rr_koeff = self.Fx_rr / self.Fz_rr

    def plot_wheel_forces(self):
        time = self.timestamp - self.timestamp[0]
        fig, (fyf_ax, fxr_ax, fyr_ax) = plt.subplots(3, 1, sharex=True)
        fig.suptitle("Dual-Track: Rozložení sil na jednotlivá kola", fontsize=14)

        fyf_ax.plot(time, self.Fyf, color="blue", label="Fyf (Celá náprava)")
        fyf_ax.set_ylabel("Fyf [N]")
        fyf_ax.legend()
        fyf_ax.grid(True)

        fxr_ax.plot(time, self.Fx_rl, label="Fx_rl (Levé)")
        fxr_ax.plot(time, self.Fx_rr, label="Fx_rr (Pravé)")
        fxr_ax.plot(time, self.Fxr, label="Fxr (Suma)", linestyle="--", color="black", alpha=0.5)
        fxr_ax.set_ylabel("Fx [N]")
        fxr_ax.legend(loc="upper right")
        fxr_ax.grid(True)

        fyr_ax.plot(time, self.Fy_rl, label="Fy_rl (Levé)")
        fyr_ax.plot(time, self.Fy_rr, label="Fy_rr (Pravé)")
        fyr_ax.plot(time, self.Fyr, label="Fyr (Suma)", linestyle="--", color="black", alpha=0.5)
        fyr_ax.set_ylabel("Fy [N]")
        fyr_ax.set_xlabel("Time [s]")
        fyr_ax.legend(loc="upper right")
        fyr_ax.grid(True)
        fig.tight_layout()

    def plot_slip_variables(self):
        time = self.timestamp - self.timestamp[0]
        fig, (alpha_f_ax, alpha_r_ax, lambda_ax) = plt.subplots(3, 1, sharex=True)
        fig.suptitle("Dual-Track: Úhly skluzu a korigované Slip Ratios", fontsize=14)

        alpha_f_ax.plot(time, self.alpha_f, label="alpha_f")
        alpha_f_ax.set_ylabel("alpha f [rad]")
        alpha_f_ax.grid(True)

        alpha_r_ax.plot(time, self.alpha_r, label="alpha_r")
        alpha_r_ax.set_ylabel("alpha r [rad]")
        alpha_r_ax.grid(True)

        lambda_ax.plot(time, self.lamda_rl, label="lambda_rl (korigovaná rychlost)")
        lambda_ax.plot(time, self.lamda_rr, label="lambda_rr (korigovaná rychlost)")
        lambda_ax.set_ylabel("Slip ratio")
        lambda_ax.set_xlabel("Time [s]")
        lambda_ax.legend()
        lambda_ax.grid(True)
        fig.tight_layout()

    def plot_tire_models(self):
        if not all(hasattr(self, name) for name in ("alpha_f", "alpha_r", "lamda_rl", "lamda_rr", "Fyf_koeff", "Fy_rl_koeff", "Fy_rr_koeff", "Fx_rl_koeff", "Fx_rr_koeff")):
            raise AttributeError("Calculate slip variables and wheel forces first.")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Dual-Track: Identifikace pneumatik jednotlivých kol", fontsize=14)

        # 1. Přední náprava: Laterální síla (zde zůstává suma)
        axes[0].scatter(self.alpha_f, self.Fyf_koeff, s=5, alpha=0.3, color='blue')
        axes[0].set_xlim([-1.0, 1.0])
        axes[0].set_title("Přední náprava: Fyf vs αf")
        axes[0].set_xlabel("Skluzový úhel αf [rad]")
        axes[0].set_ylabel("Laterální síla Fyf koeff [-]")
        axes[0].grid(True)

        # 2. Zadní kola: Laterální síla
        axes[1].scatter(self.alpha_r, self.Fy_rl_koeff, s=5, alpha=0.3, color='orange', label='Levé kolo (Fy_rl)')
        axes[1].scatter(self.alpha_r, self.Fy_rr_koeff, s=5, alpha=0.3, color='purple', label='Pravé kolo (Fy_rr)')
        axes[1].set_xlim([-1.0, 1.0])
        axes[1].set_title("Zadní kola: Fy vs αr")
        axes[1].set_xlabel("Skluzový úhel αr [rad]")
        axes[1].set_ylabel("Laterální síla Fy koeff [-]")
        axes[1].legend()
        axes[1].grid(True)

        # 3. Zadní kola: Podélná síla s odděleným slip ratiem
        axes[2].scatter(self.lamda_rl, self.Fx_rl_koeff, s=5, alpha=0.3, color='orange', label='Levé kolo (Fx_rl)')
        axes[2].scatter(self.lamda_rr, self.Fx_rr_koeff, s=5, alpha=0.3, color='purple', label='Pravé kolo (Fx_rr)')
        axes[2].set_xlim([-1.0, 1.0])
        axes[2].set_title("Zadní kola: Fx vs λ")
        axes[2].set_xlabel("Podélný skluz λ [-]")
        axes[2].set_ylabel("Podélná síla Fx koeff [-]")
        axes[2].legend()
        axes[2].grid(True)

        fig.tight_layout()

    def plot_combined_slip_forces(self):
        required_attrs = (
            "alpha_r", "lamda_rl", "lamda_rr", 
            "Fy_rl_koeff", "Fy_rr_koeff", 
            "Fx_rl_koeff", "Fx_rr_koeff"
        )
        if not all(hasattr(self, name) for name in required_attrs):
            raise AttributeError("Calculate slip variables and wheel forces first.")

        # Vytvoření 3D figure s mřížkou 2x2
        fig = plt.figure(figsize=(14, 12))
        fig.suptitle("3D Identifikace pneumatik: Levé vs Pravé kolo (Kombinovaný skluz)", fontsize=16)

        # --- SLOUPEC 1: LEVÉ KOLO (rl) ---
        # 1x1: Laterální síla Fy_rl_koeff
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        sc1 = ax1.scatter(self.alpha_r, self.lamda_rl, self.Fy_rl_koeff, c=self.Fy_rl_koeff, cmap='viridis', s=10, alpha=0.5)
        ax1.set_title("Levé kolo: Fy_rl koeff")
        ax1.set_xlabel("Skluzový úhel α_r [rad]")
        ax1.set_ylabel("Podélný skluz λ_rl [-]")
        ax1.set_zlabel("Fy_rl koeff [-]")
        # fig.colorbar(sc1, ax=ax1, shrink=0.5, label="Hodnota")

        # 2x1: Podélná síla Fx_rl_koeff (pod 1x1)
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        sc3 = ax3.scatter(self.alpha_r, self.lamda_rl, self.Fx_rl_koeff, c=self.Fx_rl_koeff, cmap='viridis', s=10, alpha=0.5)
        ax3.set_title("Levé kolo: Fx_rl koeff")
        ax3.set_xlabel("Skluzový úhel α_r [rad]")
        ax3.set_ylabel("Podélný skluz λ_rl [-]")
        ax3.set_zlabel("Fx_rl koeff [-]")
        # fig.colorbar(sc3, ax=ax3, shrink=0.5, label="Hodnota")

        # --- SLOUPEC 2: PRAVÉ KOLO (rr) ---
        # 1x2: Laterální síla Fy_rr_koeff
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        sc2 = ax2.scatter(self.alpha_r, self.lamda_rr, self.Fy_rr_koeff, c=self.Fy_rr_koeff, cmap='viridis', s=10, alpha=0.5)
        ax2.set_title("Pravé kolo: Fy_rr koeff")
        ax2.set_xlabel("Skluzový úhel α_r [rad]")
        ax2.set_ylabel("Podélný skluz λ_rr [-]")
        ax2.set_zlabel("Fy_rr koeff [-]")
        # fig.colorbar(sc2, ax=ax2, shrink=0.5, label="Hodnota")

        # 2x2: Podélná síla Fx_rr_koeff (pod 1x2)
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        sc4 = ax4.scatter(self.alpha_r, self.lamda_rr, self.Fx_rr_koeff, c=self.Fx_rr_koeff, cmap='viridis', s=10, alpha=0.5)
        ax4.set_title("Pravé kolo: Fx_rr koeff")
        ax4.set_xlabel("Skluzový úhel α_r [rad]")
        ax4.set_ylabel("Podélný skluz λ_rr [-]")
        ax4.set_zlabel("Fx_rr koeff [-]")
        # fig.colorbar(sc4, ax4, shrink=0.5, label="Hodnota")

        fig.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Load a vehicle test log CSV.")
    parser.add_argument("csv_file", type=Path, help="Path to the CSV file to load")
    parser.add_argument("out_file", type=Path, help="Path to which you want to save steer/yaw comparison .csv")
    args = parser.parse_args()
    
    data = pd.read_csv(args.csv_file)
    
    # ---------------------------------------------------------
    # 1. Zpracování podle Single-Track (Bicyklový) Modelu
    # ---------------------------------------------------------
    print("--- Zpracování dat modelem Single-Track ---")
    st_handler = SingleTrackIdentification(data, start_from=1.0)
    st_handler.prepare_data()
    # st_handler.steer_yaw_csv(args.out_file)
    st_handler.plot_prepared_data(title_prefix="Single-Track") # Společné pro oba, stačí vykreslit jednou
    st_handler.slip_variables_calculation()
    st_handler.wheel_forces_calculation()
    
    st_handler.plot_slip_variables()
    # st_handler.plot_wheel_forces()
    st_handler.plot_tire_models()
    #st_handler.plot_combined_slip_forces()
    
    # # ---------------------------------------------------------
    # # 2. Zpracování podle Dual-Track Modelu
    # # ---------------------------------------------------------
    # print("\n--- Zpracování dat modelem Dual-Track ---")
    # dt_handler = DualTrackIdentification(data)
    # dt_handler.prepare_data()
    # dt_handler.slip_variables_calculation()
    # dt_handler.wheel_forces_calculation()
    
    # dt_handler.plot_slip_variables()
    # # dt_handler.plot_wheel_forces()
    # dt_handler.plot_tire_models()
    # dt_handler.plot_combined_slip_forces()

    # Zobrazení všech vygenerovaných grafů najednou
    plt.show()

if __name__ == "__main__":
    main()