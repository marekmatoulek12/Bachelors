import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

class VehicleParams:
    front_wheel_radius: float = 0.033
    rear_wheel_radius: float = 0.033
    front_axle_len: float = 0.16
    rear_axle_len: float = 0.14
    mass: float = 2.5
    moment_of_inertia: float = 0.027
    gravity: float = 9.81
    
    front_track_width: float = 0.16
    rear_track_width: float = 0.16
    cg_height: float = 0.05

class BaseIdentification:
    def __init__(self, data: pd.DataFrame, start_from: float = 1.0) -> None:
        if start_from > 0.0:
            first_timestamp = data["timestamp"].iloc[0]
            data = data[data["timestamp"] >= (first_timestamp + start_from)]
            
        self.timestamp = data["timestamp"].to_numpy()
        self.cmd_rpm = data["cmd_rpm"].to_numpy()

        is_in_rad = True

        if "real_steer_rad" in data.columns:
            steer_column = "real_steer_rad"
            print("Artificially calibrated log")

        elif "cmd_steer_rad" in data.columns:
            steer_column = "cmd_steer_rad"
            print("Uncalibrated log")

        else:
            steer_column = "cmd_steer_deg"
            is_in_rad = False
            print("Calibrated IRL log")

        self.cmd_steer_rad = data[steer_column].to_numpy()
        if not is_in_rad:
            self.cmd_steer_rad = np.deg2rad(self.cmd_steer_rad)

        self.gps_x = data["gps_x"].to_numpy()
        self.gps_y = data["gps_y"].to_numpy()
        self.gps_yaw = data["gps_yaw"].to_numpy()
        self.vx = data["vx"].to_numpy()
        self.vy = data["vy"].to_numpy()
        self.yaw_rate = data["yaw_rate"].to_numpy()
        self.accel_x = data["accel_x"].to_numpy()
        self.accel_y = data["accel_y"].to_numpy()
        self.rl_rpm = data["rl_rpm"].to_numpy()
        self.rr_rpm = data["rr_rpm"].to_numpy()
        self.fl_rpm = data["fl_rpm"].to_numpy()
        self.fr_rpm = data["fr_rpm"].to_numpy()

        self.N = len(self.timestamp)
        self.car = VehicleParams()

    def prepare_data(self):
        dt = np.median(np.diff(self.timestamp))
        fs = 1.0 / dt if dt > 0 else 100.0
        cutoff_hz = 5

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

    @staticmethod
    def rpm2rad_s(rpm: float | np.ndarray) -> float | np.ndarray:
        return rpm * (2.0 * np.pi / 60.0)

class SingleTrackIdentification(BaseIdentification):
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

class DualTrackIdentification(BaseIdentification):
    def slip_variables_calculation(self):
        vx = self.vx_filtered
        vy = self.vy_filtered
        yaw_rate = self.yaw_rate_filtered
        
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
        
        Fz_rear_axle = (Fz_total * (self.car.front_axle_len / L) + 
                        self.car.mass * self.accel_x_filtered * (self.car.cg_height / L))
        
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

        self.Fyf_koeff = self.Fyf / (Fz_total - Fz_rear_axle)
        self.Fy_rl_koeff = self.Fy_rl / self.Fz_rl
        self.Fy_rr_koeff = self.Fy_rr / self.Fz_rr
        self.Fx_rl_koeff = self.Fx_rl / self.Fz_rl
        self.Fx_rr_koeff = self.Fx_rr / self.Fz_rr

class IdentificationOrchestrator:
    def __init__(self, target_dir: Path):
        self.target_dir = target_dir
        self.all_st_data = []
        self.all_dt_data = []
        self.all_raw_objects = []

    @staticmethod
    def _parse_log_params(filename: str) -> dict[str, float]:
        """Extrahuje parametry testu z názvu souboru (podporuje csst_ i starší rst_)."""
        def parse_val(token: str) -> float:
            return float(token.replace('_', '-'))

        # Nový formát: csst_<target_steer>_<steering_rate>_<starting_velocity>_<acceleration>
        csst_match = re.search(
            r'csst_([-_]?\d+)_([-_]?\d+)_([-_]?\d+)_([-_]?\d+)',
            filename,
        )
        if csst_match:
            return {
                'target_steer_deg': parse_val(csst_match.group(1)),
                'steering_rate': parse_val(csst_match.group(2)),
                'starting_velocity': parse_val(csst_match.group(3)),
                'acceleration': parse_val(csst_match.group(4)),
            }

        # Zpětná kompatibilita pro starší logy: rst_<target_steer>_<acceleration>
        rst_match = re.search(r'rst_([-_]?\d+)(?:_([-_]?\d+))?', filename)
        if rst_match:
            return {
                'target_steer_deg': parse_val(rst_match.group(1)),
                'steering_rate': 0.0,
                'starting_velocity': 0.0,
                'acceleration': parse_val(rst_match.group(2)) if rst_match.group(2) else 0.0,
            }

        return {
            'target_steer_deg': 0.0,
            'steering_rate': 0.0,
            'starting_velocity': 0.0,
            'acceleration': 0.0,
        }

    def load_and_process_logs(self):
        search_pattern = str(self.target_dir / '*.csv')
        log_files = glob.glob(search_pattern)
        
        if not log_files:
            print(f"Nenalezeny žádné kalibrované logy v: {self.target_dir}")
            return
            
        print(f"Orchestrátor: Nalezeno {len(log_files)} souborů. Zpracovávám pro ST a DT modely...")

        for file in log_files:
            try:
                params = self._parse_log_params(Path(file).name)
                
                # Jednotlivé proměnné pro přímé použití
                target_steer_deg = params['target_steer_deg']
                steering_rate = params['steering_rate']
                starting_velocity = params['starting_velocity']
                acceleration = params['acceleration']

                df = pd.read_csv(file)
                if len(df) < 50:
                    continue

                # Single-Track Processing
                st_handler = SingleTrackIdentification(df, start_from=2.0)
                st_handler.prepare_data()
                st_handler.slip_variables_calculation()
                st_handler.wheel_forces_calculation()
                
                # Uložení celého slovníku parametrů místo samotného úhlu
                self.all_raw_objects.append((params, st_handler))
                
                st_log_data = {
                    'cmd_angle': target_steer_deg,  # zachováno kvůli kompatibilitě se stávajícími grafy
                    **params,
                    'alpha_f': st_handler.alpha_f,
                    'Fyf_koeff': st_handler.Fyf_koeff,
                    'alpha_r': st_handler.alpha_r,
                    'Fyr_koeff': st_handler.Fyr_koeff,
                    'lamda_r': st_handler.lamda_r,
                    'Fxr_koeff': st_handler.Fxr_koeff,
                }
                self.all_st_data.append(st_log_data)
                
                # Dual-Track Processing
                dt_handler = DualTrackIdentification(df, start_from=2.0)
                dt_handler.prepare_data()
                dt_handler.slip_variables_calculation()
                dt_handler.wheel_forces_calculation()
                
                dt_log_data = {
                    'cmd_angle': target_steer_deg,
                    **params,
                    'alpha_f': dt_handler.alpha_f,
                    'Fyf_koeff': dt_handler.Fyf_koeff,
                    'alpha_r': dt_handler.alpha_r,
                    'Fy_rl_koeff': dt_handler.Fy_rl_koeff,
                    'Fy_rr_koeff': dt_handler.Fy_rr_koeff,
                    'lamda_rl': dt_handler.lamda_rl,
                    'lamda_rr': dt_handler.lamda_rr,
                    'Fx_rl_koeff': dt_handler.Fx_rl_koeff,
                    'Fx_rr_koeff': dt_handler.Fx_rr_koeff,
                }
                self.all_dt_data.append(dt_log_data)
                
            except Exception as e:
                print(f"Chyba při zpracování {file}: {e}")

    def _get_color_map_info(self, color_key: str = "target_steer_deg", use_abs: bool = True):
        transform = abs if use_abs else (lambda x: x)
        values = [transform(data[color_key]) for data in self.all_st_data if color_key in data]
        min_val, max_val = (min(values), max(values)) if values else (0.0, 20.0)
        cmap = plt.get_cmap('viridis')

        def get_color(val: float):
            v = transform(val)
            if max_val == min_val:
                norm_val = 0.5
            else:
                norm_val = (v - min_val) / (max_val - min_val)
            return cmap(norm_val)

        return cmap, min_val, max_val, get_color

    def plot_aggregated_raw_data(
        self,
        arrow_step: int = 2,
        color_key: str = "target_steer_deg",
        use_abs: bool = True,
    ):
        if not self.all_raw_objects:
            print("Žádná RAW data k vykreslení.")
            return

        labels = {
            "target_steer_deg": "Cílový úhel řízení [°]",
            "steering_rate": "Rychlost natáčení [°/s]",
            "starting_velocity": "Počáteční rychlost [m/s]",
            "acceleration": "Zrychlení [m/s²]",
        }
        prefix = "Absolutní: " if use_abs else ""
        cbar_label = f"{prefix}{labels.get(color_key, color_key)}"

        cmap, min_val, max_val, get_color = self._get_color_map_info(
            color_key=color_key, use_abs=use_abs
        )

        # --- 1. Graf naměřených vs filtrovaných hodnot (bez přejezdových čar) ---
        fig, axes = plt.subplots(6, 1, sharex=True, figsize=(12, 14))
        (yaw_rate_ax, yaw_rate_acc_ax, accel_x_ax, accel_y_ax, vx_ax, vy_ax) = axes
        fig.suptitle("Agregovaná Raw vs Filtered Data", fontsize=14)

        for params, handler in self.all_raw_objects:
            time = handler.timestamp - handler.timestamp[0]
            c = get_color(params[color_key])

            yaw_rate_ax.scatter(time, handler.yaw_rate_filtered, s=1, color=c)
            yaw_rate_acc_ax.scatter(time, handler.yaw_accel_filtered, s=1, color=c)
            accel_x_ax.scatter(time, handler.accel_x_filtered, s=1, color=c)
            accel_y_ax.scatter(time, handler.accel_y_filtered, s=1, color=c)
            vx_ax.scatter(time, handler.vx_filtered, s=1, color=c)
            vy_ax.scatter(time, handler.vy_filtered, s=1, color=c)

        yaw_rate_ax.set_ylabel("Yaw rate")
        yaw_rate_ax.grid(True)
        yaw_rate_acc_ax.set_ylabel("Yaw accel")
        yaw_rate_acc_ax.grid(True)
        accel_x_ax.set_ylabel("Accel x")
        accel_x_ax.grid(True)
        accel_y_ax.set_ylabel("Accel y")
        accel_y_ax.grid(True)
        vx_ax.set_ylabel("Vx")
        vx_ax.grid(True)
        vy_ax.set_ylabel("Vy")
        vy_ax.set_xlabel("Time [s]")
        vy_ax.grid(True)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_val, vmax=max_val))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.02)
        cbar.set_label(f"Parametr: {color_key}")

        fig.tight_layout()

        # --- 2. Graf GPS trajektorie a hlaviček ---
        fig_gps, gps_ax = plt.subplots(figsize=(10, 8))
        fig_gps.suptitle("Agregovaná GPS Trajectory and Heading", fontsize=14)

        for params, handler in self.all_raw_objects:
            c = get_color(params[color_key])

            gps_ax.scatter(handler.gps_x, handler.gps_y, s=2, color=c, alpha=0.5)

            indices = np.arange(0, len(handler.gps_yaw), arrow_step)
            gps_ax.quiver(
                handler.gps_x[indices],
                handler.gps_y[indices],
                np.cos(handler.gps_yaw[indices]),
                np.sin(handler.gps_yaw[indices]),
                color=c,
                scale=30,
                width=0.0035,
                zorder=5,
                alpha=0.6,
            )

        gps_ax.set_xlabel("x [m]")
        gps_ax.set_ylabel("y [m]")
        gps_ax.grid(True)
        gps_ax.axis("equal")

        cbar_gps = fig_gps.colorbar(sm, ax=gps_ax)
        cbar_gps.set_label(f"Parametr: {color_key}")

        fig_gps.tight_layout()

    def plot_aggregated_st_models(self, use_abs=True):
        if not self.all_st_data:
            print("Žádná ST data k vykreslení.")
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Single-Track: Agregovaná identifikace pneumatik", fontsize=14)

        cmap, min_angle, max_angle, get_color = self._get_color_map_info(use_abs=use_abs)

        for data in self.all_st_data:
            c = get_color(data['cmd_angle'])
            axes[0].scatter(data['alpha_f'], data['Fyf_koeff'], s=2, alpha=0.4, color=c)
            axes[1].scatter(data['alpha_r'], data['Fyr_koeff'], s=2, alpha=0.4, color=c)
            axes[2].scatter(data['lamda_r'], data['Fxr_koeff'], s=2, alpha=0.4, color=c)

        titles = ["Přední náprava: Fyf vs αf", "Zadní náprava: Fyr vs αr", "Zadní náprava: Fxr vs λ_avg"]
        x_labels = ["Skluzový úhel αf [rad]", "Skluzový úhel αr [rad]", "Průměrný skluz λ_avg [-]"]
        y_labels = ["Laterální síla Fyf koeff [-]", "Laterální síla Fyr koeff [-]", "Podélná síla Fxr koeff [-]"]

        for i, ax in enumerate(axes):
            ax.set_xlim([-1.0, 1.0])
            ax.set_ylim([-2.0, 2.0])
            ax.set_title(titles[i])
            ax.set_xlabel(x_labels[i])
            ax.set_ylabel(y_labels[i])
            ax.grid(True)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_angle, vmax=max_angle))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.02)
        cbar.set_label("Absolutní povel řízení [°]")

        fig.tight_layout()

    def plot_aggregated_dt_models(self, use_abs=True):
        if not self.all_dt_data:
            print("Žádná DT data k vykreslení.")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Dual-Track: Agregovaná identifikace pneumatik levého a pravého kola", fontsize=16)

        cmap, min_angle, max_angle, get_color = self._get_color_map_info(use_abs=use_abs)

        # Vlevo dole chceme prazdno - vypneme osu
        axes[1, 0].axis('off')

        for data in self.all_dt_data:
            c = get_color(data['cmd_angle'])
            
            # Predni naprava vlevo nahore
            axes[0, 0].scatter(data['alpha_f'], data['Fyf_koeff'], s=2, alpha=0.4, color=c)
            
            # Zadni naprava LEVE uprostred nahore
            axes[0, 1].scatter(data['alpha_r'], data['Fy_rl_koeff'], s=2, alpha=0.4, color=c)
            
            # Zadni naprava PRAVE uprostred dole
            axes[1, 1].scatter(data['alpha_r'], data['Fy_rr_koeff'], s=2, alpha=0.4, color=c)
            
            # Podelna sila LEVE vpravo nahore
            axes[0, 2].scatter(data['lamda_rl'], data['Fx_rl_koeff'], s=2, alpha=0.4, color=c)
            
            # Podelna sila PRAVE vpravo dole
            axes[1, 2].scatter(data['lamda_rr'], data['Fx_rr_koeff'], s=2, alpha=0.4, color=c)

        # Nastaveni pro kazdy graf
        axes[0, 0].set_xlim([-1.0, 1.0])
        axes[0, 0].set_ylim([-2.0, 2.0])
        axes[0, 0].set_title("Přední náprava: Fyf vs αf")
        axes[0, 0].set_xlabel("Skluzový úhel αf [rad]")
        axes[0, 0].set_ylabel("Laterální síla Fyf koeff [-]")
        axes[0, 0].grid(True)

        axes[0, 1].set_xlim([-1.0, 1.0])
        axes[0, 1].set_ylim([-2.0, 2.0])
        axes[0, 1].set_title("Zadní LEVÉ kolo: Fy_rl vs αr")
        axes[0, 1].set_xlabel("Skluzový úhel αr [rad]")
        axes[0, 1].set_ylabel("Laterální síla Fy_rl koeff [-]")
        axes[0, 1].grid(True)

        axes[1, 1].set_xlim([-1.0, 1.0])
        axes[1, 1].set_ylim([-2.0, 2.0])
        axes[1, 1].set_title("Zadní PRAVÉ kolo: Fy_rr vs αr")
        axes[1, 1].set_xlabel("Skluzový úhel αr [rad]")
        axes[1, 1].set_ylabel("Laterální síla Fy_rr koeff [-]")
        axes[1, 1].grid(True)
        
        axes[0, 2].set_xlim([-1.0, 1.0])
        axes[0, 2].set_ylim([-2.0, 2.0])
        axes[0, 2].set_title("Zadní LEVÉ kolo: Fx_rl vs λ_rl")
        axes[0, 2].set_xlabel("Podélný skluz λ_rl [-]")
        axes[0, 2].set_ylabel("Podélná síla Fx_rl koeff [-]")
        axes[0, 2].grid(True)

        axes[1, 2].set_xlim([-1.0, 1.0])
        axes[1, 2].set_ylim([-2.0, 2.0])
        axes[1, 2].set_title("Zadní PRAVÉ kolo: Fx_rr vs λ_rr")
        axes[1, 2].set_xlabel("Podélný skluz λ_rr [-]")
        axes[1, 2].set_ylabel("Podélná síla Fx_rr koeff [-]")
        axes[1, 2].grid(True)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_angle, vmax=max_angle))
        sm.set_array([])
        
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Absolutní povel řízení [°]")

        fig.subplots_adjust(wspace=0.3, hspace=0.3)

    def plot_aggregated_combined_slip(self):
        if not self.all_dt_data:
            print("Žádná DT data k vykreslení kombinovaného skluzu.")
            return

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle("Dual-Track: Agregovaná 3D identifikace (Kombinovaný skluz)", fontsize=16)

        cmap, min_angle, max_angle, get_color = self._get_color_map_info()

        # 1. Levé kolo - Laterální síla
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.set_title("Levé kolo: Fy_rl koeff")
        ax1.set_xlabel("Skluzový úhel α_r [rad]")
        ax1.set_ylabel("Podélný skluz λ_rl [-]")
        ax1.set_zlabel("Fy_rl koeff [-]")
        
        # 2. Pravé kolo - Laterální síla
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        ax2.set_title("Pravé kolo: Fy_rr koeff")
        ax2.set_xlabel("Skluzový úhel α_r [rad]")
        ax2.set_ylabel("Podélný skluz λ_rr [-]")
        ax2.set_zlabel("Fy_rr koeff [-]")

        # 3. Levé kolo - Podélná síla
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        ax3.set_title("Levé kolo: Fx_rl koeff")
        ax3.set_xlabel("Skluzový úhel α_r [rad]")
        ax3.set_ylabel("Podélný skluz λ_rl [-]")
        ax3.set_zlabel("Fx_rl koeff [-]")

        # 4. Pravé kolo - Podélná síla
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        ax4.set_title("Pravé kolo: Fx_rr koeff")
        ax4.set_xlabel("Skluzový úhel α_r [rad]")
        ax4.set_ylabel("Podélný skluz λ_rr [-]")
        ax4.set_zlabel("Fx_rr koeff [-]")

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlim([-1.0, 1.0])
            ax.set_ylim([-1.0, 1.0])
            ax.set_zlim([-2.0, 2.0])

        for data in self.all_dt_data:
            c = get_color(data['cmd_angle'])
            
            ax1.scatter(data['alpha_r'], data['lamda_rl'], data['Fy_rl_koeff'], s=5, alpha=0.5, color=c)
            ax2.scatter(data['alpha_r'], data['lamda_rr'], data['Fy_rr_koeff'], s=5, alpha=0.5, color=c)
            ax3.scatter(data['alpha_r'], data['lamda_rl'], data['Fx_rl_koeff'], s=5, alpha=0.5, color=c)
            ax4.scatter(data['alpha_r'], data['lamda_rr'], data['Fx_rr_koeff'], s=5, alpha=0.5, color=c)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_angle, vmax=max_angle))
        sm.set_array([])
        
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Absolutní povel řízení [°]")

        fig.subplots_adjust(wspace=0.2, hspace=0.2)

def main(target: Path = ""):
    if target == "":
        parser = argparse.ArgumentParser(description="Load and aggregate multiple vehicle test logs.")
        parser.add_argument("target_dir", type=Path, help="Path to the directory containing calibrated CSV logs")
        args = parser.parse_args()
        target_dir = args.target_dir
    else:
        target_dir = target
    
    orchestrator = IdentificationOrchestrator(target_dir)
    orchestrator.load_and_process_logs()
    
    orchestrator.plot_aggregated_raw_data(use_abs=True)
    orchestrator.plot_aggregated_st_models(use_abs=True)
    # orchestrator.plot_aggregated_dt_models()
    # orchestrator.plot_aggregated_combined_slip()
    
    plt.show()

if __name__ == '__main__':
    target = Path("Calibrated")
    main(target)
