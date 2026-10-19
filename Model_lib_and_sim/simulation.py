import numpy as np
from twintrack_model_modified import TwinTrackInput, TwinTrackModel, TwinTrackState, TwinTrackStateDerivatives
from twintrack_model_modified import RearWheelSpeedController, RearWheelSpeedControllerLockedDiff, PIDGains
from dataclasses import fields
import matplotlib.pyplot as plt
from math import sin, pi
from scipy.optimize import curve_fit
import pandas as pd
from pathlib import Path
import argparse
# model = TwinTrackModel()
# state = TwinTrackState()

# state_input = TwinTrackInput(
#     drive_torque_rl= 0.1,
#     drive_torque_rr= 0.1,
#     brake_torque_rl= 0.0,
#     brake_torque_rr= 0.0,
#     steering_angle_fl= 0.1,
#     steering_angle_fr= 0.1,
# )

class Logger:
    def __init__(self, sim_step: float) -> None:
        self.time = []
        self.sim_step = sim_step

        self.x = []
        self.y = []
        self.yaw = []

        self.vx = []
        self.vy = []
        self.yaw_rate = []

        self.ax = []
        self.ay = []
        self.yaw_rate_dot = []

        self.wheels = { 
            "fl" : {
                "omega":     [],
                "omega_dot": [],
                "fx":           [],
                "fy":           [],
                "normal_force": [],
                "slip_ratio":   [],
                "slip_angle":   []
            },
            "fr" : {
                "omega":     [],
                "omega_dot": [],
                "fx":           [],
                "fy":           [],
                "normal_force": [],
                "slip_ratio":   [],
                "slip_angle":   []
            },
            "rl" : {
                "omega":     [],
                "omega_dot": [],
                "fx":           [],
                "fy":           [],
                "normal_force": [],
                "slip_ratio":   [],
                "slip_angle":   []
            },
            "rr" : {
                "omega":     [],
                "omega_dot": [],
                "fx":           [],
                "fy":           [],
                "normal_force": [],
                "slip_ratio":   [],
                "slip_angle":   []
            }
        }

    def update(self, state : TwinTrackState, used_derivatives: TwinTrackStateDerivatives, wheel_forces: dict[str, dict[str, float]], sample: int):
        time = sample * self.sim_step
        self.time.append(time)

        self.x.append(state.x)
        self.y.append(state.y)
        self.yaw.append(state.yaw)

        self.vx.append(state.vx)
        self.vy.append(state.vy)
        self.yaw_rate.append(state.yaw_rate)

        self.ax.append(used_derivatives.vx_dot)
        self.ay.append(used_derivatives.vy_dot)
        self.yaw_rate_dot.append(used_derivatives.yaw_rate_dot)

        for w in ["fl", "fr", "rl", "rr"]:
            for attr in ["omega", "omega_dot", "fx", "fy", "normal_force", "slip_ratio", "slip_angle"]:
                self.wheels[w][attr].append(wheel_forces[w][attr])

    def print_logs(self):
        # 1. Dynamické sestavení záhlaví (hlavičky)
        headers = ["time", "x", "y", "yaw", "vx", "vy", "yaw_rate", "ax", "ay", "yaw_rate_dot"]
        
        # Přidání dynamických hlaviček pro všechna kola
        for w in ["fl", "fr", "rl", "rr"]:
            for attr in ["omega", "omega_dot", "fx", "fy", "normal_force", "slip_ratio", "slip_angle"]:
                headers.append(f"{w}_{attr}")

        # Vytisknutí řádku 0 (záhlaví)
        print(" | ".join(headers))

        # 2. Vytisknutí jednotlivých datových řádků
        num_records = len(self.time)
        for i in range(num_records):
            row = [
                self.time[i],
                self.x[i],
                self.y[i],
                self.yaw[i],
                self.vx[i],
                self.vy[i],
                self.yaw_rate[i],
                self.ax[i],
                self.ay[i],
                self.yaw_rate_dot[i],
            ]

            # Přidání hodnot pro kola ve stejném pořadí jako v hlavičce
            for w in ["fl", "fr", "rl", "rr"]:
                for attr in ["omega", "omega_dot", "fx", "fy", "normal_force", "slip_ratio", "slip_angle"]:
                    row.append(self.wheels[w][attr][i])

            # Převedení všech hodnot na string zaokrouhlený pro lepší čitelnost v terminálu
            row_str = [f"{val:.4f}" if isinstance(val, float) else str(val) for val in row]
            print(" | ".join(row_str))

    def plot_results(self, reference: list | None = None, tracking: str = "vx", arrow_step: int = 20):
        """
        Zobrazí výsledky simulace v mřížce 2x2.
        
        Args:
            reference (list, optional): Pole požadované reference (vx nebo omega_r) pro vykreslení do grafu.
            tracking (str): Určuje, co se vykreslí na druhou pozici (vpravo nahoře).
                            Možnosti: "vx" (rychlosti vozidla) nebo "omega_r" (úhlové rychlosti kol).
            arrow_step (int): Vykreslí šipku orientace (yaw) každých N kroků simulace.
        """
        # Vytvoření figury s mřížkou 2x2
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Výsledky simulace vozidla", fontsize=14, fontweight="bold")

        ax_traj   = axes[0, 0]  # Vlevo nahoře (Trajektorie)
        ax_track  = axes[0, 1]  # Vpravo nahoře (Tracking vx nebo omega)
        ax_fx     = axes[1, 0]  # Vlevo dole (Fx)
        ax_fy     = axes[1, 1]  # Vpravo dole (Fy)

        # Barvy a popisky pro jednotlivá kola
        wheel_styles = {
            "fl": ("Front Left", "red"),
            "fr": ("Front Right", "orange"),
            "rl": ("Rear Left", "green"),
            "rr": ("Rear Right", "purple"),
        }

        # 1. Graf (0,0): Trajektorie auta (x, y) + šipky orientace (yaw)
        ax_traj.plot(self.x, self.y, label="Trajektorie", color="blue", linewidth=2)
        
        x_coords = np.array(self.x)
        y_coords = np.array(self.y)
        yaw_angles = np.array(self.yaw)

        indices = range(0, len(self.time), arrow_step)
        
        x_sampled = x_coords[indices]
        y_sampled = y_coords[indices]
        yaw_sampled = yaw_angles[indices]

        u_yaw = np.cos(yaw_sampled)
        v_yaw = np.sin(yaw_sampled)

        ax_traj.quiver(
            x_sampled, y_sampled, u_yaw, v_yaw,
            color="red",
            scale=30,
            width=0.0035,
            label="Orientace auta (Yaw)",
            zorder=5
        )

        ax_traj.set_title("Trajektorie vozidla (X-Y) s orientací vozidla")
        ax_traj.set_xlabel("x [m]")
        ax_traj.set_ylabel("y [m]")
        ax_traj.grid(True)
        ax_traj.axis("equal")
        ax_traj.legend(loc="upper right")

        # 2. Graf (0,1): Dynamické vykreslení podle parametru `tracking`
        if tracking == "vx":
            ax_track.plot(self.time, self.vx, label="vx (skutečná podélná)", color="green", linewidth=1.5)
            ax_track.plot(self.time, self.vy, label="vy (příčná)", color="magenta", linewidth=1.5)
            ax_track.plot(self.time, self.yaw_rate, label="yaw rate", color="blue", linewidth=1.5)
            
            if reference is not None and len(reference) == len(self.time):
                ax_track.plot(self.time, reference, label="vx (požadovaná reference)", color="black", linestyle="--", linewidth=1.5)

            ax_track.set_title("Rychlosti vozidla (vx, vy)")
            ax_track.set_ylabel("Rychlost [m/s]")

        elif tracking == "omega_r":
            for w, (label, color) in wheel_styles.items():
                ax_track.plot(self.time, self.wheels[w]["omega"], label=f"ω {w.upper()} ({label})", color=color, linewidth=1.2)

            if reference is not None and len(reference) == len(self.time):
                ax_track.plot(self.time, reference, label="ω_ref (zadní náprava)", color="black", linestyle="--", linewidth=1.5)

            ax_track.set_title("Úhlové rychlosti kol (ω)")
            ax_track.set_ylabel("Úhlová rychlost [rad/s]")
            
        else:
            raise ValueError(f"Neznámý typ tracking modifikátoru '{tracking}'. Použij 'vx' nebo 'omega_r'.")

        ax_track.set_xlabel("Čas kroků / Čas [s]")
        ax_track.grid(True)
        ax_track.legend(loc="upper right")

        # 3. Graf (1,0): Podélné síly Fx na kolech v čase
        for w, (label, color) in wheel_styles.items():
            ax_fx.plot(self.time, self.wheels[w]["fx"], label=f"Fx {w.upper()} ({label})", color=color)
        
        ax_fx.set_title("Podélné síly na kolech (Fx)")
        ax_fx.set_xlabel("Čas kroků / Čas [s]")
        ax_fx.set_ylabel("Fx [N]")
        ax_fx.grid(True)
        ax_fx.legend(loc="upper right")

        # 4. Graf (1,1): Příčné síly Fy na kolech v čase
        for w, (label, color) in wheel_styles.items():
            ax_fy.plot(self.time, self.wheels[w]["fy"], label=f"Fy {w.upper()} ({label})", color=color)

        ax_fy.set_title("Příčné síly na kolech (Fy)")
        ax_fy.set_xlabel("Čas kroků / Čas [s]")
        ax_fy.set_ylabel("Fy [N]")
        ax_fy.grid(True)
        ax_fy.legend(loc="upper right")

        plt.tight_layout()
        plt.show()

    def plot_slip_characteristics(self):
        """
        Vykreslí průběhy příčného úhlu smyku (Side Slip Angle) a podélného skluzu (Longitudinal Slip Ratio)
        v čase v mřížce 2x2 pro zadní a přední nápravu.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Průběhy skluzů pneumatik v čase", fontsize=14, fontweight="bold")

        ax_alpha_rear  = axes[0, 0]  # Graf 1: Side slip angle (Zadní náprava)
        ax_kappa_rear  = axes[0, 1]  # Graf 2: Longitudinal slip ratio (Zadní náprava)
        ax_alpha_front = axes[1, 0]  # Graf 3: Side slip angle (Přední náprava)
        ax_kappa_front = axes[1, 1]  # Graf 4: Longitudinal slip ratio (Přední náprava)

        wheel_colors = {
            "fl": "red",
            "fr": "orange",
            "rl": "green",
            "rr": "purple"
        }

        # ---------------------------------------------------------
        # 1. Graf (0,0): Side slip angle (α) pro ZADNÍ kola (RL, RR)
        # ---------------------------------------------------------
        for w in ["rl", "rr"]:
            alpha_deg = np.degrees(self.wheels[w]["slip_angle"])
            ax_alpha_rear.plot(self.time, alpha_deg, label=f"α {w.upper()}", color=wheel_colors[w], linewidth=1.5)

        ax_alpha_rear.set_title("Zadní náprava: Příčný úhel smyku (Side Slip Angle α)")
        ax_alpha_rear.set_xlabel("Čas [s]")
        ax_alpha_rear.set_ylabel("Úhel smyku α [deg]")
        ax_alpha_rear.grid(True)
        ax_alpha_rear.legend(loc="upper right")

        # ---------------------------------------------------------
        # 2. Graf (0,1): Longitudinal slip ratio (κ) pro ZADNÍ kola (RL, RR)
        # ---------------------------------------------------------
        for w in ["rl", "rr"]:
            kappa = self.wheels[w]["slip_ratio"]
            ax_kappa_rear.plot(self.time, kappa, label=f"κ {w.upper()}", color=wheel_colors[w], linewidth=1.5)

        ax_kappa_rear.set_title("Zadní náprava: Podélný skluz (Longitudinal Slip Ratio κ)")
        ax_kappa_rear.set_xlabel("Čas [s]")
        ax_kappa_rear.set_ylabel("Skluz κ [-]")
        ax_kappa_rear.grid(True)
        ax_kappa_rear.legend(loc="upper right")

        # ---------------------------------------------------------
        # 3. Graf (1,0): Side slip angle (α) pro PŘEDNÍ kola (FL, FR)
        # ---------------------------------------------------------
        for w in ["fl", "fr"]:
            alpha_deg = np.degrees(self.wheels[w]["slip_angle"])
            ax_alpha_front.plot(self.time, alpha_deg, label=f"α {w.upper()}", color=wheel_colors[w], linewidth=1.5)

        ax_alpha_front.set_title("Přední náprava: Příčný úhel smyku (Side Slip Angle α)")
        ax_alpha_front.set_xlabel("Čas [s]")
        ax_alpha_front.set_ylabel("Úhel smyku α [deg]")
        ax_alpha_front.grid(True)
        ax_alpha_front.legend(loc="upper right")

        # ---------------------------------------------------------
        # 4. Graf (1,1): Longitudinal slip ratio (κ) pro PŘEDNÍ kola (FL, FR)
        # ---------------------------------------------------------
        for w in ["fl", "fr"]:
            kappa = self.wheels[w]["slip_ratio"]
            ax_kappa_front.plot(self.time, kappa, label=f"κ {w.upper()}", color=wheel_colors[w], linewidth=1.5)

        ax_kappa_front.set_title("Přední náprava: Podélný skluz (Longitudinal Slip Ratio κ)")
        ax_kappa_front.set_xlabel("Čas [s]")
        ax_kappa_front.set_ylabel("Skluz κ [-]")
        ax_kappa_front.grid(True)
        ax_kappa_front.legend(loc="upper right")

        plt.tight_layout()
        plt.show()

    def pacejka_identification(self):
            """
            Vykreslí grafy charakteristik pneumatik pro identifikaci parametrů Pacejkovy formule:
            1. Fy vs. Side Slip Angle (Zadní kola)
            2. Fx vs. Longitudinal Slip Ratio (Zadní kola)
            3. Fy vs. Side Slip Angle (Přední kola)
            """
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle("Charakteristiky pneumatik pro identifikaci (Magic Formula / Pacejka)", fontsize=14, fontweight="bold")

            ax_fy_rear  = axes[0]  # Graf 1: Fy na zadních kolech
            ax_fx_rear  = axes[1]  # Graf 2: Fx na zadních kolech
            ax_fy_front = axes[2]  # Graf 3: Fy na předních kolech

            wheel_colors = {
                "fl": "red",
                "fr": "orange",
                "rl": "green",
                "rr": "purple"
            }

            # 1. Graf: Fy na side slip angle (Zadní kola RL, RR)
            for w in ["rl", "rr"]:
                slip_angle_deg = np.degrees(self.wheels[w]["slip_angle"])
                fy = self.wheels[w]["fy"]
                ax_fy_rear.scatter(slip_angle_deg, fy, s=10, label=f"Kolo {w.upper()}", color=wheel_colors[w], alpha=0.6)

            ax_fy_rear.set_title("Zadní náprava: Fy vs. Slip Angle")
            ax_fy_rear.set_xlabel("Příčný úhel smyku - Side Slip Angle α [deg]")
            ax_fy_rear.set_ylabel("Příčná síla Fy [N]")
            ax_fy_rear.grid(True)
            ax_fy_rear.legend()

            # 2. Graf: Fx na longitudinal slip ratio (Zadní kola RL, RR)
            for w in ["rl", "rr"]:
                slip_ratio = self.wheels[w]["slip_ratio"]
                fx = self.wheels[w]["fx"]
                ax_fx_rear.scatter(slip_ratio, fx, s=10, label=f"Kolo {w.upper()}", color=wheel_colors[w], alpha=0.6)

            ax_fx_rear.set_title("Zadní náprava: Fx vs. Slip Ratio")
            ax_fx_rear.set_xlabel("Podélný skluz - Longitudinal Slip Ratio κ [-]")
            ax_fx_rear.set_ylabel("Podélná síla Fx [N]")
            ax_fx_rear.grid(True)
            ax_fx_rear.legend()

            # 3. Graf: Fy na side slip angle (Přední kola FL, FR)
            for w in ["fl", "fr"]:
                slip_angle_deg = np.degrees(self.wheels[w]["slip_angle"])
                fy = self.wheels[w]["fy"]
                ax_fy_front.scatter(slip_angle_deg, fy, s=10, label=f"Kolo {w.upper()}", color=wheel_colors[w], alpha=0.6)

            ax_fy_front.set_title("Přední náprava: Fy vs. Slip Angle")
            ax_fy_front.set_xlabel("Příčný úhel smyku - Side Slip Angle α [deg]")
            ax_fy_front.set_ylabel("Příčná síla Fy [N]")
            ax_fy_front.grid(True)
            ax_fy_front.legend()

            plt.tight_layout()
            plt.show()

    def pacejka_coef_fit(self, start_sample: int = 0):
        """
        Identifikuje koeficienty B, C, D, E Pacejkovy formule z naměřených dat v Loggeru
        pro podélný i příčný směr na zadní nápravě a příčný směr na přední nápravě.
        
        Args:
            start_sample (int): Index vzorku (samplu), od kterého se má začít fitovat data.
        """
        # Matematická definice Pacejkovy formule (Magic Formula)
        def pacejka_model(x, b, c, d, e):
            shaped_slip = b * x
            return d * np.sin(c * np.arctan(shaped_slip - e * (shaped_slip - np.arctan(shaped_slip))))

        # Počáteční odhad parametrů [B, C, D, E] a meze pro optimalizaci
        initial_guess = [10.0, 1.3, 4.5, 0.9]
        bounds = ([0.1, 0.5, 0.0, -5.0], [50.0, 3.0, 20.0, 2.0])

        results = {}

        # ---------------------------------------------------------
        # 1. Příčný směr (Fy vs. Slip Angle) - Zadní náprava (RL + RR)
        # ---------------------------------------------------------
        alpha_rl = np.array(self.wheels["rl"]["slip_angle"])[start_sample:]
        alpha_rr = np.array(self.wheels["rr"]["slip_angle"])[start_sample:]
        fy_rl = np.array(self.wheels["rl"]["fy"])[start_sample:]
        fy_rr = np.array(self.wheels["rr"]["fy"])[start_sample:]

        alpha_rear = np.concatenate([alpha_rl, alpha_rr])
        fy_rear = np.concatenate([fy_rl, fy_rr])
        
        # Proložení dat
        popt_fy_rear, _ = curve_fit(pacejka_model, -alpha_rear, fy_rear, p0=initial_guess, bounds=bounds, maxfev=10000)
        results["Rear_Fy"] = popt_fy_rear

        # ---------------------------------------------------------
        # 2. Podélný směr (Fx vs. Slip Ratio) - Zadní náprava (RL + RR)
        # ---------------------------------------------------------
        kappa_rl = np.array(self.wheels["rl"]["slip_ratio"])[start_sample:]
        kappa_rr = np.array(self.wheels["rr"]["slip_ratio"])[start_sample:]
        fx_rl = np.array(self.wheels["rl"]["fx"])[start_sample:]
        fx_rr = np.array(self.wheels["rr"]["fx"])[start_sample:]

        kappa_rear = np.concatenate([kappa_rl, kappa_rr])
        fx_rear = np.concatenate([fx_rl, fx_rr])

        popt_fx_rear, _ = curve_fit(pacejka_model, kappa_rear, fx_rear, p0=initial_guess, bounds=bounds, maxfev=10000)
        results["Rear_Fx"] = popt_fx_rear

        # ---------------------------------------------------------
        # 3. Příčný směr (Fy vs. Slip Angle) - Přední náprava (FL + FR)
        # ---------------------------------------------------------
        alpha_fl = np.array(self.wheels["fl"]["slip_angle"])[start_sample:]
        alpha_fr = np.array(self.wheels["fr"]["slip_angle"])[start_sample:]
        fy_fl = np.array(self.wheels["fl"]["fy"])[start_sample:]
        fy_fr = np.array(self.wheels["fr"]["fy"])[start_sample:]

        alpha_front = np.concatenate([alpha_fl, alpha_fr])
        fy_front = np.concatenate([fy_fl, fy_fr])

        popt_fy_front, _ = curve_fit(pacejka_model, -alpha_front, fy_front, p0=initial_guess, bounds=bounds, maxfev=10000)
        results["Front_Fy"] = popt_fy_front

        # ---------------------------------------------------------
        # Výpis výsledků do terminálu
        # ---------------------------------------------------------
        print("\n" + "="*55)
        print(f"  VÝSLEDKY IDENTIFIKACE PACEJKY (od samplu {start_sample})")
        print("="*55)
        for key, coefs in results.items():
            print(f" ► {key:10s} | B = {coefs[0]:7.4f} | C = {coefs[1]:7.4f} | D (Peak) = {coefs[2]:7.4f} N | E = {coefs[3]:7.4f}")
        print("="*55 + "\n")

        # ---------------------------------------------------------
        # Vykreslení porovnání fitu s oříznutými daty
        # ---------------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Identifikace parametrů Pacejky od samplu {start_sample} (Naměřená data vs. Fit)", fontsize=14, fontweight="bold")

        # Graf 1: Zadní Fy
        x_eval_a = np.linspace(min(-alpha_rear), max(-alpha_rear), 300)
        axes[0].scatter(-alpha_rear, fy_rear, s=5, color="purple", alpha=0.3, label="Data (RL+RR)")
        axes[0].plot(x_eval_a, pacejka_model(x_eval_a, *popt_fy_rear), color="black", linewidth=2.5, label="Fit Pacejka")
        axes[0].set_title("Zadní náprava: Fy vs. -α")
        axes[0].set_xlabel("Úhel smyku -α [rad]")
        axes[0].set_ylabel("Příčná síla Fy [N]")
        axes[0].grid(True)
        axes[0].legend()

        # Graf 2: Zadní Fx
        x_eval_k = np.linspace(min(kappa_rear), max(kappa_rear), 300)
        axes[1].scatter(kappa_rear, fx_rear, s=5, color="green", alpha=0.3, label="Data (RL+RR)")
        axes[1].plot(x_eval_k, pacejka_model(x_eval_k, *popt_fx_rear), color="black", linewidth=2.5, label="Fit Pacejka")
        axes[1].set_title("Zadní náprava: Fx vs. κ")
        axes[1].set_xlabel("Podélný skluz κ [-]")
        axes[1].set_ylabel("Podélná síla Fx [N]")
        axes[1].grid(True)
        axes[1].legend()

        # Graf 3: Přední Fy
        x_eval_af = np.linspace(min(-alpha_front), max(-alpha_front), 300)
        axes[2].scatter(-alpha_front, fy_front, s=5, color="orange", alpha=0.3, label="Data (FL+FR)")
        axes[2].plot(x_eval_af, pacejka_model(x_eval_af, *popt_fy_front), color="black", linewidth=2.5, label="Fit Pacejka")
        axes[2].set_title("Přední náprava: Fy vs. -α")
        axes[2].set_xlabel("Úhel smyku -α [rad]")
        axes[2].set_ylabel("Příčná síla Fy [N]")
        axes[2].grid(True)
        axes[2].legend()

        plt.tight_layout()
        plt.show()

        return results

    def steering_koeff_identification(self, log_path: Path):
        # Načtení dat z CSV
        data = pd.read_csv(log_path)
        timestamp = data["timestamp"].to_numpy()
        time = timestamp - timestamp[0]  # Normalizace času od nuly
        cmd_steer_rad = data["cmd_steer_rad"].to_numpy()
        yaw_rate_data = data["yaw_rate"].to_numpy()
        yaw_rate_data_filtered = data["yaw_rate_filtered"].to_numpy()
        N = len(timestamp)

        # Vytvoření figury se třemi podgrafy nad sebou se sdílenou osou X
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        fig.suptitle("Analýza řízení a Yaw Rate", fontsize=14, fontweight="bold")

        # 1. Horní graf: cmd_steer_rad
        ax1.plot(time, cmd_steer_rad, label="Cmd Steer [rad]", color="purple", linewidth=1.5)
        ax1.set_title("Požadovaný úhel řízení")
        ax1.set_ylabel("Úhel [rad]")
        ax1.grid(True)
        ax1.legend(loc="upper right")

        # 2. Prostřední graf: yaw_rate_data a yaw_rate_data_filtered pohromadě
        ax2.plot(time, yaw_rate_data, label="Yaw Rate (Surová)", color="blue", alpha=0.5, linewidth=1)
        ax2.plot(time, yaw_rate_data_filtered, label="Yaw Rate (Filtrovaná)", color="orange", linewidth=1.5)
        ax2.set_title("Porovnání surové a filtrované úhlové rychlosti")
        ax2.set_ylabel("Yaw Rate [rad/s]")
        ax2.grid(True)
        ax2.legend(loc="upper right")

        # 3. Dolní graf: yaw_rate (pokud existuje jako atribut, případně lze upravit na yaw_rate_data)
        # Zde předpokládám zobrazení surové nebo jiné vámi zvolené varianty yaw_rate
        ax3.plot(self.time, self.yaw_rate, label="Yaw Rate", color="green", linewidth=1.5)
        ax3.set_title("Simulovaný Yaw Rate")
        ax3.set_xlabel("Čas [s]")
        ax3.set_ylabel("Yaw Rate [rad/s]")
        ax3.grid(True)
        ax3.legend(loc="upper right")

        plt.tight_layout()
        plt.show()

class Simulation:
    def __init__(self,
                 model: TwinTrackModel | None = None,
                 reference_vx: list | None = None,
                 reference_df: list | None = None,
                 sim_length: int | None = None,
                 sim_step: float | None = None,
                 init_state: TwinTrackState | None = None
                 ) -> None:
        
        ## Determining the length of simulation
        if reference_vx != None or reference_df != None:
            len_rvx = len(reference_vx) if reference_vx != None else 0
            len_rdf = len(reference_df) if reference_df != None else 0

            if len_rvx == len_rdf:              # using both
                self.sim_length = len_rvx
            elif len_rvx != 0 and len_rdf == 0: # using just vx ref
                self.sim_length = len_rvx
            elif len_rvx == 0 and len_rdf != 0: # using just df ref
                self.sim_length = len_rdf
            else:                               # incorrect format
                raise ValueError("References aren't the same length, how am I to work dumbfuck?")
            
        else:                                   # default setting
            self.sim_length = sim_length or 100
            
        self.model = model or TwinTrackModel()
        self.reference_vx = reference_vx or np.ones(self.sim_length, dtype=float)
        self.reference_df = reference_df or np.ones(self.sim_length, dtype=float)
        self.sim_step = sim_step or 0.01
        self.init_state = init_state or TwinTrackState() 

        self.logger = Logger(self.sim_step)

    def run(self):
        state = self.init_state
        speed_controller = RearWheelSpeedControllerLockedDiff(params= self.model.params, tracking= "vx", torque_limit= 0.2) 
                                                              #wheel_gains= PIDGains(0.1, 0.4, 0))
        max_abs_torque = 0.0
        max_abs_slip_ref = 0.0

        for _ in range(self.sim_length):
            command = speed_controller.step(state, ref=self.reference_vx[_], dt=self.sim_step)
            max_abs_torque = max(max_abs_torque, abs(command.left_torque), abs(command.right_torque))
            max_abs_slip_ref = max(max_abs_slip_ref, abs(command.slip_ratio_ref))

            new_state, used_derivatives, wheel_forces = self.model.step(
                state= state,
                dt= self.sim_step,
                steer= self.reference_df[_],
                rear_left_torque= command.left_torque,
                rear_right_torque= command.right_torque
                )

            # using old state for logging, 
            # the derivatives and forces are being calculated from that state
            # doesn't therefore make sense to connect new state with variables that lead to it
            self.logger.update(
                state= state,
                used_derivatives= used_derivatives,
                wheel_forces= wheel_forces,
                sample= _
            )
            state = new_state
        


def deg2rad(angle: float) -> float:
    return angle * pi / 180

def rad2deg(angle: float) -> float:
    return angle * 180 / pi


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a vehicle test log CSV.")
    parser.add_argument("csv_file", type=Path, help="Path to the CSV file to load")
    args = parser.parse_args()
    
    reference_vx = []
    reference_df = []

    k = 0.985

    sim_step = 0.001
    acc_time = 4000
    steering_time = 2000
    test_time = 11000
    sample_count = acc_time + steering_time + test_time # 15 sec at dt = 0.001

    for _ in range(sample_count):
        if _ < acc_time:
            reference_vx.append(1)
            reference_df.append(deg2rad(0))
        elif _ < acc_time + steering_time:
            reference_vx.append(1)
            reference_df.append(deg2rad((20/1.55) * (_ - acc_time)/steering_time))
        else:
            reference_vx.append( 1 + 0 * (_ - acc_time - steering_time) * sim_step)
            reference_df.append(deg2rad((20/1.55)))

    sim = Simulation(
        reference_vx=reference_vx,
        reference_df=reference_df,
        sim_length= None,
        sim_step=sim_step,
        )
    sim.run()
    sim.logger.steering_koeff_identification(args.csv_file)
    # sim.logger.plot_results(reference=reference_vx, tracking="vx")
    # sim.logger.pacejka_coef_fit(start_sample=4000)
    # sim.logger.plot_slip_characteristics()


    # ustalene experimenty - podelna a pricna dynamika
    # kombinovne experimenty, zrychlovani a trhani volantem

    # porovnani s matlab modelem

    # 

