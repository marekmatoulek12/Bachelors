import argparse
import glob
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, curve_fit

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
        wheelbase = 0.3
        return (velocity / wheelbase) * np.tan(sim_steer_rad) * 0.8

    sim_step = 0.001
    acc_time = 2000       # 2s rovně
    steering_time = 2000  # 2s nájezd řízení
    test_time = 4000      # 4s ustálený stav
    sample_count = acc_time + steering_time + test_time

    reference_vx = []
    reference_df = []

    for i in range(sample_count):
        reference_vx.append(velocity)
        if i < acc_time:
            reference_df.append(0.0)
        elif i < acc_time + steering_time:
            reference_df.append(sim_steer_rad * (i - acc_time) / steering_time)
        else:
            reference_df.append(sim_steer_rad)

    sim = Simulation(
        reference_vx=reference_vx,
        reference_df=reference_df,
        sim_length=None,
        sim_step=sim_step
    )
    sim.run()

    try:
        if hasattr(sim, 'logger') and hasattr(sim.logger, 'yaw_rate'):
            steady_yaw = np.mean(sim.logger.yaw_rate[-1000:])
        elif hasattr(sim, 'logger') and hasattr(sim.logger, 'states'):
            steady_yaw = np.mean(sim.logger.states[-1000:, 2])
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
        return (sim_yaw - target_yaw_rate) ** 2

    res = minimize_scalar(objective, bounds=(-45, 45), method='bounded')
    if res.success:
        return res.x
    return np.nan


def steer_cmd2real_rad(cmd_rad):
    """
    Převede povel řízení v radiánech na fyzický simulační úhel v radiánech
    podle kalibrované 3-segmentové funkce s mrtvou zónou.
    """
    xb1_deg = -3.03
    xb2_deg = 2.67

    k_left = 0.748
    b_left_deg = 2.85

    k_right = 0.743
    b_right_deg = -1.40

    offset_deg = 0.58

    cmd_deg = np.rad2deg(cmd_rad)
    real_steer_deg = np.zeros_like(cmd_deg, dtype=float)

    if np.isscalar(cmd_deg):
        if cmd_deg < xb1_deg:
            real_steer_deg = k_left * cmd_deg + b_left_deg
        elif cmd_deg > xb2_deg:
            real_steer_deg = k_right * cmd_deg + b_right_deg
        else:
            real_steer_deg = offset_deg
    else:
        mask_left = cmd_deg < xb1_deg
        real_steer_deg[mask_left] = k_left * cmd_deg[mask_left] + b_left_deg

        mask_mid = (cmd_deg >= xb1_deg) & (cmd_deg <= xb2_deg)
        real_steer_deg[mask_mid] = offset_deg

        mask_right = cmd_deg > xb2_deg
        real_steer_deg[mask_right] = k_right * cmd_deg[mask_right] + b_right_deg

    return np.deg2rad(real_steer_deg)


def _detect_steer_column(df: pd.DataFrame) -> str | None:
    """Najde sloupec s povelem řízení v DataFrame."""
    for col in df.columns:
        if col.lower() in ["cmd_steer_rad", "cmd_steer_deg", "cmd_steer", "steering", "steer", "delta", "df"]:
            return col
    candidates = [c for c in df.columns if "steer" in c.lower() or "deg" in c.lower() or "df" in c.lower()]
    return candidates[0] if candidates else None


def _extract_raw_steer_rad(df: pd.DataFrame) -> np.ndarray | None:
    """Vrátí povel řízení z logu převedený na radiány."""
    cmd_col = _detect_steer_column(df)
    if cmd_col is None:
        return None
    raw_vals = df[cmd_col].to_numpy(dtype=float)
    return np.deg2rad(raw_vals) if "deg" in cmd_col.lower() else raw_vals


def _straight_driving_mask(
    df: pd.DataFrame,
    steer_rad: np.ndarray | None = None,
    max_steer_deg: float = 0.2,
    min_time_offset: float = 3.0,
) -> np.ndarray:
    """
    Identifikuje ustálené přímé úseky jízdy (nulové řízení po ustálení rozjezdu).
    """
    n = len(df)
    mask = np.ones(n, dtype=bool)

    if "timestamp" in df.columns:
        t = df["timestamp"].to_numpy(dtype=float)
        if len(t) > 0 and np.isfinite(t[0]):
            mask &= (t - t[0]) >= min_time_offset

    if steer_rad is None:
        steer_rad = _extract_raw_steer_rad(df)

    if steer_rad is not None:
        mask &= np.isfinite(steer_rad) & (np.abs(steer_rad) <= np.deg2rad(max_steer_deg))
    elif "yaw_rate" in df.columns:
        yr = df["yaw_rate"].to_numpy(dtype=float)
        mask &= np.isfinite(yr) & (np.abs(yr) <= 0.05)
    else:
        return np.zeros(n, dtype=bool)

    if "vx" in df.columns:
        vx = df["vx"].to_numpy(dtype=float)
        mask &= np.isfinite(vx) & (vx > 0.3)

    return mask


def estimate_dataset_biases(
    log_files: list[str],
    calibrate_steer: bool = False,
) -> dict[str, float]:
    """
    Odhadne společné konstantní biasy senzorů ze všech přímých jízd v sadě logů:
      - ay_bias: klidový/náklonový offset akcelerometru accel_y [m/s^2]
      - yaw_rate_bias: offset gyroskopu yaw_rate [rad/s]
      - lidar_yaw_offset_rad: úhlový offset LiDAR lokalizace gps_yaw vůči směru jízdy [rad]

    Odhad z ustálených přímých jízd funguje i tehdy, když záznam začíná až za jízdy.
    """
    ay_samples = []
    yr_samples = []
    yaw_off_samples = []

    for file in log_files:
        try:
            df = pd.read_csv(file)
            raw_steer = _extract_raw_steer_rad(df)
            steer_for_check = (
                steer_cmd2real_rad(raw_steer)
                if (calibrate_steer and raw_steer is not None)
                else raw_steer
            )

            straight_mask = _straight_driving_mask(df, steer_rad=steer_for_check)

            # Fallback na klidový stav před rozjezdem, pokud v logu nejsou ustálené přímé úseky
            if np.sum(straight_mask) < 10 and "fl_rpm" in df.columns and "fr_rpm" in df.columns:
                fl = df["fl_rpm"].to_numpy(dtype=float)
                fr = df["fr_rpm"].to_numpy(dtype=float)
                straight_mask = np.isfinite(fl) & np.isfinite(fr) & (np.abs(fl) < 1.0) & (np.abs(fr) < 1.0)

            if np.sum(straight_mask) >= 10:
                if "accel_y" in df.columns:
                    ay_vals = df["accel_y"].to_numpy(dtype=float)[straight_mask]
                    ay_vals = ay_vals[np.isfinite(ay_vals)]
                    if len(ay_vals) > 0:
                        ay_samples.append(ay_vals)
                if "yaw_rate" in df.columns:
                    yr_vals = df["yaw_rate"].to_numpy(dtype=float)[straight_mask]
                    yr_vals = yr_vals[np.isfinite(yr_vals)]
                    if len(yr_vals) > 0:
                        yr_samples.append(yr_vals)

            # Odhad montážního úhlu LiDAR lokalizace z přímých jízd
            req_gps = {"timestamp", "gps_x", "gps_y", "gps_yaw"}
            if req_gps.issubset(df.columns):
                t = df["timestamp"].to_numpy(dtype=float)
                gx = df["gps_x"].to_numpy(dtype=float)
                gy = df["gps_y"].to_numpy(dtype=float)
                gyaw = df["gps_yaw"].to_numpy(dtype=float)
                st_mask = _straight_driving_mask(df, steer_rad=steer_for_check)

                last_x = last_y = last_yaw = last_t = None
                for i in range(len(df)):
                    if not (np.isfinite(gx[i]) and np.isfinite(gy[i]) and np.isfinite(gyaw[i])):
                        continue
                    if last_x is not None and (gx[i] != last_x or gy[i] != last_y):
                        dt_gps = t[i] - last_t
                        if dt_gps > 0.0 and st_mask[i]:
                            vx_g = (gx[i] - last_x) / dt_gps
                            vy_g = (gy[i] - last_y) / dt_gps
                            if np.hypot(vx_g, vy_g) >= 0.3:
                                vel_h = np.arctan2(vy_g, vx_g)
                                yaw_m = np.arctan2(
                                    np.sin(gyaw[i]) + np.sin(last_yaw),
                                    np.cos(gyaw[i]) + np.cos(last_yaw),
                                )
                                diff = np.arctan2(np.sin(vel_h - yaw_m), np.cos(vel_h - yaw_m))
                                yaw_off_samples.append(diff)
                        last_x, last_y, last_yaw, last_t = gx[i], gy[i], gyaw[i], t[i]
                    elif last_x is None:
                        last_x, last_y, last_yaw, last_t = gx[i], gy[i], gyaw[i], t[i]
        except Exception:
            continue

    ay_bias = float(np.median(np.concatenate(ay_samples))) if ay_samples else 0.0
    yr_bias = float(np.median(np.concatenate(yr_samples))) if yr_samples else 0.0
    yaw_off = float(np.median(yaw_off_samples)) if len(yaw_off_samples) >= 5 else 0.0

    return {
        "ay_bias": ay_bias,
        "yaw_rate_bias": yr_bias,
        "lidar_yaw_offset_rad": yaw_off,
    }


def recalibrate_vy(
    df: pd.DataFrame,
    steer_rad: np.ndarray | None = None,
    wheel_radius: float = 0.033,
    use_fw_encoders: bool = True,
    ay_bias: float = 0.0,
    yaw_rate_bias: float = 0.0,
    lidar_yaw_offset_rad: float = 0.0,
    r_gps_var: float = 2.0,
    gps_lever_arm_x: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Zpětně přepočítá vx a vy v těžišti (CG) z nahraných senzorových dat.

    Důležité zásady:
      1. Odečítá globální DC bias IMU (ay_bias, yaw_rate_bias) a montážní úhel LiDARu (lidar_yaw_offset_rad).
      2. Ponechává r_gps_var = 2.0, aby GPS nekroutilo dynamiku zatáčky (kde base_link / zadní náprava
         má jinou příčnou rychlost než těžiště: v_y,base = v_y,CG - l_r * r).
      3. Pokud je zadán gps_lever_arm_x (např. -0.16 m pro base_link na zadní nápravě), přepočítává
         měření GPS rychlosti na těžiště: v_y,CG = v_y,gps - gps_lever_arm_x * yaw_rate.
    """
    t = df["timestamp"].to_numpy(dtype=float)
    ax = df["accel_x"].to_numpy(dtype=float)
    ay = df["accel_y"].to_numpy(dtype=float) - ay_bias
    yaw_rate = df["yaw_rate"].to_numpy(dtype=float) - yaw_rate_bias
    gps_x = df["gps_x"].to_numpy(dtype=float)
    gps_y = df["gps_y"].to_numpy(dtype=float)
    gps_yaw = df["gps_yaw"].to_numpy(dtype=float) + lidar_yaw_offset_rad

    has_fw = use_fw_encoders and ("fl_rpm" in df.columns) and ("fr_rpm" in df.columns)
    if has_fw:
        fl_rpm = df["fl_rpm"].to_numpy(dtype=float)
        fr_rpm = df["fr_rpm"].to_numpy(dtype=float)

    n = len(df)
    vx_out = np.zeros(n, dtype=float)
    vy_out = np.zeros(n, dtype=float)

    if "vx" in df.columns and np.isfinite(df["vx"].iloc[0]):
        vx_init = float(df["vx"].iloc[0])
    elif has_fw and np.isfinite(fl_rpm[0]) and np.isfinite(fr_rpm[0]):
        vx_init = float(((fl_rpm[0] + fr_rpm[0]) * 0.5) * (2.0 * np.pi * wheel_radius / 60.0))
    else:
        vx_init = 0.0

    x = np.array([vx_init, 0.0], dtype=float)
    P = np.eye(2) * 0.1

    R_gps = np.eye(2) * r_gps_var
    # Odpojení příčné složky z předních enkodérů (R_vy = 1e6), aby EKF nenutil alpha_f = 0
    R_wheels_base = np.diag([0.1, 1e6])

    last_gps_x = gps_x[0] if np.isfinite(gps_x[0]) else None
    last_gps_y = gps_y[0] if np.isfinite(gps_y[0]) else None
    last_gps_yaw = gps_yaw[0] if np.isfinite(gps_yaw[0]) else None
    last_gps_t = t[0]

    vx_out[0] = x[0]
    vy_out[0] = x[1]

    for i in range(1, n):
        dt = t[i] - t[i - 1]
        if dt <= 0.0:
            vx_out[i] = x[0]
            vy_out[i] = x[1]
            continue

        # 1. IMU predikce (s odečteným biasem)
        if np.isfinite(ax[i]) and np.isfinite(ay[i]) and np.isfinite(yaw_rate[i]):
            vx_new = x[0] + (ax[i] + yaw_rate[i] * x[1]) * dt
            vy_new = x[1] + (ay[i] - yaw_rate[i] * x[0]) * dt
            F = np.array([
                [1.0, yaw_rate[i] * dt],
                [-yaw_rate[i] * dt, 1.0],
            ])
            Q = np.eye(2) * (0.5 * dt)
            x = np.array([vx_new, vy_new])
            P = F @ P @ F.T + Q

        # 2. Aktualizace z předních enkodérů (pouze podélná složka vx)
        if has_fw and np.isfinite(fl_rpm[i]) and np.isfinite(fr_rpm[i]):
            wheel_vx = ((fl_rpm[i] + fr_rpm[i]) * 0.5) * (2.0 * np.pi * wheel_radius / 60.0)
            delta = float(steer_rad[i]) if (steer_rad is not None and np.isfinite(steer_rad[i])) else 0.0
            z_vx_w = np.cos(delta) * wheel_vx

            is_slipping = (abs(ay[i]) > 4.0) or (abs(yaw_rate[i]) > 2.0)
            R_w = R_wheels_base * (1000.0 if is_slipping else 1.0)

            z_w = np.array([z_vx_w, x[1]])
            y_w = z_w - x
            S_w = P + R_w
            K_w = P @ np.linalg.inv(S_w)
            x = x + K_w @ y_w
            P = (np.eye(2) - K_w) @ P

        # 3. GPS aktualizace (pouze v krocích, kdy přišel nový GPS vzorek)
        if np.isfinite(gps_x[i]) and np.isfinite(gps_y[i]) and np.isfinite(gps_yaw[i]):
            if last_gps_x is not None and (gps_x[i] != last_gps_x or gps_y[i] != last_gps_y):
                dt_gps = t[i] - last_gps_t
                if dt_gps > 0.0:
                    v_X_glob = (gps_x[i] - last_gps_x) / dt_gps
                    v_Y_glob = (gps_y[i] - last_gps_y) / dt_gps

                    yaw_mid = np.arctan2(
                        np.sin(gps_yaw[i]) + np.sin(last_gps_yaw),
                        np.cos(gps_yaw[i]) + np.cos(last_gps_yaw),
                    )
                    c, s = np.cos(-yaw_mid), np.sin(-yaw_mid)
                    z_vx_gps = c * v_X_glob - s * v_Y_glob
                    z_vy_gps = s * v_X_glob + c * v_Y_glob

                    # Přepočet rychlosti z referenčního bodu lokalizace (gps_lever_arm_x) do CG
                    if np.isfinite(yaw_rate[i]) and gps_lever_arm_x != 0.0:
                        z_vy_gps = z_vy_gps - gps_lever_arm_x * yaw_rate[i]

                    z_gps = np.array([z_vx_gps, z_vy_gps])
                    y_g = z_gps - x
                    S_g = P + R_gps
                    K_g = P @ np.linalg.inv(S_g)
                    x = x + K_g @ y_g
                    P = (np.eye(2) - K_g) @ P

                last_gps_x = gps_x[i]
                last_gps_y = gps_y[i]
                last_gps_yaw = gps_yaw[i]
                last_gps_t = t[i]
            elif last_gps_x is None:
                last_gps_x = gps_x[i]
                last_gps_y = gps_y[i]
                last_gps_yaw = gps_yaw[i]
                last_gps_t = t[i]

        vx_out[i] = x[0]
        vy_out[i] = x[1]

    return vx_out, vy_out


def calibrate_log(
    file_path: str,
    calibrate_steer: bool = True,
    calibrate_vy: bool = True,
    output_dir: str | None = None,
    ay_bias: float = 0.0,
    yaw_rate_bias: float = 0.0,
    lidar_yaw_offset_rad: float = 0.0,
    r_gps_var: float = 2.0,
    gps_lever_arm_x: float = 0.0,
):
    """
    Načte zadaný .csv soubor a podle zvolených přepínačů provede kalibraci řízení a/nebo vy.
    """
    if not os.path.exists(file_path):
        print(f"Chyba: Soubor {file_path} neexistuje.")
        return

    df = pd.read_csv(file_path)
    raw_steer_rad = _extract_raw_steer_rad(df)

    if calibrate_steer:
        if raw_steer_rad is None:
            print(f"Chyba: V souboru {file_path} nebyl nalezen sloupec pro povel řízení.")
            return
        df["real_steer_rad"] = steer_cmd2real_rad(raw_steer_rad)

    if calibrate_vy:
        if "vy" in df.columns and "vy_raw_ekf" not in df.columns:
            df["vy_raw_ekf"] = df["vy"]

        if "real_steer_rad" in df.columns:
            steer_for_ekf = df["real_steer_rad"].to_numpy(dtype=float)
        else:
            steer_for_ekf = raw_steer_rad

        vx_cal, vy_cal = recalibrate_vy(
            df,
            steer_rad=steer_for_ekf,
            ay_bias=ay_bias,
            yaw_rate_bias=yaw_rate_bias,
            lidar_yaw_offset_rad=lidar_yaw_offset_rad,
            r_gps_var=r_gps_var,
            gps_lever_arm_x=gps_lever_arm_x,
        )
        df["vx"] = vx_cal
        df["vy"] = vy_cal

        # Apply the exact same sensor bias corrections to the saved CSV columns
        if "accel_y" in df.columns:
            df["accel_y"] = df["accel_y"] - ay_bias
        if "yaw_rate" in df.columns:
            df["yaw_rate"] = df["yaw_rate"] - yaw_rate_bias
        if "gps_yaw" in df.columns:
            df["gps_yaw"] = df["gps_yaw"] + lidar_yaw_offset_rad

    target_dir = output_dir or os.path.join(repo_root, "Model_lib_and_sim", "Identification", "Calibrated")
    os.makedirs(target_dir, exist_ok=True)

    original_name = os.path.basename(file_path)
    output_filename = original_name if original_name.startswith("calibrated_") else f"calibrated_{original_name}"
    output_path = os.path.join(target_dir, output_filename)

    df.to_csv(output_path, index=False)
    print(f"Uloženo: {output_path} (steer={'ON' if calibrate_steer else 'OFF'}, vy={'ON' if calibrate_vy else 'OFF'})")


def calibrate_all_logs(
    sub_dir: str = "",
    calibrate_steer: bool = True,
    calibrate_vy: bool = True,
    output_dir: str | None = None,
    ay_bias: float | None = None,
    yaw_rate_bias: float | None = None,
    lidar_yaw_offset_rad: float | None = None,
    r_gps_var: float = 2.0,
    gps_lever_arm_x: float = 0.0,
):
    """
    Hromadně překalibruje všechny logy (rst_* i csst_*) v zadané podsložce log_files.
    Automaticky odhadne biasy (ay_bias, yaw_rate_bias, lidar_yaw_offset) ze všech přímých jízd v dávce.
    """
    base_dir = os.path.join(repo_root, "log_files", sub_dir) if sub_dir else repo_root
    if sub_dir and not os.path.exists(base_dir):
        print(f"Upozornění: Složka {base_dir} neexistuje, prohledávám celé repo {repo_root}...")
        base_dir = repo_root

    patterns = [
        os.path.join(base_dir, "**", "csst_*_log_*.csv"),
        os.path.join(base_dir, "**", "rst_*_log_*.csv"),
        os.path.join(base_dir, "**", "rst*.csv"),
    ]

    log_files = []
    for pat in patterns:
        log_files.extend(glob.glob(pat, recursive=True))
    log_files = sorted({os.path.abspath(f) for f in log_files if not os.path.basename(f).startswith("calibrated_")})

    if not log_files:
        print(f"Nenalezeny žádné CSV logy v {base_dir}")
        return

    auto_biases = estimate_dataset_biases(log_files, calibrate_steer=calibrate_steer)
    eff_ay_bias = auto_biases["ay_bias"] if ay_bias is None else ay_bias
    eff_yr_bias = auto_biases["yaw_rate_bias"] if yaw_rate_bias is None else yaw_rate_bias
    eff_yaw_off = auto_biases["lidar_yaw_offset_rad"] if lidar_yaw_offset_rad is None else lidar_yaw_offset_rad

    if calibrate_vy:
        print(
            f"Použité biasy z přímých jízd: "
            f"ay_bias={eff_ay_bias:+.4f} m/s², "
            f"yaw_rate_bias={eff_yr_bias:+.4f} rad/s, "
            f"gps_yaw_offset={np.rad2deg(eff_yaw_off):+.2f}° ({eff_yaw_off:+.4f} rad), "
            f"R_gps={r_gps_var}, lever_arm_x={gps_lever_arm_x:+.3f} m"
        )

    print(f"Nalezeno {len(log_files)} logů. Spouštím kalibraci (steer={calibrate_steer}, vy={calibrate_vy})...")

    for file in log_files:
        try:
            calibrate_log(
                file,
                calibrate_steer=calibrate_steer,
                calibrate_vy=calibrate_vy,
                output_dir=output_dir,
                ay_bias=eff_ay_bias,
                yaw_rate_bias=eff_yr_bias,
                lidar_yaw_offset_rad=eff_yaw_off,
                r_gps_var=r_gps_var,
                gps_lever_arm_x=gps_lever_arm_x,
            )
        except Exception as e:
            print(f"Přeskočeno {os.path.basename(file)}: chyba při zpracování ({e})")


def merge_calibrated_logs(input_dir, output_filepath):
    search_pattern = os.path.join(input_dir, 'calibrated_*.csv')
    log_files = sorted(glob.glob(search_pattern))

    if not log_files:
        print("Nenalezeny žádné soubory ke spojení.")
        return

    dataframes_to_merge = []

    for i, file in enumerate(log_files):
        df = pd.read_csv(file)
        dataframes_to_merge.append(df)

        if i < len(log_files) - 1:
            zeros_array = np.zeros((100, len(df.columns)))
            df_zeros = pd.DataFrame(zeros_array, columns=df.columns)
            dataframes_to_merge.append(df_zeros)

    final_df = pd.concat(dataframes_to_merge, ignore_index=True)
    final_df.to_csv(output_filepath, index=False)
    print(f"Úspěšně spojeno {len(log_files)} souborů do: {output_filepath}")
    print(f"Celkový počet řádků: {len(final_df)}")


def piecewise_3_segments_horizontal(x, k1, k3, xb1, xb2, y_offset):
    condlist = [x < xb1, (x >= xb1) & (x <= xb2), x > xb2]
    funclist = [
        lambda x: y_offset + k1 * (x - xb1),
        lambda x: y_offset,
        lambda x: y_offset + k3 * (x - xb2)
    ]
    return np.piecewise(x, condlist, funclist)


def main(log_path: str = '**'):
    search_pattern = os.path.join(repo_root, 'log_files', log_path, 'rst_*_log_*.csv')
    log_files = glob.glob(search_pattern, recursive=True)

    if not log_files:
        search_pattern = os.path.join(repo_root, 'log_files', log_path, 'rst*.csv')
        log_files = glob.glob(search_pattern, recursive=True)

    if not log_files:
        print(f"Nenalezeny žádné CSV soubory rst_*_log_* v {repo_root}")
        return

    print(f"Nalezeno {len(log_files)} logů. Analyzuji data...")
    results = []

    for file in log_files:
        match = re.search(r'rst_([-_]?\d+)', file)
        if not match:
            continue

        angle_str = match.group(1).replace('_', '-')
        cmd_angle = float(angle_str)

        try:
            df = pd.read_csv(file)
            yaw_col = YAW_RATE_COL if YAW_RATE_COL in df.columns else [c for c in df.columns if 'yaw' in c.lower()][0]
            vel_col = VELOCITY_COL if VELOCITY_COL in df.columns else [c for c in df.columns if 'vel' in c.lower() or 'speed' in c.lower() or 'vx' in c.lower()][0]

            steady_df = df.tail(STEADY_STATE_SAMPLES)
            real_yaw_rate = steady_df[yaw_col].mean()
            real_velocity = steady_df[vel_col].mean()

            results.append({
                'cmd_angle': cmd_angle,
                'real_yaw_rate': real_yaw_rate,
                'real_velocity': real_velocity
            })
        except Exception as e:
            print(f"Přeskočeno {os.path.basename(file)}: chybí sloupce nebo nelze načíst ({e})")

    if not results:
        return

    results_df = pd.DataFrame(results).dropna()
    results_df = results_df[(results_df['real_velocity'] > 0.5) & (results_df['real_velocity'] < 1.5)]
    results_df = results_df.sort_values(by='cmd_angle').groupby('cmd_angle').median().reset_index()

    sim_angles = []
    for idx, row in results_df.iterrows():
        opt_angle = optimize_steering_angle(row['real_yaw_rate'], row['real_velocity'])
        sim_angles.append(opt_angle)

    results_df['sim_angle'] = sim_angles
    results_df = results_df.dropna()

    x_data = results_df['cmd_angle'].to_numpy()
    y_data = results_df['sim_angle'].to_numpy()

    mask_center = (x_data >= -1.0) & (x_data <= 1.0)
    fixed_y_offset = np.mean(y_data[mask_center]) if np.sum(mask_center) > 0 else 0.0

    def piecewise_3_segments_fixed_horizontal(x, k1, k3, xb1, xb2):
        condlist = [x < xb1, (x >= xb1) & (x <= xb2), x > xb2]
        funclist = [
            lambda x: fixed_y_offset + k1 * (x - xb1),
            lambda x: fixed_y_offset,
            lambda x: fixed_y_offset + k3 * (x - xb2)
        ]
        return np.piecewise(x, condlist, funclist)

    p0 = [0.75, 0.75, -3.0, 3.0]
    bounds = ([0.1, 0.1, -15.0, 0.1], [2.0, 2.0, -0.1, 15.0])

    try:
        popt, _ = curve_fit(piecewise_3_segments_fixed_horizontal, x_data, y_data, p0=p0, bounds=bounds)
        k1, k3, xb1, xb2 = popt
    except Exception:
        k1, k3, xb1, xb2 = 0.75, 0.75, -3.0, 3.0

    b1 = fixed_y_offset - k1 * xb1
    b2 = fixed_y_offset
    b3 = fixed_y_offset - k3 * xb2

    plt.figure(figsize=(9, 6))
    plt.scatter(x_data, y_data, color='red', zorder=5, label='Optimalizované body')
    x_left = np.linspace(min(x_data) - 2, xb1, 50)
    x_mid = np.linspace(xb1, xb2, 50)
    x_right = np.linspace(xb2, max(x_data) + 2, 50)
    plt.plot(x_left, k1 * x_left + b1, color='#00D154', linewidth=2.5, label='Aproximace (s mrtvou zónou)')
    plt.plot(x_mid, np.full_like(x_mid, fixed_y_offset), color='#00D154', linewidth=2.5)
    plt.plot(x_right, k3 * x_right + b3, color='#00D154', linewidth=2.5)
    plt.scatter([xb1, xb2], [fixed_y_offset, fixed_y_offset], color='purple', s=70, zorder=6, label='Body zlomů')
    plt.title('Kalibrace řízení: Aproximace s vodorovnou mrtvou zónou')
    plt.xlabel('Povel z logu (stupně)')
    plt.ylabel('Hledaný fyzický úhel simulace (stupně)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plot_path = os.path.join(repo_root, 'Model_lib_and_sim', 'Identification', 'sc_after_calibration.png')
    plt.savefig(plot_path, dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Kalibrace řízení a zpětná rekonstrukce příčné rychlosti vy v těžišti (CG)."
    )
    parser.add_argument(
        "--mode",
        choices=["vy", "both", "steer", "fit"],
        default="vy",
        help="Režim běhu: 'vy' (výchozí), 'both', 'steer', 'fit'.",
    )
    parser.add_argument("--subdir", type=str, default="", help="Podsložka v log_files (např. '16.11.2026').")
    parser.add_argument("--file", type=str, default=None, help="Volitelná cesta k jednomu CSV souboru.")
    parser.add_argument("--out", type=str, default=None, help="Volitelná výstupní složka.")
    parser.add_argument(
        "--yaw-offset-deg",
        type=str,
        default="auto",
        help="Montážní offset gps_yaw [deg] ('auto' nebo číslo, např. '-2.5').",
    )
    parser.add_argument(
        "--r-gps",
        type=float,
        default=2.0,
        help="Rozptyl šumu GPS rychlosti v EKF (výchozí: 2.0, aby GPS nedeformovalo zatáčky).",
    )
    parser.add_argument(
        "--lever-arm-x",
        type=float,
        default=0.0,
        help="Podélná poloha referenčního bodu /gps/pose vůči CG [m] (např. -0.16 pro base_link na zadní nápravě; výchozí: 0.0).",
    )
    args = parser.parse_args()

    yaw_off_rad = None if args.yaw_offset_deg.strip().lower() == "auto" else float(np.deg2rad(float(args.yaw_offset_deg)))

    if args.mode == "fit":
        main(args.subdir if args.subdir else "**")
    else:
        cal_steer = args.mode in ("both", "steer")
        cal_vy = args.mode in ("both", "vy")

        if args.file:
            b = estimate_dataset_biases([args.file], calibrate_steer=cal_steer)
            calibrate_log(
                file_path=args.file,
                calibrate_steer=cal_steer,
                calibrate_vy=cal_vy,
                output_dir=args.out,
                ay_bias=b["ay_bias"],
                yaw_rate_bias=b["yaw_rate_bias"],
                lidar_yaw_offset_rad=b["lidar_yaw_offset_rad"] if yaw_off_rad is None else yaw_off_rad,
                r_gps_var=args.r_gps,
                gps_lever_arm_x=args.lever_arm_x,
            )
        else:
            calibrate_all_logs(
                sub_dir=args.subdir,
                calibrate_steer=cal_steer,
                calibrate_vy=cal_vy,
                output_dir=args.out,
                lidar_yaw_offset_rad=yaw_off_rad,
                r_gps_var=args.r_gps,
                gps_lever_arm_x=args.lever_arm_x,
            )
