import os
import sys
import glob
import re
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

def steer_cmd2real_rad(cmd_rad):
    """
    Převede povel řízení v radiánech na fyzický simulační úhel v radiánech
    podle kalibrované 3-segmentové funkce s mrtvou zónou.
    """
    # Parametry z kalibračního grafu (hodnoty ve stupních)
    xb1_deg = -3.03
    xb2_deg = 2.67
    
    k_left = 0.748
    b_left_deg = 2.85
    
    k_right = 0.743
    b_right_deg = -1.40
    
    offset_deg = 0.58
    
    # Převod vstupního povelu z radiánů na stupně pro vyhodnocení
    cmd_deg = np.rad2deg(cmd_rad)
    
    # Inicializace výstupního pole stejného tvaru (podporuje skalár i numpy array)
    real_steer_deg = np.zeros_like(cmd_deg)
    
    # Vyhodnocení pro skalární vstup
    if np.isscalar(cmd_deg):
        if cmd_deg < xb1_deg:
            real_steer_deg = k_left * cmd_deg + b_left_deg
        elif cmd_deg > xb2_deg:
            real_steer_deg = k_right * cmd_deg + b_right_deg
        else:
            real_steer_deg = offset_deg
            
    # Vyhodnocení pro numpy pole (používáno v pandasu)
    else:
        # Levá část
        mask_left = cmd_deg < xb1_deg
        real_steer_deg[mask_left] = k_left * cmd_deg[mask_left] + b_left_deg
        
        # Střed (mrtvá zóna)
        mask_mid = (cmd_deg >= xb1_deg) & (cmd_deg <= xb2_deg)
        real_steer_deg[mask_mid] = offset_deg
        
        # Pravá část
        mask_right = cmd_deg > xb2_deg
        real_steer_deg[mask_right] = k_right * cmd_deg[mask_right] + b_right_deg

    # Výsledek musí být převeden zpět na radiány
    return np.deg2rad(real_steer_deg)

def calibrate_log(file_path):
    STEERING_CMD_COL = "cmd_steer_rad"
    """
    Načte zadaný .csv soubor, přidá sloupec real_steer_rad obsahující data
    převedená 3-segmentovou funkcí s mrtvou zónou a uloží jej jako calibrated_{original_name}.csv
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

    # Aplikace nového třísegmentového kalibračního modelu
    # Vstup do funkce musí být v radiánech, což data v logu už jsou
    df['real_steer_rad'] = steer_cmd2real_rad(df[cmd_col].to_numpy())

    # Výstupní složka Model_lib_and_sim/Identification/Calibrated
    target_dir = os.path.join(repo_root, 'Model_lib_and_sim', 'Identification', 'Calibrated')
    os.makedirs(target_dir, exist_ok=True)

    original_name = os.path.basename(file_path)
    output_filename = f"calibrated_{original_name}"
    output_path = os.path.join(target_dir, output_filename)

    df.to_csv(output_path, index=False)
    print(f"Kalibrovaný soubor uložen do: {output_path}")

def calibrate_all_logs():
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
          
        try:
            calibrate_log(os.path.abspath(file))
            print(f"Zpracován Log: {os.path.basename(file)}") 
    
        except Exception as e:
            # Navráceno: Informace o přeskočených souborech
            print(f"Přeskočeno {os.path.basename(file)}: chybí sloupce nebo nelze načíst ({e})")

def merge_calibrated_logs(input_dir, output_filepath):
    # Najde všechny CSV soubory ve složce (upravte si masku podle potřeby)
    search_pattern = os.path.join(input_dir, 'calibrated_*.csv')
    log_files = sorted(glob.glob(search_pattern))
    
    if not log_files:
        print("Nenalezeny žádné soubory ke spojení.")
        return

    dataframes_to_merge = []
    
    for i, file in enumerate(log_files):
        # Načtení konkrétního logu
        df = pd.read_csv(file)
        dataframes_to_merge.append(df)
        
        # Pokud se nejedná o poslední soubor, přidáme 100 nulových řádků
        if i < len(log_files) - 1:
            # Vytvoření pole nul o velikosti (100 řádků, počet sloupců aktuálního df)
            zeros_array = np.zeros((100, len(df.columns)))
            
            # Převedení na DataFrame se stejnými názvy sloupců
            df_zeros = pd.DataFrame(zeros_array, columns=df.columns)
            
            dataframes_to_merge.append(df_zeros)

    # Sloučení všeho do jednoho velkého DataFrame
    # ignore_index=True zajistí, že se řádky plynule přečíslují od 0 do N
    final_df = pd.concat(dataframes_to_merge, ignore_index=True)
    
    # Uložení výsledku
    final_df.to_csv(output_filepath, index=False)
    print(f"Úspěšně spojeno {len(log_files)} souborů do: {output_filepath}")
    print(f"Celkový počet řádků: {len(final_df)}")

# --- POUŽITÍ ---
# input_folder = 'Model_lib_and_sim/Identification/Calibrated'
# output_file = 'Model_lib_and_sim/Identification/combined_calibrated_logs.csv'
# merge_calibrated_logs(input_folder, output_file)

def piecewise_3_segments_horizontal(x, k1, k3, xb1, xb2, y_offset):
    """
    Spojitá 3-segmentová funkce s přesně vodorovnou mrtvou zónou:
    - Segment 1 (x < xb1): směrnice k1
    - Segment 2 (xb1 <= x <= xb2): konstantní hodnota y_offset (vodorovně)
    - Segment 3 (x > xb2): směrnice k3
    """
    condlist = [x < xb1, (x >= xb1) & (x <= xb2), x > xb2]
    funclist = [
        lambda x: y_offset + k1 * (x - xb1),
        lambda x: y_offset,
        lambda x: y_offset + k3 * (x - xb2)
    ]
    return np.piecewise(x, condlist, funclist)

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
        match = re.search(r'rst_([-_]?\d+)', file)
        if not match: continue
        
        angle_str = match.group(1).replace('_', '-')
        cmd_angle = float(angle_str)

        try:
            df = pd.read_csv(file)
            yaw_col = YAW_RATE_COL if YAW_RATE_COL in df.columns else [c for c in df.columns if 'yaw' in c.lower()][0]
            vel_col = VELOCITY_COL if VELOCITY_COL in df.columns else [c for c in df.columns if 'vel' in c.lower() or 'speed' in c.lower() or 'vx' in c.lower()][0]

            steady_df = df.tail(STEADY_STATE_SAMPLES)
            real_yaw_rate = steady_df[yaw_col].mean()
            real_velocity = steady_df[vel_col].mean()
            
            # Navráceno: Výpis stavu zpracovávaného logu
            print(f"Zpracován Log: {os.path.basename(file)} | Cmd: {cmd_angle}° | V_x: {real_velocity:.2f} m/s | Yaw: {real_yaw_rate:.3f} rad/s")

            results.append({
                'cmd_angle': cmd_angle,
                'real_yaw_rate': real_yaw_rate,
                'real_velocity': real_velocity
            })
        except Exception as e:
            # Navráceno: Informace o přeskočených souborech
            print(f"Přeskočeno {os.path.basename(file)}: chybí sloupce nebo nelze načíst ({e})")

    if not results: return

    results_df = pd.DataFrame(results).dropna()
    results_df = results_df[(results_df['real_velocity'] > 0.5) & (results_df['real_velocity'] < 1.5)]
    results_df = results_df.sort_values(by='cmd_angle').groupby('cmd_angle').median().reset_index()

    print("\nOptimalizuji úhly v simulaci (toto chvíli potrvá)...")
    sim_angles = []
    for idx, row in results_df.iterrows():
        opt_angle = optimize_steering_angle(row['real_yaw_rate'], row['real_velocity'])
        sim_angles.append(opt_angle)
        # Navráceno: Výpis průběžných výsledků z optimalizátoru simulace
        print(f"  Povel {row['cmd_angle']}° -> Simulační úhel {opt_angle:.2f}°")

    results_df['sim_angle'] = sim_angles
    results_df = results_df.dropna()

    # ... (předchozí kód až k načtení x_data a y_data) ...
    x_data = results_df['cmd_angle'].to_numpy()
    y_data = results_df['sim_angle'].to_numpy()

    # --- URČENÍ FIXNÍHO OFFSETU ZE STŘEDOVÝCH BODŮ ---
    # Zprůměrujeme body, které leží blízko nuly (např. od -1° do 1°)
    mask_center = (x_data >= -1.0) & (x_data <= 1.0)
    if np.sum(mask_center) > 0:
        fixed_y_offset = np.mean(y_data[mask_center])
    else:
        # Fallback, pokud by u nuly nebyly žádné body
        fixed_y_offset = 0.0
        
    print(f"Fixovaný offset ze středových bodů: {fixed_y_offset:.4f}°")

    # Upravená funkce, která neočekává y_offset jako volný parametr, ale používá fixní
    def piecewise_3_segments_fixed_horizontal(x, k1, k3, xb1, xb2):
        condlist = [x < xb1, (x >= xb1) & (x <= xb2), x > xb2]
        funclist = [
            lambda x: fixed_y_offset + k1 * (x - xb1),
            lambda x: fixed_y_offset,
            lambda x: fixed_y_offset + k3 * (x - xb2)
        ]
        return np.piecewise(x, condlist, funclist)

    # --- FITACE ZBÝVAJÍCÍCH PARAMETRŮ ---
    # Parametry: [k1, k3, xb1, xb2]
    p0 = [0.75, 0.75, -3.0, 3.0]
    bounds = (
        [0.1, 0.1, -15.0,  0.1],  # spodní limity
        [2.0, 2.0,  -0.1, 15.0]   # horní limity
    )

    try:
        popt, _ = curve_fit(piecewise_3_segments_fixed_horizontal, x_data, y_data, p0=p0, bounds=bounds)
        k1, k3, xb1, xb2 = popt
    except Exception as e:
        print(f"Fallback na odhad přímek: {e}")
        k1, k3 = 0.75, 0.75
        xb1, xb2 = -3.0, 3.0

    b1 = fixed_y_offset - k1 * xb1
    b2 = fixed_y_offset
    b3 = fixed_y_offset - k3 * xb2

    print("\n=== VÝSLEDKY KALIBRACE S FIXNÍ VODOROVNOU MRTVOU ZÓNOU ===")
    print(f"Levá oblast  (x < {xb1:.2f}°):  y = {k1:.4f}x + {b1:.4f}")
    print(f"Mrtvá zóna   (hystereze):   y = {b2:.4f} (konstanta)")
    print(f"Pravá oblast (x > {xb2:.2f}°):  y = {k3:.4f}x + {b3:.4f}")
    print(f"Offset (posun v neutrálu):     {fixed_y_offset:.4f}°")
    print(f"Zlom vlevo:  [{xb1:.2f}°, {fixed_y_offset:.2f}°]")
    print(f"Zlom vpravo: [{xb2:.2f}°, {fixed_y_offset:.2f}°]")
    # ... (zbytek vykreslování zůstává identický, stačí jen přejmenovat proměnnou offset na fixed_y_offset)

    # --- VIZUALIZACE ---
    plt.figure(figsize=(9, 6))
    plt.scatter(x_data, y_data, color='red', zorder=5, label='Optimalizované body')

    # Generování bodů pro vykreslení spojité křivky
    x_left = np.linspace(min(x_data) - 2, xb1, 50)
    x_mid = np.linspace(xb1, xb2, 50)
    x_right = np.linspace(xb2, max(x_data) + 2, 50)

    # Zelená křivka
    plt.plot(x_left, k1 * x_left + b1, color='#00D154', linewidth=2.5, label='Aproximace (s mrtvou zónou)')
    plt.plot(x_mid, np.full_like(x_mid, fixed_y_offset), color='#00D154', linewidth=2.5)
    plt.plot(x_right, k3 * x_right + b3, color='#00D154', linewidth=2.5)

    # Zvýraznění bodů zlomů
    plt.scatter([xb1, xb2], [fixed_y_offset, fixed_y_offset], color='purple', s=70, zorder=6, label='Body zlomů')

    # Textový box s parametry
    info_text = (
        f"Offset: {fixed_y_offset:+.2f}°\n"
        f"Zlom L: [{xb1:.2f}°, {fixed_y_offset:.2f}°]\n"
        f"Zlom P: [{xb2:.2f}°, {fixed_y_offset:.2f}°]\n"
        f"Levá:  y = {k1:.3f}x {b1:+.2f}\n"
        f"Střed: y = {b2:+.2f} (Dead zone)\n"
        f"Pravá: y = {k3:.3f}x {b3:+.2f}"
    )
    
    plt.gca().text(0.03, 0.95, info_text, transform=plt.gca().transAxes,
                   fontsize=9, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.title('Kalibrace řízení: Aproximace s vodorovnou mrtvou zónou')
    plt.xlabel('Povel z logu (stupně)')
    plt.ylabel('Hledaný fyzický úhel simulace (stupně)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    plot_path = os.path.join(repo_root, 'Model_lib_and_sim', 'Identification', 'steering_calibration_result_deadzone.png')
    plt.savefig(plot_path, dpi=300)
    print(f"\nGraf uložen do: {plot_path}")

if __name__ == '__main__':
    # main()
    # file_path = os.path.join(repo_root, 'log_files', '18.10.2026', 'rst_20_log_1792330050.csv')
    # calibrate_log(file_path)
    # calibrate_all_logs()
    input_dir_path = os.path.join(repo_root, 'Model_lib_and_sim', 'Identification', 'Calibrated')
    merged_file_path = os.path.join(repo_root, 'Model_lib_and_sim', 'Identification', 'Calibrated', 'combined_calibrated_logs.csv')
    merge_calibrated_logs(input_dir_path, merged_file_path)
