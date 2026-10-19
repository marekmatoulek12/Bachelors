import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
import pandas as pd  # Přidáno pro ukládání strukturovaných CSV
import os

warnings.filterwarnings("ignore", category=RuntimeWarning)

from twintrack_model_modified import TwinTrackModel, TwinTrackState

# ------------------------------------------------------------------
# Dynamika systému - 3 proměnné s normalizací na Síly a Momenty
# ------------------------------------------------------------------
def model_eq_3(x, vx, df, model):
    vy, yaw_rate, omega_rear = x
    p = model.params
    
    wy_fl = p.front_track_width / 2.0
    v_wx_fl = vx - yaw_rate * wy_fl
    v_wy_fl = vy + yaw_rate * p.front_axle_len
    v_long_fl = np.cos(df) * v_wx_fl + np.sin(df) * v_wy_fl
    omega_fl = v_long_fl / p.front_wheel_radius
    
    wy_fr = -p.front_track_width / 2.0
    v_wx_fr = vx - yaw_rate * wy_fr
    v_wy_fr = vy + yaw_rate * p.front_axle_len
    v_long_fr = np.cos(df) * v_wx_fr + np.sin(df) * v_wy_fr
    omega_fr = v_long_fr / p.front_wheel_radius

    state = TwinTrackState(
        vx=vx, vy=vy, yaw_rate=yaw_rate,
        omega_fl=omega_fl, omega_fr=omega_fr,
        omega_rl=omega_rear, omega_rr=omega_rear
    )
    
    state_dot, _ = model.derivatives(state, steer=df)
    
    return np.array([
        state_dot.vx * p.mass,
        state_dot.vy * p.mass,
        state_dot.yaw_rate * p.moment_of_inertia
    ])

# ------------------------------------------------------------------
# Řešič
# ------------------------------------------------------------------
def solve_point_lsq(x0, vx, df, model, lb, ub, resid_tol=1e-1):
    x0_clipped = np.clip(x0, lb, ub)
    
    res = least_squares(model_eq_3, x0_clipped, args=(vx, df, model), bounds=(lb, ub), 
                        method='trf', ftol=1e-8, xtol=1e-10, max_nfev=1000,
                        x_scale='jac', diff_step=1e-3)
    
    ok = res.success and np.linalg.norm(res.fun) <= resid_tol
    x = res.x
    
    if not ok:
        x0_alt = np.clip(x0_clipped + np.array([0.5, 2.0, 5.0]), lb, ub)
        res2 = least_squares(model_eq_3, x0_alt, args=(vx, df, model), bounds=(lb, ub), 
                             method='trf', ftol=1e-8, xtol=1e-10, max_nfev=1000,
                             x_scale='jac', diff_step=1e-3)
        ok2 = res2.success and np.linalg.norm(res2.fun) <= resid_tol
        if ok2 or np.linalg.norm(res2.fun) < np.linalg.norm(res.fun):
            x = res2.x
            ok = ok2
            
    return x, ok

def get_bounds(vx, model):
    omega_free = vx / model.params.rear_wheel_radius
    lb = np.array([-15.0, -40.0, 0.1 * omega_free])
    ub = np.array([ 15.0,  40.0, 3.0 * omega_free])
    return lb, ub

# ------------------------------------------------------------------
# Výpočet stabilní větve: Analytický Ackermannův odhad
# ------------------------------------------------------------------
def get_ackermann_guess(vx, df, model):
    p = model.params
    L = p.front_axle_len + p.rear_axle_len
    
    yaw_rate = (vx / L) * np.tan(df)
    vy = p.rear_axle_len * yaw_rate
    omega_rear = vx / p.rear_wheel_radius
    
    return np.array([vy, yaw_rate, omega_rear])

def solve_stable_layer(vx, df_vec, model, resid_tol):
    n = len(df_vec)
    layer = np.zeros((n, 3))
    conv = np.ones(n, dtype=bool)
    lb, ub = get_bounds(vx, model)
    
    for i in range(n):
        x0 = get_ackermann_guess(vx, df_vec[i], model)
        layer[i], conv[i] = solve_point_lsq(x0, vx, df_vec[i], model, lb, ub, resid_tol)
        
    return layer, conv

# ------------------------------------------------------------------
# Výpočet nestabilních větví
# ------------------------------------------------------------------
def solve_seed_layer_unstable(vx, df_vec, model, resid_tol, branch_type):
    n = len(df_vec)
    layer = np.zeros((n, 3))
    conv = np.ones(n, dtype=bool)
    lb, ub = get_bounds(vx, model)
    omega_free = vx / model.params.rear_wheel_radius
    
    x0_base = np.array([-vx * 0.4, 15.0, omega_free * 1.1]) if branch_type == 1 else np.array([vx * 0.4, -15.0, omega_free * 1.1])
        
    for i in range(n):
        if i > 1:
            x0 = 2 * layer[i-1] - layer[i-2]
        elif i == 1:
            x0 = layer[0]
        else:
            x0 = x0_base
        layer[i], conv[i] = solve_point_lsq(x0, vx, df_vec[i], model, lb, ub, resid_tol)
    return layer, conv

def solve_layer_seq_unstable(vx, df_vec, model, resid_tol, prevLayer, prevPrevLayer):
    n = len(df_vec)
    layer = np.zeros((n, 3))
    conv = np.ones(n, dtype=bool)
    lb, ub = get_bounds(vx, model)
    
    for i in range(n):
        if prevPrevLayer is None:
            x0 = prevLayer[i]
        else:
            x0 = 2 * prevLayer[i] - prevPrevLayer[i]
        layer[i], conv[i] = solve_point_lsq(x0, vx, df_vec[i], model, lb, ub, resid_tol)
    return layer, conv

# ------------------------------------------------------------------
# Funkce pro export tabulek
# ------------------------------------------------------------------
def save_branch_to_csv(vx_vec, df_vec, Z_matrix, filename, folder='lookups'):
    """Vytvoří 2D vyhledávací tabulku pro konkrétní větev a uloží ji do složky."""
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    df_deg = np.round(np.rad2deg(df_vec), 2)
    # Z_matrix má tvar (nDf, nVx), Pandas vyžaduje řádky=vx, sloupce=df, proto .T (transpozice)
    df_out = pd.DataFrame(Z_matrix.T, index=np.round(vx_vec, 2), columns=df_deg)
    df_out.index.name = 'vx_m_s'
    df_out.columns.name = 'df_deg'
    
    file_path = os.path.join(folder, filename)
    df_out.to_csv(file_path)

# ------------------------------------------------------------------
# Spuštění
# ------------------------------------------------------------------
def main():
    model = TwinTrackModel()
    RESID_TOL = 1e-1
    
    df1 = np.deg2rad(np.linspace(-25, 7, 65))
    df2 = np.deg2rad(np.linspace(-7, 7, 29))
    df3 = np.deg2rad(np.linspace(25, -7, 65))

    nDf1, nDf2, nDf3 = len(df1), len(df2), len(df3)

    vx_vec = np.linspace(4, 14.5, 30)
    vx_seed = 8.0
    seed_idx = np.argmin(np.abs(vx_vec - vx_seed))
    nVx = len(vx_vec)

    results1 = np.zeros((nDf1, 3, nVx))
    results2 = np.zeros((nDf2, 3, nVx))
    results3 = np.zeros((nDf3, 3, nVx))
    
    conv1 = np.ones((nDf1, nVx), dtype=bool)
    conv2 = np.ones((nDf2, nVx), dtype=bool)
    conv3 = np.ones((nDf3, nVx), dtype=bool)

    print("Počítám semínkovou vrstvu (vx = 8.0)...")
    vx = vx_vec[seed_idx]
    results2[:, :, seed_idx], conv2[:, seed_idx] = solve_stable_layer(vx, df2, model, RESID_TOL)
    results1[:, :, seed_idx], conv1[:, seed_idx] = solve_seed_layer_unstable(vx, df1, model, RESID_TOL, branch_type=1)
    results3[:, :, seed_idx], conv3[:, seed_idx] = solve_seed_layer_unstable(vx, df3, model, RESID_TOL, branch_type=3)

    print("Šířím vrstvy pro vyšší rychlosti (vx > 8.0)...")
    for k in range(seed_idx + 1, nVx):
        vx = vx_vec[k]
        p21 = results1[:, :, k-2] if k >= seed_idx + 2 else None
        p23 = results3[:, :, k-2] if k >= seed_idx + 2 else None

        results1[:, :, k], conv1[:, k] = solve_layer_seq_unstable(vx, df1, model, RESID_TOL, results1[:, :, k-1], p21)
        results2[:, :, k], conv2[:, k] = solve_stable_layer(vx, df2, model, RESID_TOL)
        results3[:, :, k], conv3[:, k] = solve_layer_seq_unstable(vx, df3, model, RESID_TOL, results3[:, :, k-1], p23)

    print("Šířím vrstvy pro nižší rychlosti (vx < 8.0)...")
    for k in range(seed_idx - 1, -1, -1):
        vx = vx_vec[k]
        p21 = results1[:, :, k+2] if k <= seed_idx - 2 else None
        p23 = results3[:, :, k+2] if k <= seed_idx - 2 else None

        results1[:, :, k], conv1[:, k] = solve_layer_seq_unstable(vx, df1, model, RESID_TOL, results1[:, :, k+1], p21)
        results2[:, :, k], conv2[:, k] = solve_stable_layer(vx, df2, model, RESID_TOL)
        results3[:, :, k], conv3[:, k] = solve_layer_seq_unstable(vx, df3, model, RESID_TOL, results3[:, :, k+1], p23)

    # Data příprava a Filtrace
    DF1, VX1 = np.meshgrid(np.rad2deg(df1), vx_vec, indexing='ij')
    DF2, VX2 = np.meshgrid(np.rad2deg(df2), vx_vec, indexing='ij')
    DF3, VX3 = np.meshgrid(np.rad2deg(df3), vx_vec, indexing='ij')

    VY1, VY2, VY3 = results1[:, 0, :].copy(), results2[:, 0, :].copy(), results3[:, 0, :].copy()
    R1,  R2,  R3  = results1[:, 1, :].copy(), results2[:, 1, :].copy(), results3[:, 1, :].copy()
    W1,  W2,  W3  = results1[:, 2, :].copy(), results2[:, 2, :].copy(), results3[:, 2, :].copy()

    for data in [VY1, R1, W1]: data[~conv1] = np.nan
    for data in [VY2, R2, W2]: data[~conv2] = np.nan
    for data in [VY3, R3, W3]: data[~conv3] = np.nan

    BETA1 = np.rad2deg(np.arctan2(VY1, VX1))
    BETA2 = np.rad2deg(np.arctan2(VY2, VX2))
    BETA3 = np.rad2deg(np.arctan2(VY3, VX3))
    
    # Export do 12 CSV tabulek ve složce 'lookups'
    print("Ukládám data do 12 CSV tabulek ve složce 'lookups'...")
    
    # Stabilní větev
    save_branch_to_csv(vx_vec, df2, BETA2, 'beta_stable.csv')
    save_branch_to_csv(vx_vec, df2, VY2, 'vy_stable.csv')
    save_branch_to_csv(vx_vec, df2, R2, 'yaw_rate_stable.csv')
    save_branch_to_csv(vx_vec, df2, W2, 'omega_rear_stable.csv')

    # Nestabilní větev 1 (záporné úhly / drift)
    save_branch_to_csv(vx_vec, df1, BETA1, 'beta_drift1.csv')
    save_branch_to_csv(vx_vec, df1, VY1, 'vy_drift1.csv')
    save_branch_to_csv(vx_vec, df1, R1, 'yaw_rate_drift1.csv')
    save_branch_to_csv(vx_vec, df1, W1, 'omega_rear_drift1.csv')

    # Nestabilní větev 3 (kladné úhly / drift)
    save_branch_to_csv(vx_vec, df3, BETA3, 'beta_drift3.csv')
    save_branch_to_csv(vx_vec, df3, VY3, 'vy_drift3.csv')
    save_branch_to_csv(vx_vec, df3, R3, 'yaw_rate_drift3.csv')
    save_branch_to_csv(vx_vec, df3, W3, 'omega_rear_drift3.csv')
    
    print("Tabulky byly úspěšně vytvořeny.")

    # ------------------------------------------------------------------
    # Vykreslení (Bodový mrak / Scatter plot)
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor('white')

    def plot_scatter(ax, Z1, Z2, Z3, z_label):
        for X, Y, Z, color in zip([DF1, DF2, DF3], [VX1, VX2, VX3], [Z1, Z2, Z3], ['r', 'b', 'r']):
            mask = ~np.isnan(Z)
            ax.scatter(X[mask], Y[mask], Z[mask], c=color, s=15, alpha=0.6, edgecolors='none')
            
        ax.set_xlabel('$\delta_f$ (deg)')
        ax.set_ylabel('$v_x$ (m/s)')
        ax.set_zlabel(z_label)
        ax.view_init(elev=30, azim=-37.5)

    ax1 = fig.add_subplot(221, projection='3d')
    plot_scatter(ax1, BETA1, BETA2, BETA3, '$\\beta$ (deg)')

    ax2 = fig.add_subplot(222, projection='3d')
    plot_scatter(ax2, VY1, VY2, VY3, '$v_y$ (m/s)')

    ax3 = fig.add_subplot(223, projection='3d')
    plot_scatter(ax3, R1, R2, R3, 'Yaw rate r (rad/s)')

    ax4 = fig.add_subplot(224, projection='3d')
    plot_scatter(ax4, W1, W2, W3, '$\omega_{rear}$ (rad/s)')

    plt.suptitle('RC Model equilibria surfaces (Scatter Points + CSV Export)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()