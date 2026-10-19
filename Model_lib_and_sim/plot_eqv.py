import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def load_and_plot_equilibria():
    folder = 'lookups'
    if not os.path.exists(folder):
        print(f"Chyba: Složka '{folder}' neexistuje.")
        return

    # Definice větví k načtení a jejich příslušných barev
    branches = ['drift1', 'stable', 'drift3']
    colors = ['r', 'b', 'r'] 
    
    prefixes = ['beta', 'vy', 'yaw_rate', 'omega_rear']
    labels = ['$\\beta$ (deg)', '$v_y$ (m/s)', 'Yaw rate r (rad/s)', '$\omega_{rear}$ (rad/s)']

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor('white')
    
    axes = [
        fig.add_subplot(221, projection='3d'),
        fig.add_subplot(222, projection='3d'),
        fig.add_subplot(223, projection='3d'),
        fig.add_subplot(224, projection='3d')
    ]

    print("Načítám data ze složky 'lookups' a vykresluji...")
    
    for branch, color in zip(branches, colors):
        for ax, prefix, label in zip(axes, prefixes, labels):
            file_path = os.path.join(folder, f"{prefix}_{branch}.csv")
            
            if not os.path.exists(file_path):
                print(f"Upozornění: Soubor {file_path} chybí, přeskakuji.")
                continue
                
            # Načtení matice (řádky = vx, sloupce = df)
            df_data = pd.read_csv(file_path, index_col=0)
            
            # Rekonstrukce os (musíme převést názvy sloupců zpět na float)
            vx_vals = df_data.index.values
            df_vals = df_data.columns.values.astype(float)
            DF, VX = np.meshgrid(df_vals, vx_vals)
            
            Z = df_data.values
            valid = ~np.isnan(Z)
            
            # Vykreslení
            ax.scatter(DF[valid], VX[valid], Z[valid], c=color, s=15, alpha=0.6, edgecolors='none')
            
            # Nastavení popisků os pouze v prvním průchodu
            if branch == branches[0]:
                ax.set_xlabel('$\delta_f$ (deg)')
                ax.set_ylabel('$v_x$ (m/s)')
                ax.set_zlabel(label)
                ax.view_init(elev=30, azim=-37.5)

    plt.suptitle('RC Model equilibria - Vykresleno z 12 CSV souborů', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    load_and_plot_equilibria()