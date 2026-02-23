import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import glob
import os

def visualize_price_surface_final(csv_file):
    print(f"Generating Final Polished Plot from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    base_name = csv_file.replace("_prices.csv", "")
    json_file = f"{base_name}_meta.json"
    s0 = df['K'].median()
    
    try:
        with open(json_file, 'r') as f:
            meta = json.load(f)
            s0 = meta['market']['S0']
    except Exception:
        pass

    df['Moneyness'] = df['K'] / s0

    fig = plt.figure(figsize=(10, 7), facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')
    ax.computed_zorder = False
    
    ax.scatter(
        df['Moneyness'], df['T'], df['Market'], 
        color='r', s=25, label='Market Price', 
        depthshade=False, zorder=1
    )
    
    surf = ax.plot_trisurf(
        df['Moneyness'], df['T'], df['Model'], 
        cmap=cm.viridis, alpha=0.6, 
        edgecolor='none', linewidth=0, 
        antialiased=True, zorder=3
    )

    for x, y, z_mkt, z_mod in zip(df['Moneyness'], df['T'], df['Market'], df['Model']):
        ax.plot([x, x], [y, y], [z_mkt, z_mod], color='black', alpha=0.3, linewidth=0.8, zorder=2)

    ax.legend(
        loc='upper left', 
        bbox_to_anchor=(0.09935, 0.799), 
        frameon=True, 
        edgecolor='none', 
        framealpha=1,
        facecolor=(0.95, 0.95, 0.95),
        labelcolor="black", 
        fontsize=10,
    )

    ax.dist = 11
    ax.set_xlabel('Moneyness ($K/S_0$)', labelpad=5, fontsize=11)
    ax.set_ylabel('Maturity ($T$ Years)', labelpad=5, fontsize=11)
    ax.set_zlabel('Option Price ($)', labelpad=5, fontsize=11)

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"].update({'linewidth': 0.575, 'alpha': 1})
        for label in axis.get_ticklabels():
            label.set_fontsize(11)

    ax.view_init(elev=30, azim=-120)
    fig.subplots_adjust(left=0.25, bottom=0.05, right=0.95, top=0.95)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=-0.01)
    cbar.ax.set_title("Model Price", fontsize=10, pad=9)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=10)

    save_path = csv_file.replace(".csv", "_price_surface_final.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.22)
    print(f"-> Saved True Vector PDF: {save_path}")

if __name__ == "__main__":
    files = glob.glob("results/calibration_*_prices.csv") or glob.glob("calibration_*_prices.csv")
        
    if files:
        files_sorted = sorted(files, key=os.path.getctime, reverse=True)
        visualize_price_surface_final(files_sorted[0])
    else:
        print("No calibration files found.")