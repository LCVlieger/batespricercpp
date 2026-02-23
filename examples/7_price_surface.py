import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import glob
import os
from matplotlib.transforms import Bbox
import matplotlib.patheffects as path_effects

def visualize_price_surface_final(csv_file):
    print(f"Generating Final Polished Plot from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    base_name = csv_file.replace("_prices.csv", "")
    json_file = f"{base_name}_meta.json"
    
    s0, ticker, params = df['K'].median(), "Asset", {}
    
    try:
        if "calibration_" in base_name:
            ticker = base_name.split("calibration_")[1].split("_")[0]
        with open(json_file, 'r') as f:
            meta = json.load(f)
            s0 = meta['market']['S0']
            params = meta.get('analytical', {})
    except Exception: pass

    df['Moneyness'] = df['K'] / s0

    fig = plt.figure(figsize=(14, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')
    ax.computed_zorder = False
    
    ax.scatter(df['Moneyness'], df['T'], df['Market'], color='r', s=13, label='Market Price', depthshade=False, zorder=1)
    
    surf = ax.plot_trisurf(df['Moneyness'], df['T'], df['Model'], cmap=cm.viridis, alpha=0.6, edgecolor='none', linewidth=0, antialiased=True, zorder=3)

    for x, y, z_mkt, z_mod in zip(df['Moneyness'], df['T'], df['Market'], df['Model']):
        ax.plot([x, x], [y, y], [z_mkt, z_mod], color='black', alpha=0.3, linewidth=0.8, zorder=2)

    ax.legend(loc='upper left', bbox_to_anchor=(0.131, 0.81), frameon=False, labelcolor="black", fontsize=11)

    k, th, xi, rho, v0 = params.get('kappa', 0.0), params.get('theta', 0.0), params.get('xi', 0.0), params.get('rho', 0.0), params.get('v0', 0.0)
    la, mj, sj = params.get('lamb', 0.0), params.get('mu_j', 0.0), params.get('sigma_j', 0.0)

    title = fig.text(0.5897, 0.843, f"Bates Calibration Price Surface: {ticker}", fontsize=18, fontweight='bold', family='monospace', ha='center', color='black')
    title.set_path_effects([path_effects.withStroke(linewidth=0.5, foreground='black')])
    
    subtitle = (rf"$\kappa={k:.2f}, \theta={th:.2f}, \xi={xi:.2f}, \rho={rho:.2f}, v_0={v0:.3f}$" + "\n" +
                rf"$\lambda={la:.2f}, \mu_J={mj:.2f}, \sigma_J={sj:.2f}, S_0={s0:.2f}$")
    fig.text(0.5815, 0.7989, subtitle, fontsize=10, family='monospace', ha='center', color='#555555')

    ax.dist = 14
    ax.set_xlabel('Moneyness ($K/S_0$)', labelpad=10)
    ax.set_ylabel('Maturity ($T$ Years)', labelpad=10)
    ax.set_zlabel('Option Price ($)', labelpad=10)

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"].update({'linewidth': 0.575, 'alpha': 1})
        for label in axis.get_ticklabels(): label.set_family('monospace')

    ax.view_init(elev=30, azim=-120)
    fig.subplots_adjust(left=0.25, bottom=0.05, right=0.95, top=0.85)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.01)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=9)
    for l in cbar.ax.yaxis.get_ticklabels(): l.set_family('monospace')

    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    save_path = csv_file.replace(".csv", "_price_surface_final.png")
    plt.savefig(save_path, dpi=300, bbox_inches=Bbox([[bbox.x0 - 0.15, bbox.y0], [bbox.x1 + 0.075, bbox.y1 + 0.15]]))
    print(f"-> Saved: {save_path}")

if __name__ == "__main__":
    files = glob.glob("results/calibration_*_prices.csv") or glob.glob("calibration_*_prices.csv")
    if files:
        files_sorted = sorted(files, key=os.path.getctime, reverse=True)
        visualize_price_surface_final(files_sorted[0])
    else:
        print("No calibration files found.")