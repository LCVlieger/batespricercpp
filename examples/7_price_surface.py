import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def visualize_price_surface(csv_file):
    """
    Visualizes the Price Surface (Model) vs Market Prices (Dots) with Error Needles.
    Automatically fetches S0 from the corresponding _meta.json file.
    Saves the output to a PNG file.
    """
    print(f"Generating Price Surface Plot from {csv_file}...")
    df = pd.read_csv(csv_file)

    # 1. Load S0 from the matching JSON file
    # Assumes file naming convention: results/name_prices.csv -> results/name_meta.json
    json_file = csv_file.replace("_prices.csv", "_meta.json")
    
    try:
        with open(json_file, 'r') as f:
            meta_data = json.load(f)
            s0 = meta_data['market']['S0']
            print(f"Loaded S0 from metadata: {s0:.4f}")
    except FileNotFoundError:
        print(f"Warning: Metadata file {json_file} not found. Estimating S0 from data...")
        # Fallback estimation if JSON is missing
        s0 = df['K'].median()
    
    # 2. Calculate Moneyness
    df['Moneyness'] = df['K'] / s0

    # 3. Setup Plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # --- Layer A: Model Surface ---
    # plot_trisurf creates a smooth surface from unstructured points
    surf = ax.plot_trisurf(
        df['Moneyness'], df['T'], df['Model'], 
        cmap=cm.viridis, 
        alpha=0.6, 
        edgecolor='none', 
        linewidth=0, 
        antialiased=True
    )
    
    # --- Layer B: Market Data (Dots) ---
    ax.scatter(
        df['Moneyness'], df['T'], df['Market'], 
        color='r', 
        s=20, 
        label='Market Prices', 
        depthshade=False
    )

    # --- Layer C: Error Needles ---
    # Draws a vertical line from Market Price to Model Price to show the "Miss"
    for i in range(len(df)):
        x = df['Moneyness'].iloc[i]
        y = df['T'].iloc[i]
        z_market = df['Market'].iloc[i]
        z_model = df['Model'].iloc[i]
        
        # Draw black lines (needles) with slight transparency
        ax.plot(
            [x, x], [y, y], [z_market, z_model], 
            color='black', 
            alpha=0.5, 
            linewidth=1
        )

    # 4. Styling
    ax.set_xlabel('Moneyness ($K/S_0$)')
    ax.set_ylabel('Time to Maturity ($T$)')
    ax.set_zlabel('Option Price')
    ax.set_title(f'Price Surface Calibration\nModel (Surface) vs Market (Dots)\n$S_0 = {s0:.2f}$')
    
    # Rotate view to see the "hockey stick" profile
    ax.view_init(elev=30, azim=-120)

    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label('Model Price', rotation=270, labelpad=15)

    plt.tight_layout()
    
    # --- 5. SAVE & SHOW ---
    save_path = csv_file.replace(".csv", "_price_surface.png")
    print(f"Saving plot to: {save_path}")
    plt.savefig(save_path, dpi=300)  # High resolution save
    
    plt.show()

# --- Usage Example ---
visualize_price_surface("results/calibration_^SPX_20260204_223026_prices.csv")