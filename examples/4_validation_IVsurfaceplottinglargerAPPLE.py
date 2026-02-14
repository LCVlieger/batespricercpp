import json
import glob
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource
from matplotlib.ticker import FixedLocator
import io
from PIL import Image

# Local package imports
try:
    from heston_pricer.calibration import BatesCalibrator
    from heston_pricer.analytics import BatesAnalyticalPricer, implied_volatility
except ImportError:
    pass

def create_gamma_cmap(base_cmap_name, gamma=0.5):
    base = cm.get_cmap(base_cmap_name)
    return mcolors.LinearSegmentedColormap.from_list(
        f'Warped_{base_cmap_name}',
        [base(x**gamma) for x in np.linspace(0, 1, 1024)] 
    )

# --- 1. COMPATIBILITY HELPERS ---
class RobustYieldCurve:
    def __init__(self, curve_data):
        times, rates = [], []
        if isinstance(curve_data, dict):
            for k, v in curve_data.items():
                try:
                    t_str = str(k).lower().replace("y", "").replace("week", "")
                    times.append(float(t_str))
                    rates.append(float(v))
                except: 
                    continue
        elif hasattr(curve_data, 'tenors'):
            times = curve_data.tenors
            rates = curve_data.rates
        else:
            times = [0.0, 30.0]
            rates = [float(curve_data), float(curve_data)]
            
        sorted_pairs = sorted(zip(times, rates))
        self.ts = np.array([p[0] for p in sorted_pairs])
        self.rs = np.array([p[1] for p in sorted_pairs])
        self.interp = interp1d(self.ts, self.rs, kind='linear', 
                               bounds_error=False, fill_value=(self.rs[0], self.rs[-1]))

    def get_rate(self, T):
        return float(self.interp(max(T, 1e-4)))

class ReconstructedOption:
    def __init__(self, strike, maturity, price, option_type="CALL"):
        self.strike = float(strike)
        self.maturity = float(maturity)
        self.market_price = float(price)
        self.option_type = str(option_type)

def load_calibration_by_index(index):
    patterns = ["results/*_meta.json", "*_meta.json"]
    files = []
    for p in patterns: 
        files.extend(glob.glob(p))
        
    if not files: 
        raise FileNotFoundError("No calibration meta file found.")
        
    # Sort by modification time, newest first
    files_sorted = sorted(files, key=os.path.getmtime, reverse=True)
    
    if index >= len(files_sorted):
        return None # Handle cases where fewer than 2 files exist
    latest_meta = files_sorted[index]
    base_name = latest_meta.replace("_meta.json", "")
    print(f"Loading Artifact: {base_name}...")

    with open(latest_meta, 'r') as f: 
        data = json.load(f)

    r_data = data['market'].get('r_sample', data['market'].get('r'))
    q_data = data['market'].get('q_sample', data['market'].get('q'))

    r_curve = RobustYieldCurve(r_data)
    q_curve = RobustYieldCurve(q_data)

    csv_file = f"{base_name}_prices.csv"
    market_options = []
    
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            k = row.get('K', row.get('Strike', 0))
            t = row.get('T', row.get('Maturity', 0))
            p = row.get('Market', row.get('Price', 0))
            otype = row.get('Type', "CALL")
            market_options.append(ReconstructedOption(k, t, p, otype))

    return data, r_curve, q_curve, market_options, base_name

def select_best_parameters(data):
    res_mc = data.get('analytical', {})
    return res_mc, "Monte Carlo"

# --- 2. EXACT PLOTTING FUNCTION ---
def plot_surface_professional(S0, r_curve, q_curve, params, ticker, filename, market_options, data_full, dropped_count, source_name):
    kappa, theta, xi, rho, v0 = params['kappa'], params['theta'], params['xi'], params['rho'], params['v0']
    lamb = params.get('lamb', 0.0)
    mu_j = params.get('mu_j', 0.0)
    sigma_j = params.get('sigma_j', 0.0)
    is_bates = lamb > 0.0

    # --- CONFIGURATION ---
    LOWER_M, UPPER_M = 0.685, 1.315 
    LOWER_T, UPPER_T = 0.04, 1.5 
    GRID_DENSITY = 550

    print(f"-> Generating Surface for: {ticker}")
    print(f"   Model: {'Bates' if is_bates else 'Heston'}")
    print(f"   Calculating true gradient-based adaptive mesh...")
    
    # ==========================================
    # --- PHASE 1: ROBUST HYBRID MESH ---
    # ==========================================
    
    # 1A. Coarse Pass (Scan the surface)
    COARSE_N = 150
    c_M = np.linspace(LOWER_M, UPPER_M, COARSE_N)
    c_T = np.linspace(LOWER_T, UPPER_T, COARSE_N)
    cX, cY = np.meshgrid(c_M, c_T)
    cZ = np.zeros_like(cX)

    for i in range(COARSE_N):
        for j in range(COARSE_N):
            T_val, M_val = cY[i, j], cX[i, j]
            r_T = r_curve.get_rate(T_val)
            q_T = q_curve.get_rate(T_val)
            try:
                prices = BatesAnalyticalPricer.price_european_call_vectorized(
                    S0, np.array([S0 * M_val]), np.array([T_val]), np.array([r_T]), np.array([q_T]),
                    kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j
                )
                iv = implied_volatility(float(prices[0]), S0, S0 * M_val, T_val, r_T, q_T, "CALL")
                cZ[i, j] = iv if 0.01 < iv < 2.5 else np.nan
            except:
                cZ[i, j] = np.nan

    # Robust NaN cleaning
    mask = np.isnan(cZ)
    if np.any(mask):
        cZ = pd.DataFrame(cZ).interpolate(method='linear', axis=1, limit_direction='both') \
                             .interpolate(method='linear', axis=0, limit_direction='both').values

    # 1B. Extract Gradient
    dZ_dT, dZ_dM = np.gradient(cZ, c_T, c_M)
    grad_mag = np.sqrt(dZ_dT**2 + dZ_dM**2)
    
    # Clip Singularity: T=0 has infinite gradient. We cap it to avoid breaking the weighting logic.
    max_grad = np.percentile(grad_mag, 92) 
    grad_mag = np.clip(grad_mag, 0, max_grad)
    
    # 1C. Density Function
    DENSITY_POWER = 2.3
    dens_M = np.mean(grad_mag, axis=0)**DENSITY_POWER
    dens_T = np.mean(grad_mag, axis=1)**DENSITY_POWER
    
    # Smooth slightly
    dens_M = gaussian_filter(dens_M, sigma=1.0)
    dens_T = gaussian_filter(dens_T, sigma=1.0)

    # 1D. MIXING STRATEGY (CRITICAL FIX)
    # To prevent "starvation" of the flat areas (which caused the faceting/tenting on the right),
    # we mix the Gradient CDF with a Linear CDF.
    # 70% Gradient-based (Detail where needed)
    # 30% Uniform (Guaranteed coverage everywhere)
    
    def get_hybrid_spacing(density_array, grid_points, mix_ratio=0.7):
        # 1. Gradient CDF
        cdf_grad = np.cumsum(density_array)
        if cdf_grad[-1] - cdf_grad[0] == 0:
            cdf_grad = np.linspace(0, 1, len(density_array))
        else:
            cdf_grad = (cdf_grad - cdf_grad[0]) / (cdf_grad[-1] - cdf_grad[0])
            
        # 2. Uniform CDF
        cdf_linear = np.linspace(0, 1, len(density_array))
        
        # 3. Mix
        cdf_final = (mix_ratio * cdf_grad) + ((1 - mix_ratio) * cdf_linear)
        
        return cdf_final

    cdf_M_final = get_hybrid_spacing(dens_M, COARSE_N, mix_ratio=0.7)
    cdf_T_final = get_hybrid_spacing(dens_T, COARSE_N, mix_ratio=0.7)
    
    uniform_space = np.linspace(0, 1, GRID_DENSITY)
    M_range = np.interp(uniform_space, cdf_M_final, c_M)
    T_range = np.interp(uniform_space, cdf_T_final, c_T)

    # 1E. Generate Final Mesh
    X, Y = np.meshgrid(M_range, T_range)
    Z = np.zeros_like(X)

    # ==========================================
    # --- PHASE 2: FINAL SURFACE CALCULATION ---
    # ==========================================

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            T_val, M_val = Y[i, j], X[i, j]
            r_T = r_curve.get_rate(T_val)
            q_T = q_curve.get_rate(T_val)
            try:
                prices = BatesAnalyticalPricer.price_european_call_vectorized(
                    S0, np.array([S0 * M_val]), np.array([T_val]), np.array([r_T]), np.array([q_T]),
                    kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j
                )
                iv = implied_volatility(float(prices[0]), S0, S0 * M_val, T_val, r_T, q_T, "CALL")
                Z[i, j] = iv if 0.01 < iv < 2.5 else np.nan
            except:
                Z[i, j] = np.nan

    mask = np.isnan(Z)
    if np.any(mask):
        Z = pd.DataFrame(Z).interpolate(method='linear', axis=1, limit_direction='both') \
                           .interpolate(method='linear', axis=0, limit_direction='both').values
        
    Z_smooth = gaussian_filter(Z, sigma=0.5)
    
    # --- PLOTTING ---
    with plt.style.context('dark_background'):
        fig = plt.figure(figsize=(10, 7), facecolor='black') 
        ax = fig.add_subplot(111, projection='3d', facecolor='black')

        ls = LightSource(azdeg=270, altdeg=45)
        vmin, vmax = 0.1151, 0.72
        my_cmap = create_gamma_cmap('RdYlBu_r', gamma=1.1)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        
        rgb = ls.shade(Z_smooth, cmap=my_cmap, norm=norm, vert_exag=0.1)
        
        # Added faint wireframe definition
        surf = ax.plot_surface(X, Y, Z_smooth, facecolors=rgb, cmap=my_cmap, 
                               rcount=X.shape[0], 
                               ccount=X.shape[1], 
                               edgecolor='none', linewidth=0.2, alpha=0.85, 
                               shade=False, antialiased=True, zorder=1, rasterized=True)
                               
        m = cm.ScalarMappable(cmap=my_cmap, norm=norm)
        m.set_array([])
        
        if market_options:
            plot_opts = [o for o in market_options 
                         if (LOWER_M <= (o.strike/S0) <= UPPER_M) and (LOWER_T <= o.maturity <= UPPER_T)]
            
            valid_needles = 0
            for opt in plot_opts:
                m_mkt, t_mkt = opt.strike / S0, opt.maturity
                try:
                    r_T_mkt = r_curve.get_rate(t_mkt)
                    q_T_mkt = q_curve.get_rate(t_mkt)
                    iv_mkt = implied_volatility(opt.market_price, S0, opt.strike, t_mkt, r_T_mkt, q_T_mkt, opt.option_type)
                    
                    prices_mod = BatesAnalyticalPricer.price_vectorized(
                        S0, np.array([opt.strike]), np.array([t_mkt]), np.array([r_T_mkt]), np.array([q_T_mkt]), np.array([opt.option_type]),
                        kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j
                    )
                    iv_mod_exact = implied_volatility(float(prices_mod[0]), S0, opt.strike, t_mkt, r_T_mkt, q_T_mkt, opt.option_type)
                    if iv_mkt < 0.01 or iv_mkt > 2.5: continue
                except: continue

                valid_needles += 1
                is_above = iv_mkt >= iv_mod_exact
                dot_zorder = 10 if is_above else 1

                ax.plot([m_mkt, m_mkt], [t_mkt, t_mkt], [iv_mod_exact, iv_mkt], 
                        color='white', linestyle='-', linewidth=0.8, alpha=0.65, zorder=dot_zorder)
                lbl = 'Market IV' if valid_needles == 1 else ""
                ax.plot([m_mkt, m_mkt], [t_mkt, t_mkt], [iv_mkt], 
                        marker='o', linestyle='None', color="#F0F0F0", markersize=4.62,
                        markerfacecolor='#F0F0F0', markeredgecolor='none', markeredgewidth=0.01,
                        alpha=0.85, zorder=dot_zorder + 1, label=lbl)
                        
        # --- AESTHETICS ---
        ax.dist = 11  
        ax.set_xlim(LOWER_M, UPPER_M)
        ax.set_ylim(UPPER_T, LOWER_T) 
        ax.set_zlim(0.0, 0.75)
        
        grid_style = (0.23, 0.23, 0.23, 0.75) 
        linewidth_val = 1.77
        
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_pane_color((0,0,0,1))
            axis.line.set_color("#D7D7D7")  
            axis.line.set_linewidth(0.8)
            axis._axinfo["grid"]['color'] = grid_style 
            axis._axinfo["grid"]['linewidth'] = linewidth_val

        ax.view_init(elev=28, azim=-115) 

        ax.set_xlabel('Moneyness ($K/S_0$)', color="#D7D7D7", labelpad=5, fontsize=11)
        ax.set_ylabel('Maturity ($T$ Years)', color="#D7D7D7", labelpad=5, fontsize=11)
        ax.set_zlabel(r'Implied Volatility', color="#D7D7D7", labelpad=6.75, fontsize=11)
        ax.tick_params(axis='both', which='major', colors='#D7D7D7', labelsize=10)

        if market_options and valid_needles > 0:
            ax.legend(loc='upper left', bbox_to_anchor=(0.175, 0.79), frameon=True, 
                      labelcolor="#D7D7D7", handletextpad=0.5, edgecolor='none', fontsize=10)
            leg = ax.get_legend()
            for handle in leg.legend_handles:
                handle.set_alpha(1)
                
        cbar = fig.colorbar(m, ax=ax, shrink=0.5, aspect=15, pad=-0.02, alpha=0.8)
        cbar.locator = FixedLocator(np.arange(0.1, 0.8, 0.1))
        cbar.update_ticks()
        cbar.ax.yaxis.set_tick_params(color="#D7D7D7", labelcolor="#D7D7D7", labelsize=10)
        cbar.outline.set_visible(False)
        cbar.ax.set_title("Model IV", color="#D7D7D7", fontsize=10, pad=9)
        
        fig.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98)

        save_path_vector = f"{filename}_surface_FINAL.pdf"
        plt.savefig(save_path_vector, format='pdf', bbox_inches='tight',    
                    pad_inches=0.15, facecolor='black', dpi=800)
        
        print(f"-> Saved True Vector: {save_path_vector}")

def main():
    # Define how many of the most recent calibrations you want to plot
    num_to_plot = 2 
    
    for i in range(num_to_plot):
        try:
            print(f"\n--- Processing Artifact {i+1} ---")
            result = load_calibration_by_index(i) # Use the updated sorting above
            if result is None:
                print(f"No file found for index {i}. Skipping.")
                continue
                
            data, r_curve, q_curve, market_options, base_name = result
            S0 = data['market']['S0']
            best_params, source_name = select_best_parameters(data)
            
            ticker = base_name.split("calibration_")[1].split("_")[0] if "calibration_" in base_name else "Asset"
            
            plot_surface_professional(
                S0, r_curve, q_curve, best_params, ticker, base_name, 
                market_options, data, 0, source_name
            )
        except Exception as e:
            print(f"[Error at index {i}] {e}")

if __name__ == "__main__":
    main()