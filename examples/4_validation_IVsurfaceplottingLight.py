import json
import glob
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource
from matplotlib.ticker import FixedLocator
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import zoom, gaussian_filter
from scipy.interpolate import interp1d

try:
    from heston_pricer.calibration import BatesCalibrator
    from heston_pricer.analytics import BatesAnalyticalPricer, implied_volatility
except ImportError:
    pass

def create_premium_cmap():
    colors = [
        "#000F3B",  # Base Blue
        "#1D408B",  # Vibrant Royal Blue
        "#367BDCFF", # Bright Cyan 
        "#AFDFFF",  # Ice White
        "#C8E9FF"   # Pure White 
    ]
    nodes = [0.0, 0.1, 0.45, 0.8, 1.0]
    base_cmap = mcolors.LinearSegmentedColormap.from_list("PremiumMatte", list(zip(nodes, colors)))
    
    values = np.linspace(0, 1.0, 256)
    warped_values = values ** 1.1 
    
    return mcolors.ListedColormap(base_cmap(warped_values), name="PremiumMatte_Warped")

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
        
        self.interp = interp1d(
            self.ts, self.rs, kind='linear', 
            bounds_error=False, fill_value=(self.rs[0], self.rs[-1])
        )

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
        
    files_sorted = sorted(files, key=os.path.getmtime, reverse=True)
    if index >= len(files_sorted):
        return None 
        
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
    return res_mc, "Analytical"

def plot_surface_professional(S0, r_curve, q_curve, params, ticker, filename, market_options, data_full, dropped_count, source_name):
    kappa, theta, xi, rho, v0 = params['kappa'], params['theta'], params['xi'], params['rho'], params['v0']
    lamb = params.get('lamb', 0.0)
    mu_j = params.get('mu_j', 0.0)
    sigma_j = params.get('sigma_j', 0.0)
    is_bates = lamb > 0.0

    LOWER_M, UPPER_M = 0.685, 1.315                    
    LOWER_T, UPPER_T = 0.04, 1.5 
    GRID_DENSITY = 550 
    COARSE_N = 150 

    print(f"-> Generating Surface for: {ticker}")
    print(f"   Model: {'Bates' if is_bates else 'Heston'}")
    print(f"   Calculating true gradient-based adaptive mesh...")
    
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

    mask = np.isnan(cZ)
    if np.any(mask):
        cZ = pd.DataFrame(cZ).interpolate(method='linear', axis=1, limit_direction='both') \
                             .interpolate(method='linear', axis=0, limit_direction='both').values

    DENSITY_POWER = 2.3
    if cZ is not None and hasattr(cZ, 'shape') and cZ.shape[0] >= 2 and cZ.shape[1] >= 2:
        try:
            dZ_dT, dZ_dM = np.gradient(cZ, c_T, c_M)
            grad_mag = np.sqrt(dZ_dT**2 + dZ_dM**2)
            max_grad = np.percentile(grad_mag, 92) 
            grad_mag = np.clip(grad_mag, 0, max_grad)
            dens_M = np.mean(grad_mag, axis=0) ** DENSITY_POWER
            dens_T = np.mean(grad_mag, axis=1) ** DENSITY_POWER
        except ValueError:
            dens_M, dens_T = np.ones(len(c_M)), np.ones(len(c_T))
    else:
        dens_M, dens_T = np.ones(len(c_M)), np.ones(len(c_T))

    def get_hybrid_spacing(density_array, mix_ratio=0.7):
        cdf_grad = np.cumsum(density_array)
        if cdf_grad[-1] - cdf_grad[0] == 0:
            cdf_grad = np.linspace(0, 1, len(density_array))
        else:
            cdf_grad = (cdf_grad - cdf_grad[0]) / (cdf_grad[-1] - cdf_grad[0])
            
        cdf_linear = np.linspace(0, 1, len(density_array))
        return (mix_ratio * cdf_grad) + ((1 - mix_ratio) * cdf_linear)

    cdf_M_final = get_hybrid_spacing(dens_M, mix_ratio=0.7)
    cdf_T_final = get_hybrid_spacing(dens_T, mix_ratio=0.7)
    
    uniform_space = np.linspace(0, 1, GRID_DENSITY)
    M_range = np.interp(uniform_space, cdf_M_final, c_M)
    T_range = np.interp(uniform_space, cdf_T_final, c_T)

    X, Y = np.meshgrid(M_range, T_range)
    Z = np.zeros_like(X)

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
    
    fig = plt.figure(figsize=(10, 7), facecolor='white') 
    ax = fig.add_subplot(111, projection='3d', facecolor='white')

    ls = LightSource(azdeg=270, altdeg=45)
    vmin, vmax = 0.1151, 0.72
    my_cmap = create_premium_cmap()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    surf = ax.plot_surface(
        X, Y, Z_smooth, cmap=my_cmap, 
        rcount=X.shape[0], ccount=X.shape[1], lightsource=ls, norm=norm,
        edgecolor='none', linewidth=0.2, alpha=0.85, 
        shade=True, antialiased=True, zorder=1, rasterized=True
    )
                               
    m = cm.ScalarMappable(cmap=my_cmap, norm=norm)
    m.set_array([])
    markersize = 5.2
    lbl_trigger = False
    color_above = "#ECE7D3"
    color_below = "#2B1600"

    if market_options:
        plot_opts = [o for o in market_options if (LOWER_M <= (o.strike/S0) <= UPPER_M) and (LOWER_T <= o.maturity <= UPPER_T)]
        
        valid_needles = 0
        is_spx = "SPX" in filename.upper()

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
            except: 
                continue

            valid_needles += 1
            is_above = iv_mkt >= iv_mod_exact
            dot_zorder = 10 if is_above else 1
            alpha_above = 0.98 
            alpha_below = 1.0 
            
            current_color = color_above if is_above else color_below
            current_alpha = alpha_above if is_above else alpha_below
            
            if is_above and not lbl_trigger:
                lbl = 'Market IV'
                lbl_trigger = True
            else: 
                lbl = ""
                
            if is_above:
                ax.plot([m_mkt], [t_mkt], [iv_mkt], 
                        marker='o', linestyle='None', color=current_color, 
                        markersize=markersize, markerfacecolor=current_color, 
                        markeredgecolor='none', alpha=current_alpha, 
                        zorder=dot_zorder + 1, label=lbl)

            condition1a = ((t_mkt < 0.06) & (m_mkt < 1.05) & (not is_spx) & (iv_mkt < 0.2656) & (not is_above)) 
            condition1b = ((iv_mkt > 0.2) & (iv_mkt < 0.35) & (t_mkt < 0.06) & (not is_spx) & (not is_above))
            condition2a = ((t_mkt < 0.06) & (m_mkt < 1.03) & is_spx & (iv_mkt < 0.1205) & (not is_above))
            condition2b = ((iv_mkt < 0.22) & (iv_mkt > 0.18) & (m_mkt < 1) & (t_mkt < 0.1) & is_spx & (not is_above))
            
            if condition1a or condition1b or condition2a or condition2b: 
                ax.plot([m_mkt], [t_mkt], [iv_mkt], 
                        marker='o', linestyle='None', color=color_above, 
                        markersize=markersize, markerfacecolor=color_above, 
                        markeredgecolor='none', alpha=alpha_above, 
                        zorder=dot_zorder + 1)
                        
                if condition1a or condition2a: 
                    ax.plot([m_mkt+0.001], [t_mkt + 0.001], [iv_mkt - 0.004], 
                            marker='o', linestyle='None', color="#140B00", 
                            markersize=markersize, markeredgecolor='none', 
                            alpha=0.6, zorder=9)
                    continue
                    
                if condition1b: 
                    ax.plot([m_mkt+0.001], [t_mkt + 0.005], [iv_mkt - 0.004], 
                            marker='o', linestyle='None', color="#483700", 
                            markersize=markersize, markeredgecolor='none', 
                            alpha=0.6, zorder=9)
                    continue
                    
                if condition2b: 
                    continue
                
            if not is_above:
                ax.plot([m_mkt], [t_mkt], [iv_mkt], 
                        marker='o', linestyle='None', color=current_color, 
                        markersize=markersize, markerfacecolor=current_color, 
                        markeredgecolor='none', alpha=current_alpha, 
                        zorder=dot_zorder + 1)

            alpha_needle = 0.65 if is_above else 0.95
            current_color_needle = "white" if is_above else color_below
            ax.plot([m_mkt, m_mkt], [t_mkt, t_mkt], [iv_mod_exact, iv_mkt], 
                    color=current_color_needle, linestyle='-', linewidth=0.8, 
                    alpha=alpha_needle, zorder=dot_zorder)

            if not is_above:
                ax.plot([m_mkt], [t_mkt], [iv_mkt], 
                        marker='o', linestyle='None', color="#322500FF", 
                        markersize=markersize, markeredgecolor='none', 
                        alpha=1.0, zorder=dot_zorder)
                ax.plot([m_mkt], [t_mkt], [iv_mkt], 
                        marker='o', linestyle='None', color="#231D00", 
                        markersize=markersize, markeredgecolor='none', 
                        alpha=0.5, zorder=9)
                
    ax.dist = 11  
    ax.set_xlim(LOWER_M, UPPER_M)
    ax.set_ylim(UPPER_T, LOWER_T) 
    ax.set_zlim(0.0, 0.75)
    
    grid_style = (0.68, 0.68, 0.68, 0.5) 
    linewidth_val = 0.575
    
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_pane_color((0.95, 0.95, 0.95, 1.0)) 
        axis.line.set_color("black")  
        axis.line.set_linewidth(0.8)
        axis._axinfo["grid"]['color'] = grid_style 
        axis._axinfo["grid"]['linewidth'] = linewidth_val

    ax.view_init(elev=28, azim=-115) 

    ax.set_xlabel('Moneyness ($K/S_0$)', color="black", labelpad=2.5, fontsize=13)
    ax.set_ylabel('Maturity ($T$ Years)', color="black", labelpad=9.75, fontsize=13)
    ax.set_zlabel(r'Implied Volatility', color="black", labelpad=11.75, fontsize=13)
    ax.tick_params(axis='both', which='major', colors='black', labelsize=12)
    ax.tick_params(axis='z', which='major', pad=6)
    ax.tick_params(axis='x', which='major', pad=-1)
    
    if market_options and valid_needles > 0:
        scatter_above = ax.scatter([], [], color=color_above, label=r"Market IV", s=30)
        leg = ax.legend(
            handles=[scatter_above], loc='upper left', bbox_to_anchor=(0.175, 0.79), 
            frameon=True, facecolor=(0.95, 0.95, 0.95), labelcolor="black", 
            handletextpad=0.5, edgecolor='none', fontsize=13, framealpha=1.0
        )
        for h in leg.legendHandles:
            h.set_edgecolor("black")
            h.set_linewidth(0.01)
        for handle in ax.get_legend().legend_handles:
            handle.set_alpha(0.95)

    cb_values = np.linspace(vmax, vmin, 256)
    cb_base_colors = my_cmap(norm(cb_values)) 

    cb_hsv = mcolors.rgb_to_hsv(cb_base_colors[:, :3])
    cb_hsv[:, 1] = np.clip(cb_hsv[:, 1] * 1.2, 0, 1) 
    cb_hsv[:, 2] = np.clip(cb_hsv[:, 2] * 1.05, 0, 1) 
    
    cb_rgb_vibrant = mcolors.hsv_to_rgb(cb_hsv)

    cb_rgba_final = np.zeros((256, 1, 4))
    cb_rgba_final[:, 0, :3] = cb_rgb_vibrant
    cb_rgba_final[:, 0, 3] = 1

    cbar = fig.colorbar(m, ax=ax, shrink=0.5, aspect=15, pad=-0.02)
    cbar.ax.clear()
    
    cbar.ax.imshow(
        cb_rgba_final, aspect='auto', extent=[0, 1, vmin, vmax], 
        origin='upper', interpolation='bilinear'
    )
    
    cbar.ax.set_rasterized(False) 
    cbar.ax.xaxis.set_visible(False)
    cbar.ax.set_frame_on(False)

    cbar.locator = FixedLocator(np.arange(0.1, 0.8, 0.1))
    cbar.update_ticks()
    cbar.ax.yaxis.set_tick_params(color="black", labelcolor="black", labelsize=12, width=0.5)
    cbar.outline.set_visible(False)
    cbar.ax.set_title("Model IV", color="black", fontsize=13, pad=9)
    
    fig.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98)

    save_path_vector = f"{filename}_surface_FINAL_LIGHT.pdf"
    plt.savefig(
        save_path_vector, format='pdf', bbox_inches='tight',    
        pad_inches=0.15, facecolor='white', dpi=800
    )

def main():
    num_to_plot = 2 
    for i in range(num_to_plot):
        try:
            print(f"\n--- Processing Artifact {i+1} ---")
            result = load_calibration_by_index(i) 
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