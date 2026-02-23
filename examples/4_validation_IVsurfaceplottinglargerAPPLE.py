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
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

try:
    from heston_pricer.calibration import BatesCalibrator
    from heston_pricer.analytics import BatesAnalyticalPricer, implied_volatility
except ImportError:
    pass

def create_premium_cmap(base_cmap_name):
    base = cm.get_cmap(base_cmap_name)
    N = 256
    values = np.linspace(0.007125, 0.6485, N)
    
    t1, t2 = 0.1, 0.21
    g_floor, g_slope, g_peak = 1.3, 0.78, 0.7
    
    warped_values = np.zeros_like(values)

    mask1 = values <= t1
    s1 = values[mask1] / t1
    warped_values[mask1] = (s1 ** g_floor) * t1

    mask2 = (values > t1) & (values <= t2)
    if np.any(mask2):
        s2 = (values[mask2] - t1) / (t2 - t1)
        s2_smooth = s2 * s2 * s2 * (s2 * (s2 * 6.0 - 15.0) + 10.0)
        warped_values[mask2] = np.power(s2_smooth, 0.3) * (t2 - t1) + t1

    mask3 = values > t2
    if np.any(mask3):
        v_max = values.max()
        s3 = np.maximum(0, (values[mask3] - t2) / (v_max - t2))
        warped_values[mask3] = np.sin(s3 * np.pi / 2.0) * (v_max - t2) + t2

    return mcolors.ListedColormap(base(np.clip(warped_values, 0, 1)), name='Premium_Warped')

class RobustYieldCurve:
    def __init__(self, curve_data):
        times, rates = [], []
        if isinstance(curve_data, dict):
            for k, v in curve_data.items():
                try:
                    t_str = str(k).lower().replace("y", "").replace("week", "")
                    times.append(float(t_str))
                    rates.append(float(v))
                except: continue
        elif hasattr(curve_data, 'tenors'):
            times, rates = curve_data.tenors, curve_data.rates
        else:
            times, rates = [0.0, 30.0], [float(curve_data)] * 2
            
        sorted_pairs = sorted(zip(times, rates))
        self.ts, self.rs = np.array([p[0] for p in sorted_pairs]), np.array([p[1] for p in sorted_pairs])
        self.interp = interp1d(self.ts, self.rs, kind='linear', bounds_error=False, fill_value=(self.rs[0], self.rs[-1]))

    def get_rate(self, T):
        return float(self.interp(max(T, 1e-4)))

class ReconstructedOption:
    def __init__(self, strike, maturity, price, option_type="CALL"):
        self.strike, self.maturity, self.market_price = float(strike), float(maturity), float(price)
        self.option_type = str(option_type)

def load_calibration_by_index(index):
    files = glob.glob("results/*_meta.json") + glob.glob("*_meta.json")
    if not files: raise FileNotFoundError("No calibration meta file found.")
    
    files_sorted = sorted(files, key=os.path.getmtime, reverse=True)
    if index >= len(files_sorted): return None 
    
    meta_path = files_sorted[index]
    with open(meta_path, 'r') as f: data = json.load(f)

    r_curve = RobustYieldCurve(data['market'].get('r_sample', data['market'].get('r')))
    q_curve = RobustYieldCurve(data['market'].get('q_sample', data['market'].get('q')))

    market_options = []
    csv_file = meta_path.replace("_meta.json", "_prices.csv")
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            market_options.append(ReconstructedOption(row.get('K', 0), row.get('T', 0), row.get('Market', 0), row.get('Type', "CALL")))

    return data, r_curve, q_curve, market_options, meta_path.replace("_meta.json", "")

def plot_surface_professional(S0, r_curve, q_curve, params, ticker, filename, market_options, data_full, dropped_count, source_name):
    kappa, theta, xi, rho, v0 = params['kappa'], params['theta'], params['xi'], params['rho'], params['v0']
    lamb, mu_j, sigma_j = params.get('lamb', 0.0), params.get('mu_j', 0.0), params.get('sigma_j', 0.0)

    LOWER_M, UPPER_M, LOWER_T, UPPER_T = 0.685, 1.315, 0.04, 1.5 
    GRID_DENSITY = 60 

    c_M, c_T = np.linspace(LOWER_M, UPPER_M, 30), np.linspace(LOWER_T, UPPER_T, 30)
    cX, cY = np.meshgrid(c_M, c_T)
    cZ = np.zeros_like(cX)

    for i in range(30):
        for j in range(30):
            r_T, q_T = r_curve.get_rate(cY[i, j]), q_curve.get_rate(cY[i, j])
            try:
                prices = BatesAnalyticalPricer.price_european_call_vectorized(S0, np.array([S0 * cX[i, j]]), np.array([cY[i, j]]), np.array([r_T]), np.array([q_T]), kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j)
                iv = implied_volatility(float(prices[0]), S0, S0 * cX[i, j], cY[i, j], r_T, q_T, "CALL")
                cZ[i, j] = iv if 0.01 < iv < 2.5 else np.nan
            except: cZ[i, j] = np.nan

    cZ = pd.DataFrame(cZ).interpolate(method='linear', axis=1, limit_direction='both').interpolate(method='linear', axis=0, limit_direction='both').values
    dZ_dT, dZ_dM = np.gradient(cZ, c_T, c_M)
    grad_mag = np.clip(np.sqrt(dZ_dT**2 + dZ_dM**2), 0, np.percentile(np.sqrt(dZ_dT**2 + dZ_dM**2), 92))
    
    cdf_M = np.cumsum(np.mean(grad_mag, axis=0)**2.3)
    cdf_T = np.cumsum(np.mean(grad_mag, axis=1)**2.3)
    
    M_range = np.interp(np.linspace(0, 1, GRID_DENSITY), (cdf_M - cdf_M[0])/(cdf_M[-1]-cdf_M[0]), c_M)
    T_range = np.interp(np.linspace(0, 1, GRID_DENSITY), (cdf_T - cdf_T[0])/(cdf_T[-1]-cdf_T[0]), c_T)

    X, Y = np.meshgrid(M_range, T_range)
    Z = np.zeros_like(X)

    for i in range(GRID_DENSITY):
        for j in range(GRID_DENSITY):
            r_T, q_T = r_curve.get_rate(Y[i, j]), q_curve.get_rate(Y[i, j])
            try:
                prices = BatesAnalyticalPricer.price_european_call_vectorized(S0, np.array([S0 * X[i, j]]), np.array([Y[i, j]]), np.array([r_T]), np.array([q_T]), kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j)
                iv = implied_volatility(float(prices[0]), S0, S0 * X[i, j], Y[i, j], r_T, q_T, "CALL")
                Z[i, j] = iv if 0.01 < iv < 2.5 else np.nan
            except: Z[i, j] = np.nan

    Z = pd.DataFrame(Z).interpolate(method='linear', axis=1, limit_direction='both').interpolate(method='linear', axis=0, limit_direction='both').values
    Z_smooth = gaussian_filter(Z, sigma=0.5)
    
    with plt.style.context('dark_background'):
        fig = plt.figure(figsize=(10, 7), facecolor='black') 
        ax = fig.add_subplot(111, projection='3d', facecolor='black')
        my_cmap = create_premium_cmap('RdYlBu_r')
        norm = mcolors.Normalize(vmin=0.1151, vmax=0.72)
        rgb = LightSource(azdeg=270, altdeg=45).shade(Z_smooth, cmap=my_cmap, norm=norm, vert_exag=0.1)
        
        ax.plot_surface(X, Y, Z_smooth, facecolors=rgb, cmap=my_cmap, rcount=GRID_DENSITY, ccount=GRID_DENSITY, edgecolor='none', alpha=0.85, antialiased=True, zorder=1)
        
        if market_options:
            for opt in [o for o in market_options if (LOWER_M <= (o.strike/S0) <= UPPER_M) and (LOWER_T <= o.maturity <= UPPER_T)]:
                try:
                    rt, qt = r_curve.get_rate(opt.maturity), q_curve.get_rate(opt.maturity)
                    iv_m = implied_volatility(opt.market_price, S0, opt.strike, opt.maturity, rt, qt, opt.option_type)
                    p_mod = float(BatesAnalyticalPricer.price_vectorized(S0, np.array([opt.strike]), np.array([opt.maturity]), np.array([rt]), np.array([qt]), np.array([opt.option_type]), kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j)[0])
                    iv_mod = implied_volatility(p_mod, S0, opt.strike, opt.maturity, rt, qt, opt.option_type)
                    
                    zord = 10 if iv_m >= iv_mod else 1
                    ax.plot([opt.strike/S0]*2, [opt.maturity]*2, [iv_mod, iv_m], color='white', linewidth=0.8, alpha=0.65, zorder=zord)
                    ax.plot([opt.strike/S0], [opt.maturity], [iv_m], marker='o', color="#F0F0F0", markersize=4.62, alpha=0.85 if iv_m >= iv_mod else 1.0, zorder=zord + 1)
                except: continue

        ax.view_init(elev=28, azim=-115) 
        ax.set_xlim(LOWER_M, UPPER_M); ax.set_ylim(UPPER_T, LOWER_T); ax.set_zlim(0.0, 0.75)
        
        plt.savefig(f"{filename}_surface_FINAL.pdf", format='pdf', bbox_inches='tight', pad_inches=0.15, facecolor='black', dpi=800)

def main():
    for i in range(2):
        try:
            result = load_calibration_by_index(i)
            if not result: continue
            data, r, q, opts, name = result
            plot_surface_professional(data['market']['S0'], r, q, data.get('analytical', {}), "Asset", name, opts, data, 0, "Monte Carlo")
        except Exception as e: print(f"[Error at index {i}] {e}")

if __name__ == "__main__": main()