import json
import glob
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter 
from scipy.interpolate import interp1d

try:
    from heston_pricer.calibration import BatesCalibrator
    from heston_pricer.analytics import BatesAnalyticalPricer, implied_volatility
except ImportError:
    pass

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
            times, rates = [0.0, 30.0], [float(curve_data), float(curve_data)]

        sorted_pairs = sorted(zip(times, rates))
        self.ts = np.array([p[0] for p in sorted_pairs])
        self.rs = np.array([p[1] for p in sorted_pairs])
        
        self.interp = interp1d(self.ts, self.rs, kind='linear', 
                               bounds_error=False, fill_value=(self.rs[0], self.rs[-1]))

    def get_rate(self, T):
        return float(self.interp(max(T, 1e-4)))

class ReconstructedOption:
    def __init__(self, strike, maturity, price, option_type="CALL"):
        self.strike, self.maturity, self.market_price = float(strike), float(maturity), float(price)
        self.option_type = str(option_type)

def load_latest_calibration():
    patterns = ['results/calibration_*_meta.json', 'calibration_*_meta.json']
    files = []
    for p in patterns: files.extend(glob.glob(p))
    
    if not files: raise FileNotFoundError("No calibration meta file found.")
    latest_meta = sorted(files, key=os.path.getctime)[2]
    base_name = latest_meta.replace("_meta.json", "")
    
    with open(latest_meta, 'r') as f: data = json.load(f)
    
    r_curve = RobustYieldCurve(data['market'].get('r_sample', data['market'].get('r')))
    q_curve = RobustYieldCurve(data['market'].get('q_sample', data['market'].get('q')))

    market_options = []
    csv_file = f"{base_name}_prices.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            k = row.get('K', row.get('Strike', 0))
            t = row.get('T', row.get('Maturity', 0))
            p = row.get('Market', row.get('Price', 0))
            market_options.append(ReconstructedOption(k, t, p, row.get('Type', "CALL")))

    return data, r_curve, q_curve, market_options, base_name

def plot_surface_professional(S0, r_curve, q_curve, params, ticker, filename, market_options, data_full, dropped_count, source_name):
    kappa, theta, xi, rho, v0 = params['kappa'], params['theta'], params['xi'], params['rho'], params['v0']
    lamb, mu_j, sigma_j = params.get('lamb', 0.0), params.get('mu_j', 0.0), params.get('sigma_j', 0.0)
    is_bates = lamb > 0.0

    M_range = np.linspace(0.685, 1.315, 100)
    T_range = np.linspace(0.04, 1.5, 100)
    X, Y = np.meshgrid(M_range, T_range)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            T_val, M_val = Y[i, j], X[i, j]
            r_T, q_T = r_curve.get_rate(T_val), q_curve.get_rate(T_val)
            
            prices = BatesAnalyticalPricer.price_european_call_vectorized(
                S0, np.array([S0 * M_val]), np.array([T_val]), np.array([r_T]), np.array([q_T]),
                kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j
            )
            try:
                iv = implied_volatility(float(prices[0]), S0, S0 * M_val, T_val, r_T, q_T, "CALL")
                Z[i, j] = iv if 0.01 < iv < 2.5 else np.nan
            except: Z[i, j] = np.nan

    Z = pd.DataFrame(Z).interpolate(method='linear', axis=1).ffill(axis=1).bfill(axis=1).values
    Z_smooth = gaussian_filter(Z, sigma=0)

    with plt.style.context('dark_background'):
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z_smooth, cmap=cm.RdYlBu_r, rcount=100, ccount=100, 
                               edgecolor='black', linewidth=0.085, alpha=0.8, shade=False)

        if market_options:
            valid_needles = 0
            for opt in [o for o in market_options if (0.685 <= (o.strike/S0) <= 1.315) and (0.04 <= o.maturity <= 1.5)]:
                try:
                    rt, qt = r_curve.get_rate(opt.maturity), q_curve.get_rate(opt.maturity)
                    iv_mkt = implied_volatility(opt.market_price, S0, opt.strike, opt.maturity, rt, qt, opt.option_type)
                    p_mod = BatesAnalyticalPricer.price_vectorized(S0, np.array([opt.strike]), np.array([opt.maturity]), 
                                                                  np.array([rt]), np.array([qt]), np.array([opt.option_type]),
                                                                  kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j)[0]
                    iv_mod = implied_volatility(float(p_mod), S0, opt.strike, opt.maturity, rt, qt, opt.option_type)
                    if not (0.01 < iv_mkt < 2.5): continue
                    
                    valid_needles += 1
                    zord = 10 if iv_mkt >= iv_mod else 1
                    ax.plot([opt.strike/S0]*2, [opt.maturity]*2, [iv_mod, iv_mkt], color='white', linewidth=0.8, alpha=0.65, zorder=zord)
                    ax.plot([opt.strike/S0], [opt.maturity], [iv_mkt], marker='o', color="#F0F0F0", markersize=4.0, zorder=zord+1)
                except: continue

        ax.view_init(elev=28, azim=-115) 
        ax.set_xlabel('Moneyness', color='white')
        ax.set_ylabel('Maturity', color='white')
        ax.set_zlabel('Implied Volatility', color='white')

        plt.savefig(f"{filename}_surface.png", dpi=300, facecolor='black', bbox_inches='tight')

def main():
    try:
        data, r_curve, q_curve, market_options, base_name = load_latest_calibration()
        plot_surface_professional(data['market']['S0'], r_curve, q_curve, data['analytical'], "Asset", base_name, market_options, data, 0, "Analytical")
    except Exception as e:
        print(f"[Error] {e}")

if __name__ == "__main__":
    main()