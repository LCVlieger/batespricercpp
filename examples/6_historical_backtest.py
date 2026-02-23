import os
import requests
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple
from scipy.optimize import minimize
from nelson_siegel_svensson.calibrate import calibrate_nss_ols

@dataclass
class MarketOption:
    strike: float
    maturity: float
    market_price: float
    option_type: str = "CALL"

class NSSYieldCurve:
    def __init__(self, curve_fit):
        self.curve = curve_fit

    def get_rate(self, T: float) -> float:
        return float(self.curve(T))

class ImpliedDividendCurve:
    def __init__(self, df: pd.DataFrame, S0: float, r_curve: NSSYieldCurve):
        self.yields = {}
        
        for T in sorted(df['T'].unique()):
            subset = df[df['T'] == T]
            r = r_curve.get_rate(T)
            F_approx = S0 * np.exp(r * T)
            
            valid_rows = subset.dropna(subset=['C_MID', 'P_MID'])
            if valid_rows.empty: 
                continue
            
            best_idx = (valid_rows['STRIKE'] - F_approx).abs().idxmin()
            row = valid_rows.loc[best_idx]
            
            K, C, P = row['STRIKE'], row['C_MID'], row['P_MID']
            rhs = C - P + K * np.exp(-r * T)
            
            if rhs > 0:
                self.yields[T] = -np.log(rhs / S0) / T
            else:
                self.yields[T] = 0.015

    def get_rate(self, T: float) -> float:
        mats = sorted(self.yields.keys())
        if not mats: 
            return 0.0
        return np.interp(T, mats, [self.yields[m] for m in mats])

def fetch_treasury_rates_fred(date_str: str, api_key: str) -> NSSYieldCurve:
    print(f"Fetching Treasury Yields from FRED for {date_str}...")
    series_map = {
        1/12: "DGS1MO", 3/12: "DGS3MO", 6/12: "DGS6MO", 
        1.0: "DGS1", 2.0: "DGS2", 5.0: "DGS5", 
        10.0: "DGS10", 20.0: "DGS20", 30.0: "DGS30"
    }
    maturities, yields = [], []
    for tenor, series_id in series_map.items():
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json&observation_start={date_str}&observation_end={date_str}"
        try:
            obs = requests.get(url, timeout=10).json()['observations']
            if obs and obs[0]['value'] != '.':
                maturities.append(tenor)
                yields.append(float(obs[0]['value']) / 100.0)
        except Exception: 
            continue
            
    curve_fit, _ = calibrate_nss_ols(np.array(maturities), np.array(yields))
    return NSSYieldCurve(curve_fit)

class HestonAnalyticalPricer:
    @staticmethod
    def price_european_call_vectorized(S0, K, T, r, q, kappa, theta, xi, rho, v0):
        N_grid, u_max = 400, 100.0
        du = u_max / N_grid
        u = np.linspace(1e-8, u_max, N_grid)[:, np.newaxis] 

        T_mat, r_mat, q_mat, K_mat = T[np.newaxis, :], r[np.newaxis, :], q[np.newaxis, :], K[np.newaxis, :]

        def get_cf(phi):
            d = np.sqrt((rho * xi * phi * 1j - kappa)**2 + xi**2 * (phi * 1j + phi**2))
            g = (kappa - rho * xi * phi * 1j - d) / (kappa - rho * xi * phi * 1j + d)
            e_neg_dT = np.exp(-d * T_mat)
            
            C = (1/xi**2) * ((1 - e_neg_dT) / (1 - g * e_neg_dT)) * (kappa - rho * xi * phi * 1j - d)
            D = (kappa * theta / xi**2) * ((kappa - rho * xi * phi * 1j - d) * T_mat - 2 * (np.log(1 - g * e_neg_dT) - np.log(1 - g)))
            drift = 1j * phi * np.log(S0 * np.exp((r_mat - q_mat) * T_mat))
            return np.exp(C * v0 + D + drift)

        cf1, cf2 = get_cf(u - 1j), get_cf(u)
        int1 = np.real((np.exp(-1j * u * np.log(K_mat)) * cf1) / (1j * u * S0 * np.exp((r_mat - q_mat) * T_mat)))
        int2 = np.real((np.exp(-1j * u * np.log(K_mat)) * cf2) / (1j * u))

        P1, P2 = 0.5 + (1/np.pi) * np.sum(int1 * du, axis=0), 0.5 + (1/np.pi) * np.sum(int2 * du, axis=0)
        price = S0 * np.exp(-q_mat * T_mat) * P1 - K_mat * np.exp(-r_mat * T_mat) * P2
        return np.maximum(price.flatten(), 0.0)

class HestonCalibrator:
    def __init__(self, S0: float, r_curve: NSSYieldCurve, q_curve: ImpliedDividendCurve):
        self.S0, self.r_curve, self.q_curve = S0, r_curve, q_curve

    def calibrate(self, options: List[MarketOption]) -> Dict:
        strikes = np.array([o.strike for o in options])
        maturities = np.array([o.maturity for o in options])
        mkt_prices = np.array([o.market_price for o in options])
        
        r_vec = np.array([self.r_curve.get_rate(t) for t in maturities])
        q_vec = np.array([self.q_curve.get_rate(t) for t in maturities])
        
        bounds = [(1e-3, 0.1), (1e-3, 5.0), (1e-3, 0.1), (1e-2, 1.5), (-1.0, 0.0)]
        x0 = [0.04, 2.5, 0.04, 0.5, -0.7]

        def objective(p):
            v0, k, th, xi, rho = p
            try:
                mod_p = HestonAnalyticalPricer.price_european_call_vectorized(self.S0, strikes, maturities, r_vec, q_vec, k, th, xi, rho, v0)
                return np.mean((mod_p - mkt_prices)**2)
            except: 
                return 1e9

        def cb(xk):
            print(f"==> RMSE: {np.sqrt(objective(xk)):.4f} | v0={xk[0]:.4f} k={xk[1]:.2f} th={xk[2]:.4f} xi={xk[3]:.4f} rho={xk[4]:.2f}")

        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, callback=cb, tol=1e-9, options={'eps': 1e-3, 'maxiter': 500})
        return {**dict(zip(["v0", "kappa", "theta", "xi", "rho"], res.x)), "rmse": np.sqrt(res.fun)}

def load_spx_replication(file_path, target_date) -> Tuple[List[MarketOption], pd.DataFrame, float]:
    df = pd.read_csv(file_path, low_memory=False, skipinitialspace=True)
    df.columns = df.columns.str.strip(' []')
    
    for c in ['STRIKE','C_BID','C_ASK','P_BID','P_ASK','UNDERLYING_LAST']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df = df.dropna(subset=['STRIKE','UNDERLYING_LAST'])
    df['QUOTE_DATE'], df['EXPIRE_DATE'] = pd.to_datetime(df['QUOTE_DATE']), pd.to_datetime(df['EXPIRE_DATE'])
    
    day_data = df[df['QUOTE_DATE'] == pd.to_datetime(target_date)].copy()
    S0 = day_data['UNDERLYING_LAST'].iloc[0]
    
    day_data['C_MID'] = (day_data['C_BID'] + day_data['C_ASK']) / 2
    if 'P_BID' in day_data.columns:
        day_data['P_MID'] = (day_data['P_BID'] + day_data['P_ASK']) / 2
    
    day_data['T'] = (day_data['EXPIRE_DATE'] - day_data['QUOTE_DATE']).dt.days / 365.25
    day_data = day_data[(day_data['T'] > 0.04) & (day_data['T'] < 2.5)]
    
    groups = day_data.groupby('T')['STRIKE'].apply(set)
    common = {k for k in set.intersection(*groups.tolist()) if 3200 < k < 4800} 
    calib = day_data[day_data['STRIKE'].isin(common)]
    
    return [MarketOption(r['STRIKE'], r['T'], r['C_MID']) for _, r in calib.iterrows()], day_data, S0

def print_full_curves(r_curve, q_curve):
    print("\n" + "="*60 + f"\n{'Tenor':<10} | {'Risk-Free (r)':<15} | {'Div Yield (q)':<15}\n" + "-"*60)
    for t, l in zip([1/12, 3/12, 6/12, 1.0, 2.0], ["1 Month", "3 Month", "6 Month", "1 Year", "2 Year"]):
        print(f"{l:<10} | {r_curve.get_rate(t)*100:>13.4f}% | {q_curve.get_rate(t)*100:>13.4f}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    TARGET_DATE = "2022-03-25"
    options, raw_df, S0 = load_spx_replication("src/spx_eod_202203.txt", TARGET_DATE)
    
    r_curve = fetch_treasury_rates_fred(TARGET_DATE, os.getenv("FRED_API_KEY"))
    q_curve = ImpliedDividendCurve(raw_df, S0, r_curve)
    
    print_full_curves(r_curve, q_curve)
    res = HestonCalibrator(S0, r_curve, q_curve).calibrate(options)
    
    for k, v in res.items(): 
        print(f"  {k}: {v:.6f}")