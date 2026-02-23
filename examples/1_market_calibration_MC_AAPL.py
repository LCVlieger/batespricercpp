import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from heston_pricer.calibration import BatesCalibratorMC 
from heston_pricer.analytics import implied_volatility, BatesAnalyticalPricer

class InterpolatedCurve:
    def __init__(self, curve_dict):
        times = sorted([float(k.replace('Y', '')) for k in curve_dict.keys()])
        rates = [curve_dict[f"{t:.4f}Y"] for t in times]
        
        self.interp = interp1d(
            times, rates, kind='linear', 
            fill_value=(rates[0], rates[-1]), bounds_error=False
        )

    def get_rate(self, T):
        return float(self.interp(T))

class SimpleOption:
    def __init__(self, T, K, type_str, price, spread):
        self.maturity = T
        self.strike = K
        self.option_type = type_str
        self.market_price = price
        self.ask = price + (spread / 2.0)
        self.bid = price - (spread / 2.0)

def save_results(ticker, S0, r_curve, q_curve, res_params, options):
    os.makedirs("results", exist_ok=True)
    base_name = f"results/calibration_MC_{ticker}_20260208_020354"
    
    tenors = [(0.02, "1w"), (0.04, "2w"), (0.0833, "1M"), (0.25, "3M"), (0.5, "6M"), (1.0, "1Y")]
    r_sample = {f"{t:.4f}Y": float(r_curve.get_rate(t)) for t, _ in tenors}
    q_sample = {f"{t:.4f}Y": float(q_curve.get_rate(t)) for t, _ in tenors}
    
    keys = ['kappa', 'theta', 'xi', 'rho', 'v0', 'lamb', 'mu_j', 'sigma_j']
    p_values = [res_params[k] for k in keys] if isinstance(res_params, dict) else list(res_params.x)
    ka, th, xi, rho, v0, la, mj, sj = p_values
    
    strikes = np.array([o.strike for o in options])
    mats = np.array([o.maturity for o in options])
    types = np.array([o.option_type for o in options])
    r_T = np.array([r_curve.get_rate(o.maturity) for o in options])
    q_T = np.array([q_curve.get_rate(o.maturity) for o in options])
    
    model_prices = BatesAnalyticalPricer.price_vectorized(
        S0, strikes, mats, r_T, q_T, types, 
        ka, th, xi, rho, v0, la, mj, sj
    )
    
    rows = []
    for i, opt in enumerate(options):
        rv, qv, mp = r_T[i], q_T[i], model_prices[i]
        err = mp - opt.market_price
        
        try:
            iv_mkt = implied_volatility(opt.market_price, S0, opt.strike, opt.maturity, rv, qv, opt.option_type)
            iv_mod = implied_volatility(mp, S0, opt.strike, opt.maturity, rv, qv, opt.option_type)
        except:
            iv_mkt, iv_mod = 0.0, 0.0
            
        rows.append({
            "T": round(opt.maturity, 3), 
            "K": opt.strike, 
            "Spread": opt.ask - opt.bid, 
            "Weight": 1 / max(abs(opt.ask - opt.bid), 0.01), 
            "Type": opt.option_type, 
            "Market": round(opt.market_price, 2), 
            "Model": round(mp, 2), 
            "Err": round(err, 2), 
            "IV_Mkt": round(iv_mkt, 4), 
            "IV_Mod": round(iv_mod, 4)
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{base_name}_prices.csv", index=False)
    rmse = np.sqrt((df["Err"]**2).mean())
    res_dict = {**dict(zip(keys, p_values)), 'rmse': rmse}
    
    output_meta = {
        "model": "Bates_MC_Val", 
        "market": {"S0": S0, "r": r_sample, "q": q_sample}, 
        "params": res_dict
    }
    
    with open(f"{base_name}_meta.json", "w") as f:
        json.dump(output_meta, f, indent=4)
        
    print("="*80 + f"\nVALIDATION | S0: {S0:.2f}\n" + "-"*80)
    print(df.head(10).to_string(index=False))
    print("-" * 80 + f"\nRMSE: {rmse:.4f}\nSaved: {base_name}_prices.csv")

def print_curves(r_curve, q_curve):
    print("\n" + "="*60 + f"\n{'Tenor':<10} | {'Rate (r)':<15} | {'Yield (q)':<15}\n" + "-"*60)
    tenors = [(0.02, "1w"), (0.04, "2w"), (0.0833, "1M"), (0.25, "3M"), (0.5, "6M"), (1.0, "1Y")]
    for t, l in tenors:
        print(f"{l:<10} | {r_curve.get_rate(t)*100:>13.4f}% | {q_curve.get_rate(t)*100:>13.4f}%")
    print("="*60 + "\n")

def main():
    json_p = "results/calibration_Analytic_AAPL_20260208_020354_meta.json"
    csv_p = "results/calibration_Analytic_AAPL_20260208_020354_prices.csv"
    ticker = "AAPL"
    
    with open(json_p, 'r') as f:
        meta = json.load(f)
        
    S0 = meta['market']['S0']
    r_curve = InterpolatedCurve(meta['market']['r_sample'])
    q_curve = InterpolatedCurve(meta['market']['q_sample'])
    print_curves(r_curve, q_curve)
    
    df_prices = pd.read_csv(csv_p)
    options = []
    for _, row in df_prices.iterrows():
        options.append(SimpleOption(row['T'], row['K'], row['Type'], row['Market'], row['Spread']))
    
    mc = BatesCalibratorMC(S0=S0, r_curve=r_curve, q_curve=q_curve, n_paths=5000, n_steps=5000)
    mc._precompute(options)
    
    short_opts = [o for o in options if o.maturity < 0.1]
    closest = min(short_opts, key=lambda x: abs(x.strike - S0))
    iv = implied_volatility(
        closest.market_price, S0, closest.strike, closest.maturity, 
        r_curve.get_rate(closest.maturity), q_curve.get_rate(closest.maturity), closest.option_type
    )
    print(f"LOCK v0: {iv**2:.4f}")
    
    def obj(p):
        try:
            mp, mkp, w = mc.get_prices(p)
            return np.sqrt(np.mean(((mp - mkp) * w)**2))
        except:
            return 1e10
    
    initial_guess = [1.5, 0.25, 0.6, -0.2, 0.21, 0.5, -0.05, 0.2]
    bounds = [
        (1.0, 5.0), (0.001, 0.5), (0.01, 1), (-0.99, 0.0), 
        (0.001, 0.5), (0.0, 0.5), (-0.3, 0.0), (0.05, 0.3)
    ]
    
    res = minimize(
        obj, initial_guess, method='SLSQP', bounds=bounds, 
        tol=1e-8, options={'maxiter': 500, 'eps': 7e-3}
    )
    
    save_results(ticker, S0, r_curve, q_curve, res, options)

if __name__ == "__main__": 
    main()