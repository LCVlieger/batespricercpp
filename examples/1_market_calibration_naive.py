import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize
from heston_pricer.calibration import BatesCalibratorMC
from heston_pricer.analytics import BatesAnalyticalPricer, implied_volatility
from heston_pricer.data import (
    fetch_treasury_rates_fred, fetch_raw_data, fetch_options,
    get_market_implied_spot, ImpliedDividendCurve,
    save_options_to_cache, load_options_from_cache
)

def save_results(ticker, S0, r_curve, q_curve, res_params, options):
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"results/calibration_Analytic_{ticker}_{timestamp}"
    
    tenors = [(0.02, "1w"), (0.04, "2w"), (0.0833, "1M"), (0.25, "3M"), (0.5, "6M"), (1.0, "1Y")]
    r_sample = {f"{t:.4f}Y": float(r_curve.get_rate(t)) for t, _ in tenors}
    q_sample = {f"{t:.4f}Y": float(q_curve.get_rate(t)) for t, _ in tenors}
    
    print(f"\n[Validation] Re-pricing {len(options)} instruments...")
    
    keys = ['kappa', 'theta', 'xi', 'rho', 'v0', 'lamb', 'mu_j', 'sigma_j']
    p_values = [res_params[k] for k in keys] if isinstance(res_params, dict) else list(res_params.x)
    kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j = p_values
    
    strikes = np.array([o.strike for o in options])
    mats = np.array([o.maturity for o in options])
    types = np.array([o.option_type for o in options])
    r_T = np.array([r_curve.get_rate(o.maturity) for o in options])
    q_T = np.array([q_curve.get_rate(o.maturity) for o in options])
    
    model_prices = BatesAnalyticalPricer.price_vectorized(
        S0, strikes, mats, r_T, q_T, types, 
        kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j
    )
    
    rows = []
    for i, opt in enumerate(options):
        model_p = model_prices[i]
        bid = getattr(opt, 'bid', 0.0)
        ask = getattr(opt, 'ask', 0.0)
        
        spread = opt.spread if hasattr(opt, 'spread') and opt.spread > 0 else (ask - bid if ask > 0 and bid > 0 else 0.0)
        
        try:
            r_val = r_curve.get_rate(opt.maturity)
            q_val = q_curve.get_rate(opt.maturity)
            iv_mkt = implied_volatility(opt.market_price, S0, opt.strike, opt.maturity, r_val, q_val, opt.option_type)
            iv_model = implied_volatility(model_p, S0, opt.strike, opt.maturity, r_val, q_val, opt.option_type)
        except:
            iv_mkt, iv_model = 0.0, 0.0
            
        rows.append({
            "T": round(opt.maturity, 4), "K": opt.strike, "Type": opt.option_type,
            "Bid": round(bid, 4), "Ask": round(ask, 4), "Spread": round(spread, 5),
            "Weight": 1/max(abs(ask - bid), 0.01), "Market": round(opt.market_price, 4),
            "Model": round(model_p, 4), "Err": round(model_p - opt.market_price, 4),
            "IV_Mkt": round(iv_mkt, 4), "IV_Model": round(iv_model, 4)
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{base_name}_prices.csv", index=False)
    
    rmse = float(np.sqrt((df["Err"]**2).mean()))
    res_dict = dict(zip(keys, p_values))
    res_dict['rmse'] = rmse
    
    meta = {
        "model": "Bates_MC_Val", 
        "market": {"S0": S0, "r_sample": r_sample, "q_sample": q_sample}, 
        "params": res_dict
    }
    
    with open(f"{base_name}_meta.json", "w") as f:
        json.dump(meta, f, indent=4)
        
    print("="*80 + f"\nVALIDATION | S0: {S0:.2f} | RMSE: {rmse:.4f}\n" + "-"*80)
    print(df[['T', 'K', 'Type', 'Market', 'Model', 'Spread', 'IV_Mkt']].head(5).to_string(index=False))
    print("-" * 80 + f"\nSaved to: {base_name}_prices.csv")

def print_curves(r_curve, q_curve):
    print("\n" + "="*60 + f"\n{'Tenor':<10} | {'Rate (r)':<15} | {'Yield (q)':<15}\n" + "-"*60)
    tenors = [(0.02, "1w"), (0.04, "2w"), (0.0833, "1M"), (0.25, "3M"), (0.5, "6M"), (1.0, "1Y")]
    for t, label in tenors:
        print(f"{label:<10} | {r_curve.get_rate(t)*100:>13.4f}% | {q_curve.get_rate(t)*100:>13.4f}%")
    print("="*60 + "\n")

def main():
    ticker = "^SPX"
    target_date = datetime.now().strftime("%Y-%m-%d")
    
    r_curve = fetch_treasury_rates_fred(target_date, os.getenv("FRED_API_KEY"))
    raw_df = fetch_raw_data(ticker)
    S0 = get_market_implied_spot(ticker, raw_df, r_curve)
    
    opts = fetch_options(ticker, S0, target_size=300)
    save_options_to_cache(opts, ticker)
    
    q_curve = ImpliedDividendCurve(raw_df, S0, r_curve, ticker)
    print_curves(r_curve, q_curve)
    
    print(f"Processing {len(opts)} options...\n" + "="*20 + " BATES MC CALIBRATION " + "="*20)
    mc = BatesCalibratorMC(S0=S0, r_curve=r_curve, q_curve=q_curve, n_paths=5000, n_steps=1000)
    mc._precompute(opts)
    
    short_opts = [o for o in opts if o.maturity < 0.1]
    closest = min(short_opts, key=lambda x: abs(x.strike - S0))
    iv = implied_volatility(
        closest.market_price, S0, closest.strike, closest.maturity, 
        r_curve.get_rate(closest.maturity), q_curve.get_rate(closest.maturity), closest.option_type
    )
    print(f"LOCK v0: {iv**2:.4f} (Vol: {iv:.1%})")
    
    def obj(p):
        try:
            model_p, market_p, weights = mc.get_prices(p)
            return np.sqrt(np.mean(((model_p - market_p) * weights)**2))
        except:
            return 1e10
            
    def cb(xk):
        print(f" [MC-Iter] Obj: {obj(xk):10.4f} | v0:{xk[4]:.4f} th:{xk[1]:.4f} ka:{xk[0]:.3f} xi:{xk[2]:.3f} rho:{xk[3]:.2f}")
        
    t0 = time.time()
    x0 = [1.5, 0.04, 0.6, -0.4, 0.04, 0.5, -0.05, 0.2]
    bounds = [
        (1.0, 5.0), (0.001, 0.5), (0.01, 0.9), (-0.99, 0.0), 
        (0.001, 0.1), (0.0, 0.5), (-0.3, 0.0), (0.05, 0.3)
    ]
    
    res = minimize(obj, x0, method='SLSQP', bounds=bounds, callback=cb, tol=1e-8, options={'maxiter': 500, 'eps': 3e-3})
    print(f"DONE ({time.time()-t0:.2f}s)")
    
    final_params = dict(zip(['kappa', 'theta', 'xi', 'rho', 'v0', 'lamb', 'mu_j', 'sigma_j'], res.x))
    save_results(ticker, S0, r_curve, q_curve, final_params, opts)

if __name__ == "__main__": 
    main()