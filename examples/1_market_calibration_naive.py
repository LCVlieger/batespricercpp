import os
import json
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from scipy.optimize import minimize
from heston_pricer.calibration import BatesCalibratorMC
from heston_pricer.analytics import BatesAnalyticalPricer, implied_volatility
from heston_pricer.data import (
    fetch_treasury_rates_fred, 
    fetch_raw_data, 
    fetch_options, 
    get_market_implied_spot, 
    ImpliedDividendCurve,
    save_options_to_cache,
    load_options_from_cache
)

# =================================================================
# 3. SAVING & VALIDATION 
# =================================================================
def save_results(ticker, S0, r_curve, q_curve, res_params, options):
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"results/calibration_Analytic_{ticker}_{timestamp}"
    
    tenors = [(0.02, "1 week"), (0.04, "2 weeks"), (0.0833, "1 Month"), (0.25, "3 Months"), (0.5, "6 Months"), (1.0, "1 Year")]
    r_sample = {f"{t:.4f}Y": float(r_curve.get_rate(t)) for t, label in tenors}
    q_sample = {f"{t:.4f}Y": float(q_curve.get_rate(t)) for t, label in tenors}
    
    print(f"\n[Validation] Re-pricing {len(options)} instruments using Final Parameters...")

    # 1. Extract Parameter Values (Ensure correct order matching keys)
    keys = ['kappa', 'theta', 'xi', 'rho', 'v0', 'lamb', 'mu_j', 'sigma_j']
    
    if isinstance(res_params, dict):
        p_values = [res_params[k] for k in keys]
    else:
        # If res_params is a scipy OptimizeResult, assume .x follows the same order
        p_values = list(res_params.x)
    
    print(f"Params: {p_values}")

    # Correct Unpacking (Clean & Readable)
    kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j = p_values

    # 2. Vectorized Pricing
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
        err = model_p - opt.market_price
        
        # --- NEW: Capture Spread Data ---
        # Using getattr defaults to 0.0 if the attribute is missing
        bid = getattr(opt, 'bid', 0.0)
        ask = getattr(opt, 'ask', 0.0)
        
        # Calculate spread if available, otherwise 0
        spread = 0.0
        if hasattr(opt, 'spread') and opt.spread > 0:
            spread = opt.spread
        elif ask > 0 and bid > 0:
            spread = ask - bid
        
        # IV Calculation (Safe)
        try:
            r_val = r_curve.get_rate(opt.maturity)
            q_val = q_curve.get_rate(opt.maturity)
            # Assuming you have an implied_volatility function imported
            iv_mkt = implied_volatility(opt.market_price, S0, opt.strike, opt.maturity, r_val, q_val, opt.option_type)
            iv_model = implied_volatility(model_p, S0, opt.strike, opt.maturity, r_val, q_val, opt.option_type)
        except:
            iv_mkt, iv_model = 0.0, 0.0
        
        rows.append({
            "T": round(opt.maturity, 4), 
            "K": opt.strike, 
            "Type": opt.option_type,
            "Bid": round(bid, 4),
            "Ask": round(ask, 4),
            "Spread": round(spread, 5),
            "Weight": 1/max(abs(opt.ask - opt.bid), 0.01),
            "Market": round(opt.market_price, 4), 
            "Model": round(model_p, 4), 
            "Err": round(err, 4), 
            "IV_Mkt": round(iv_mkt, 4), 
            "IV_Model": round(iv_model, 4)
            # --- SAVING SPREADS ---
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{base_name}_prices.csv", index=False)
    
    # --- METRICS ---
    rmse_val = float(np.sqrt((df["Err"]**2).mean()))
    
    # --- JSON METADATA ---
    res_dict = dict(zip(keys, p_values))
    res_dict['rmse'] = rmse_val

    meta = {
        "model": "Bates_MC_Validation",
        "market": {"S0": S0, "r_sample": r_sample, "q_sample": q_sample},
        "analytical": res_dict,
        "notes": "Includes Bid/Ask/Spread for MC re-simulation"
    }
    
    with open(f"{base_name}_meta.json", "w") as f:
        json.dump(meta, f, indent=4)
    
    print("\n" + "="*80)
    print(f"VALIDATION SAMPLE (First 5) | S0: {S0:.2f}")
    print("-" * 80)
    print(df[['T', 'K', 'Type', 'Market', 'Model', 'Spread', 'IV_Mkt']].head(5).to_string(index=False))
    print("-" * 80)
    print(f"Final RMSE: {rmse_val:.4f}")
    print(f"Saved to: {base_name}_prices.csv")
    
def print_curves(r_curve, q_curve):
    print("\n" + "="*60)
    print(f"{'Tenor':<10} | {'Market Rate (r)':<15} | {'Div Yield (q)':<15}")
    print("-" * 60)
    tenors = [(0.02, "1 week"), (0.04, "2 weeks"), (0.0833, "1 Month"), (0.25, "3 Months"), (0.5, "6 Months"), (1.0, "1 Year")]

    for t, label in tenors:
        print(f"{label:<10} | {r_curve.get_rate(t)*100:>13.4f}% | {q_curve.get_rate(t)*100:>13.4f}%")
    print("="*60 + "\n")

# =================================================================
# 4. MAIN EXECUTION
# =================================================================
def main():
    FRED_API_KEY = os.getenv("FRED_API_KEY")
    target_date = datetime.now().strftime("%Y-%m-%d")
    ticker = "^SPX" #all work: "AAPL" #"^SPX"
    
    print(f"Initializing Bates Calibration for {ticker}...")
    
    # 1. Data Fetching
    # r_curve now automatically adds the BOX_SPREAD in data.py
    cache_file = None 

    # 1. Data Fetching
    r_curve = fetch_treasury_rates_fred(target_date, FRED_API_KEY)
    
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        options_processed = load_options_from_cache(cache_file)
        # Note: You still need a S0. You can either save S0 in the JSON 
        # or fetch the historical close for that specific date.
        S0_actual = 190.45 # Example: replace with actual saved valuex
    else:
        raw_df = fetch_raw_data(ticker)
        S0_actual = get_market_implied_spot(ticker, raw_df, r_curve)
        options_processed = fetch_options(ticker, S0_actual, target_size=300)
        
        # SAVE IT FOR TOMORROW
        save_options_to_cache(options_processed, ticker)
        
    raw_df = fetch_raw_data(ticker)
    
    # S0_actual uses the higher r_curve to synchronize correctly
    S0_actual = get_market_implied_spot(ticker, raw_df, r_curve)
    print(f"Market-Consistent Spot: {S0_actual:.2f}")
    
    q_curve = ImpliedDividendCurve(raw_df, S0_actual, r_curve, ticker)
    print_curves(r_curve, q_curve)
    
    # options_processed no longer contains the .forward field
    options_processed = fetch_options(ticker, S0_actual, target_size=300)
    print(f"Processing {len(options_processed)} standard OTM options...")
    
    # 3. Calibration
    print(f"\n{'='*20} BATES ANALYTICAL CALIBRATION {'='*20}")
    t0 = time.time()
    
    mc_calib = BatesCalibratorMC(S0=S0_actual, r_curve=r_curve, q_curve=q_curve, n_paths=5000, n_steps=1000)  #10k10k / 5k5k
    mc_calib._precompute(options_processed)
    
    # 3. Optimization Setup     
    print(f"\n{'='*20} STARTING MC OPTIMIZATION {'='*20}")

    short_opts = [o for o in options_processed if o.maturity < 0.1]
    closest = min(short_opts, key=lambda x: abs(x.strike - S0_actual))

    implied_v0_root = implied_volatility(closest.market_price, S0_actual, closest.strike, closest.maturity, r_curve.get_rate(closest.maturity), q_curve.get_rate(closest.maturity), closest.option_type)
    fixed_v0 = implied_v0_root ** 2
    v0_min, v0_max = fixed_v0 * 0.95, fixed_v0 * 1.05
    
    print(f"LOCKING v0 to Market Reality: {fixed_v0:.4f} (Vol: {implied_v0_root:.1%})")

    bounds = [
            (1.0, 5.0),   # kappa (Speed of mean reversion)
            (0.001, 0.5),  # theta (Long run variance)
            (0.01, 0.9),   # xi (Vol of Vol - allow high values for steep smile)
            (-0.99, 0.0), # rho (Correlation - Locked negative for Equity Skew)
            (0.001, 0.1),  # v0 (Initial variance)
            (0.0, 0.5),    # lamb (Jump intensity)
            (-0.3, 0.0),   # mu_j (Mean jump size)
            (0.05, 0.3)    # sigma_j (Jump volatility)
        ]
    x0 = [1.5, 0.04, 0.6, -0.4, 0.04, 0.5, -0.05, 0.2] #[1.5, 0.25, 0.6, -0.2, 0.21, 0.5, -0.05, 0.2]#[2.0, fixed_v0, 1.0, -0.7, fixed_v0, 0.1, -0.1, 0.1]
    
    def objective(p):
        try:
            model_p, market_p, weights = mc_calib.get_prices(p)
            diff = (model_p - market_p)
            w_rmse = np.sqrt(np.mean((diff * weights)**2))
            return w_rmse
        except Exception:
            return 1e10

    def callback(xk):
        obj_val = objective(xk)
        print(f" [MC-Iter] Obj: {obj_val:10.4f} | "
              f"v0:{xk[4]:.4f} th:{xk[1]:.4f} ka:{xk[0]:.3f} xi:{xk[2]:.3f} rho:{xk[3]:.2f} | lam:{xk[5]:1.3f}  mu_j:{xk[6]:1.3f}  sig_j:{xk[7]:1.3f}")
        
    t0 = time.time()
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, callback=callback, tol=1e-8, options={'maxiter': 500, 'eps': 3e-3}) #was -2 5e-3 beste met 5k 5k
    t1 = time.time() 
    print(f"Calibration time: ")
    print(t1 - t0)
    print(f"CALIBRATION DONE (Time: {time.time()-t0:.2f}s)")
    
    final_params = dict(zip(['kappa', 'theta', 'xi', 'rho', 'v0', 'lamb', 'mu_j', 'sigma_j'], res.x))
    
    save_results(ticker, S0_actual, r_curve, q_curve, final_params, options_processed)

if __name__ == "__main__":
    main()