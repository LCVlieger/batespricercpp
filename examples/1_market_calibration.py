import time
import json
import os
import shutil
import copy
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import List

# Local package imports
try:
    from heston_pricer.calibration import HestonCalibrator, HestonCalibratorMC, implied_volatility, SimpleYieldCurve
    from heston_pricer.analytics import HestonAnalyticalPricer
    from heston_pricer.data import fetch_options
    from heston_pricer.instruments import EuropeanOption, OptionType
except ImportError:
    raise ImportError("heston_pricer package not found. Ensure PYTHONPATH is set correctly.")

# --- CONFIGURATION ---
RUN_MC = False  # <--- SET TO FALSE TO DISABLE SLOW MC CALIBRATION

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# -----------------------------
# RISK-FREE RATE CURVE
# -----------------------------
def fetch_dynamic_risk_free_curve():
    log("Fetching Dynamic US Treasury Curve + 40bps Spread...")
    try:
        tickers = ["^IRX", "^FVX", "^TNX", "^TYX"]
        df = yf.download(tickers, period="5d", progress=False)['Close']
        latest = df.iloc[-1]
        if latest.isnull().any():
            latest = df.dropna().iloc[-1]

        rates_map = {
            0.25: (float(latest['^IRX']) / 100.0) + 0.0040,
            5.0:  (float(latest['^FVX']) / 100.0) + 0.0040,
            10.0: (float(latest['^TNX']) / 100.0) + 0.0040,
            30.0: (float(latest['^TYX']) / 100.0) + 0.0040
        }
        
        target_tenors = [0.08, 0.25, 0.5, 1.0, 2.0, 3.0]
        final_rates = []
        for t in target_tenors:
            if t <= 0.25:
                r = rates_map[0.25]
            elif t <= 5.0:
                ratio = (t - 0.25) / (5.0 - 0.25)
                r = rates_map[0.25] + ratio * (rates_map[5.0] - rates_map[0.25])
            else:
                r = rates_map[5.0]
            final_rates.append(r)
        return SimpleYieldCurve(target_tenors, final_rates)
    except Exception as e:
        log(f"Dynamic rate fetch failed ({e}). Using fallback 4.2% flat.")
        return SimpleYieldCurve([0.1, 3.0], [0.042, 0.042])

# -----------------------------
# IMPLIED DIVIDEND CURVE
# -----------------------------
def extract_implied_dividends(ticker_symbol, S0, r_curve):
    log(f"Extracting implied dividends for {ticker_symbol}...")
    is_index = ticker_symbol.startswith("^")
    
    if not is_index:
        return _calculate_historical_yield(ticker_symbol, S0)
    
    ticker = yf.Ticker(ticker_symbol)
    expirations = ticker.options
    today = datetime.now()
    candidates = []

    for exp_str in expirations:
        try:
            d = datetime.strptime(exp_str, "%Y-%m-%d")
            T = (d - today).days / 365.25
            if not (0.05 <= T <= 3.0):
                continue

            r = r_curve.get_rate(T)
            chain = ticker.option_chain(exp_str)
            calls = {row['strike']: (row['bid'] + row['ask'])/2 for _, row in chain.calls.iterrows() if row['bid'] > 0}
            puts = {row['strike']: (row['bid'] + row['ask'])/2 for _, row in chain.puts.iterrows() if row['bid'] > 0}
            common_strikes = set(calls.keys()).intersection(set(puts.keys()))
            
            implied_qs = []
            for K in common_strikes:
                if 0.95 <= K/S0 <= 1.05:
                    C, P = calls[K], puts[K]
                    lhs = (C - P + K * np.exp(-r*T)) / S0
                    if lhs > 0:
                        q_val = -np.log(lhs)/T
                        if -0.02 < q_val < 0.10:
                            implied_qs.append(q_val)
            if implied_qs:
                candidates.append({'T': T, 'q_raw': np.mean(implied_qs)})
        except Exception:
            continue

    if not candidates:
        return _calculate_historical_yield(ticker_symbol, S0)

    candidates.sort(key=lambda x: x['T'])
    tenors = [c['T'] for c in candidates]
    rates = [c['q_raw'] for c in candidates]
    log(f"Generated term-structured dividend curve with {len(tenors)} tenors.")
    return SimpleYieldCurve(tenors, rates)

def _calculate_historical_yield(ticker_symbol, S0):
    try:
        log(f"Calculating trailing 12-month yield for {ticker_symbol}...")
        ticker = yf.Ticker(ticker_symbol)
        divs = ticker.dividends
        if divs.empty:
            return SimpleYieldCurve([0.1, 5.0], [0.0, 0.0])
        one_year_ago = pd.Timestamp.now(tz=divs.index.tz) - pd.Timedelta(days=365)
        recent_divs = divs[divs.index >= one_year_ago]
        total_payout = recent_divs.sum()
        q = total_payout / S0
        log(f"-> Calculated yield: {q:.4%}")
    except Exception:
        q = 0.0
    return SimpleYieldCurve([0.1, 5.0], [q, q])

# -----------------------------
# SAVE RESULTS
# -----------------------------
def save_results(ticker, S0, r_curve, q_curve, res_ana, res_mc, options, init_guess):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"results/calibration_{ticker}_{timestamp}"
    
    with open(f"{base_name}_meta.json", "w") as f: 
        json.dump({
            "market": { "S0": S0, "r": r_curve.to_dict(), "q": q_curve.to_dict() }, 
            "initial_guess": { "kappa": init_guess[0], "theta": init_guess[1], "xi": init_guess[2], "rho": init_guess[3], "v0": init_guess[4] },
            "analytical": res_ana, 
            "monte_carlo_results": res_mc 
        }, f, indent=4)

    get_params = lambda res: [res.get(k, 0) for k in ['kappa', 'theta', 'xi', 'rho', 'v0']]
    rows = []

    print(f"\n[Validation] Re-pricing {len(options)} instruments (Unscaled)...")

    for opt in options:
        is_put = (opt.option_type == "PUT")
        r_T = r_curve.get_rate(opt.maturity)
        q_T = q_curve.get_rate(opt.maturity)
        
        # We still use the robust single-pricer here for final validation output
        def price_with_params(params):
            if is_put:
                return HestonAnalyticalPricer.price_european_put(S0, opt.strike, opt.maturity, r_T, q_T, *params)
            else:
                return HestonAnalyticalPricer.price_european_call(S0, opt.strike, opt.maturity, r_T, q_T, *params)

        p_ana = price_with_params(get_params(res_ana))
        # If MC didn't run, p_mc will just reuse Analytical params
        p_mc = price_with_params(get_params(res_mc))
        iv_mkt = implied_volatility(opt.market_price, S0, opt.strike, opt.maturity, r_T, q_T, opt.option_type)

        rows.append({
            "Type": opt.option_type, "T": round(opt.maturity, 3), "K": opt.strike, "Mkt": opt.market_price, 
            "Ana": round(p_ana, 2), "Err_A": round(p_ana - opt.market_price, 2),
            "MC": round(p_mc, 2), "Err_MC": round(p_mc - opt.market_price, 2),
            "IV_Mkt": round(iv_mkt, 4)
        })

    df = pd.DataFrame(rows)
    print(df[["Type", "T", "K", "Mkt", "Ana", "Err_A", "IV_Mkt"]].to_string(index=False))
    df.to_csv(f"{base_name}_prices.csv", index=False)
    log(f"Saved results to {base_name}_prices.csv")

def clear_numba_cache():
    for root, dirs, files in os.walk("src"):
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)

# -----------------------------
# MAIN
# -----------------------------
def main():
    clear_numba_cache()
    os.makedirs("results", exist_ok=True)
    
    ticker = "^SPX" 
    options_raw, S0_raw = fetch_options(ticker)
    if not options_raw:
        log(f"No liquidity for {ticker}")
        return

    # --- SCALING LOGIC ---
    #log(f"Scaling market data by 1/{S0_raw:.2f} for calibration...")
    S0_scaled = 1 #S0_raw
    options_scaled = []
    
    for opt in options_raw:
        scaled_opt = copy.copy(opt)
        try:
            scaled_opt.strike = opt.strike / S0_raw
            scaled_opt.market_price = opt.market_price / S0_raw
        except AttributeError:
            scaled_opt = EuropeanOption(
                opt.strike / S0_raw, opt.maturity, opt.option_type, opt.market_price / S0_raw
            )
        options_scaled.append(scaled_opt)
        
    options_scaled.sort(key=lambda x: (x.maturity, x.strike))
    options_raw.sort(key=lambda x: (x.maturity, x.strike))
    
    log(f"Target: {ticker} (Original S0={S0_raw:.2f}, Scaled S0=1.0) | N={len(options_scaled)}")
    
    r_curve = fetch_dynamic_risk_free_curve()
    q_curve = SimpleYieldCurve([0.0, 30.0], [0.11, 0.11])

    # 1. ANALYTICAL CALIBRATION
    cal_ana = HestonCalibrator(S0_scaled, r_curve=r_curve, q_curve=q_curve)
    init_guess = [2.0, 0.025, 0.1, -0.5, 0.015] 

    t0 = time.time()    
    res_ana = cal_ana.calibrate(options_scaled, init_guess)
    
    # Note: 'fun' is now MSE (Price squared error), so sqrt(fun) is RMSE of Price
    rmse_scaled = np.sqrt(res_ana['fun']) 
    log(f"Analytical (Scaled Price RMSE): {rmse_scaled:.6f} ({time.time()-t0:.2f}s)") 
    
    # 2. MONTE CARLO CALIBRATION (CONDITIONAL)
    res_mc = {}
    if RUN_MC:
        max_maturity = options_scaled[-1].maturity if options_scaled else 1.0
        n_steps_mc = max(int(max_maturity * 252), 50)
        log(f"Monte Carlo Config: 30,000 Paths | {n_steps_mc} Steps")
        cal_mc = HestonCalibratorMC(S0_scaled, r_curve=r_curve, q_curve=q_curve, n_paths=30_000, n_steps=n_steps_mc)
        
        t1 = time.time()
        try:
            res_mc = cal_mc.calibrate(options_scaled, init_guess)
            rmse_mc_scaled = np.sqrt(res_mc['fun']) 
            log(f"MonteCarlo (Scaled): rmse={rmse_mc_scaled:.6f} ({time.time()-t1:.2f}s)")
        except Exception as e:
            log(f"MC Fail: {e}")
            res_mc = res_ana
    else:
        log("Skipping Monte Carlo Calibration (RUN_MC = False)")
        res_mc = res_ana # Use Analytical results as placeholder

    params = ['kappa', 'theta', 'xi', 'rho', 'v0']
    df_params = pd.DataFrame({
        'Init': init_guess,
        'Ana': [res_ana.get(p, 0.0) for p in params],
        'MC': [res_mc.get(p, 0.0) for p in params]
    }, index=params)
    print("\nParameter Comparison:")
    print(df_params.to_string(float_format="{:.4f}".format))
    
    #save_results(ticker, S0_raw, r_curve, q_curve, res_ana, res_mc, options_raw, init_guess)

if __name__ == "__main__":
    main()