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
    from heston_pricer.data import fetch_options, get_market_implied_spot
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

def extract_implied_dividends(ticker_symbol: str, S0: float, r_curve: SimpleYieldCurve) -> SimpleYieldCurve:
    """
    Extracts the Term Structure of Dividend Yields (q) from Market Options.
    Uses Put-Call Parity: C - P = S0 * exp(-qT) - K * exp(-rT)
    Solved for q: q = -ln( (C - P + K*exp(-rT)) / S0 ) / T
    """
    log(f"Extracting implied dividend term structure for {ticker_symbol}...")
    
    ticker = yf.Ticker(ticker_symbol)
    expirations = ticker.options
    today = datetime.now()
    
    tenors = []
    q_rates = []

    # Iterate through maturities to build the curve
    for exp_str in expirations:
        try:
            d = datetime.strptime(exp_str, "%Y-%m-%d")
            T = (d - today).days / 365.25
            
            # Filter: Ignore very short (noise) and very long (illiquid)
            if not (0.05 <= T <= 2.5): continue

            # Get Rates
            r = r_curve.get_rate(T)
            discount_r = np.exp(-r * T)
            
            # Fetch Chain
            chain = ticker.option_chain(exp_str)
            
            # Prepare Dataframes
            calls = chain.calls[['strike', 'bid', 'ask']].copy()
            puts = chain.puts[['strike', 'bid', 'ask']].copy()
            calls['mid'] = (calls['bid'] + calls['ask']) / 2
            puts['mid'] = (puts['bid'] + puts['ask']) / 2
            
            # Find ATM Overlap (Intersection)
            merged = pd.merge(calls, puts, on='strike', suffixes=('_c', '_p'))
            
            # Filter for ATM ONLY (Moneyness 0.98 - 1.02)
            # We only trust the dividend signal from liquid ATM options
            atm = merged[
                (merged['strike'] > S0 * 0.98) & 
                (merged['strike'] < S0 * 1.02)
            ].copy()
            
            if atm.empty: continue
            
            # Solve for q for each ATM pair
            # LHS = exp(-qT) = (C - P + K*exp(-rT)) / S0
            atm.loc[:,'parity_balance'] = (atm['mid_c'] - atm['mid_p'] + atm['strike'] * discount_r) / S0
            
            # Filter invalid logs (arbitrage violations)
            valid = atm[atm['parity_balance'] > 0].copy()
            if valid.empty: continue
            
            valid['q_implied'] = -np.log(valid['parity_balance']) / T
            
            # Take the median q for this maturity (robust to outliers)
            avg_q = valid['q_implied'].median()
            
            # Sanity check: Dividends shouldn't be -50% or +50%
            if -0.05 < avg_q < 0.15:
                tenors.append(T)
                q_rates.append(avg_q)

        except Exception:
            continue
            
    if not tenors:
        log("! Warning: Could not extract implied dividends. Defaulting to 1.1%.")
        return SimpleYieldCurve([0.0, 30.0], [0.011, 0.011])
        
    log(f"-> Derived q-curve with {len(tenors)} points (Avg: {np.mean(q_rates):.2%})")
    
    # Sort by time
    sorted_pairs = sorted(zip(tenors, q_rates))
    return SimpleYieldCurve([t for t, _ in sorted_pairs], [q for _, q in sorted_pairs])


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
    #df.to_csv(f"{base_name}_prices.csv", index=False)
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
    ticker_symbol = "^SPX"
    
    # 1. Get the Raw Anchor (Downloaded) Spot first for comparison
    ticker = yf.Ticker(ticker_symbol)
    try:
        S0_downloaded = ticker.fast_info.get('last_price') or ticker.history(period="1d")['Close'].iloc[-1]
    except:
        log("! Critical: Could not fetch any price data.")
        return

    r_curve = fetch_dynamic_risk_free_curve()
    # 2. Get the Market-Implied Spot (The "Truth" from Put-Call Parity)
    S0_implied = get_market_implied_spot(ticker_symbol, r_curve)
    
    # 3. Print the Comparison
    # This will show you the exact "gap" that was causing your calibration errors
    diff = S0_implied - S0_downloaded
    log("="*50)
    log(f"SPOT PRICE COMPARISON for {ticker_symbol}")
    log(f"-> Downloaded Spot: {S0_downloaded:10.2f}")
    log(f"-> Implied Spot:    {S0_implied:10.2f}")
    log(f"-> Basis (Gap):     {diff:10.2f} ({diff/S0_downloaded:.4%})")
    log("="*50)

    # 4. Fetch purely OTM options using the Implied Spot as the centergi
    # This ensures your Call/Put split is perfectly aligned with the market
    options_raw = fetch_options(ticker_symbol, S0_implied, target_size=300)
    
    if not options_raw:
        log("No liquidity found for calibration.")
        return

    # 5. Proceed with Curves and Scaling
    q_curve = extract_implied_dividends(ticker_symbol, S0_implied, r_curve) #SimpleYieldCurve([0.0, 30.0], [0.011, 0.011])
    print_curve("Dividend (q)", q_curve)
    # We scale using the Implied Spot (our new "1.0")
    S0_raw = S0_implied 
    S0_scaled = 1.0
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
    
    save_results(ticker_symbol, S0_raw, r_curve, q_curve, res_ana, res_mc, options_raw, init_guess)

def print_curve(name, curve):
    print(f"\n--- {name} Curve ---")
    print(f"{'Tenor (T)':<12} | {'Rate (%)':<10}")
    print("-" * 25)
    # Most curve implementations have .tenors and .rates or .times and .values
    # Adjust the attribute names based on your SimpleYieldCurve definition
    for t, r in zip(curve.tenors, curve.rates):
        print(f"{t:<12.4f} | {r*100:<10.2f}%")


def get_implied_spot(options, r_curve, q_curve):
    # Filter for the shortest maturity available
    min_T = min(o.maturity for o in options)
    short_opts = [o for o in options if o.maturity == min_T]
    
    # Find overlapping strikes
    calls = {o.strike: o.market_price for o in short_opts if o.option_type == "CALL"}
    puts = {o.strike: o.market_price for o in short_opts if o.option_type == "PUT"}
    
    common_strikes = set(calls.keys()).intersection(set(puts.keys()))
    if not common_strikes:
        return None
        
    # Calculate Implied Spot for each pair: S = (C - P + K*exp(-rT)) / exp(-qT)
    r = r_curve.get_rate(min_T)
    q = q_curve.get_rate(min_T)
    spots = []
    
    for K in common_strikes:
        C = calls[K]
        P = puts[K]
        S_imp = (C - P + K * np.exp(-r * min_T)) / np.exp(-q * min_T)
        spots.append(S_imp)
        
    return np.mean(spots)



if __name__ == "__main__":
    main()

