import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import os
import sys

# Ensure local imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from heston_pricer.calibration import HestonCalibrator, SimpleYieldCurve, implied_volatility
    from heston_pricer.instruments import EuropeanOption, OptionType
    from heston_pricer.data import MarketOption
except ImportError:
    print("Error: Could not find heston_pricer package. Run from the project root.")
    sys.exit(1)

# --- CONFIGURATION ---
TARGET_DATE = "2022-03-25" 
# Point this to your new .txt file
CSV_FILE_PATH = "src/spx_eod_202203.txt" 

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ---------------------------------------------------------
# 1. DYNAMIC RISK-FREE RATE (Matching Jan 2022 Market)
# ---------------------------------------------------------
def fetch_historical_rates_dynamic(date_str):
    log(f"Fetching Treasury yields for {date_str}...")
    tickers = ["^IRX", "^FVX", "^TNX", "^TYX"]
    target_dt = pd.to_datetime(date_str)
    
    try:
        df = yf.download(tickers, start=target_dt - timedelta(days=5), 
                         end=target_dt + timedelta(days=2), progress=False)['Close']
        idx = df.index.get_indexer([target_dt], method='nearest')[0]
        row = df.iloc[idx]
        
        rates_map = {0.25: float(row['^IRX'])/100, 5.0: float(row['^FVX'])/100, 
                     10.0: float(row['^TNX'])/100, 30.0: float(row['^TYX'])/100}
        
        tenors = [0.08, 0.25, 0.5, 1.0, 2.0, 3.0] 
        final_rates = []
        for t in tenors:
            if t <= 0.25: r = rates_map[0.25]
            elif t <= 5.0:
                ratio = (t - 0.25) / (5.0 - 0.25)
                r = rates_map[0.25] + ratio * (rates_map[5.0] - rates_map[0.25])
            else:
                ratio = (t - 5.0) / (10.0 - 5.0)
                r = rates_map[5.0] + ratio * (rates_map[10.0] - rates_map[5.0])
            final_rates.append(r)
        return SimpleYieldCurve(tenors, final_rates)
    except:
        return SimpleYieldCurve([0.1, 30.0], [0.015, 0.015])
# ---------------------------------------------------------
# 2. "LAZY" FILTERING (Matches the Notebook's logic)
# ---------------------------------------------------------
def smart_filter_options(df_raw, S0, target_size=300):
    # This filter replicates the notebook's approach:
    # 1. It keeps ITM options (which dominate the error).
    # 2. It essentially just takes a slice of the chain.
    
    all_candidates = []
    
    for _, row in df_raw.iterrows():
        K = float(row['STRIKE'])
        mid = (float(row['C_BID']) + float(row['C_ASK'])) / 2.0
        
        # WIDE FILTER: Match notebook's 3200-4800 range (approx 0.65 to 1.05 moneyness)
        if not (0.65 < K/S0 < 1.05): continue

        all_candidates.append({
            'strike': K, 
            'maturity': float(row['T']), 
            'market_price': mid, 
            'type': 'CALL' # Notebook uses Calls only
        })

    if not all_candidates: return []
    
    # Simple sampling to fit target size
    df = pd.DataFrame(all_candidates)
    if len(df) > target_size:
        df = df.sample(n=target_size, random_state=42).sort_values('strike')
        
    return [MarketOption(r['strike'], r['maturity'], r['market_price'], r['type']) for _, r in df.iterrows()]

# ---------------------------------------------------------
# 3. SPX LOADER (Notebook Replication Mode)
# ---------------------------------------------------------
def load_spx_txt(file_path, target_date):
    log(f"Parsing SPX (Replicating Notebook Data Selection)...")
    df = pd.read_csv(file_path, low_memory=False, skipinitialspace=True)
    df.columns = df.columns.str.strip(' []') 
    
    # Numeric Conversion
    cols = ['STRIKE', 'C_BID', 'C_ASK', 'UNDERLYING_LAST'] 
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['STRIKE', 'UNDERLYING_LAST'])
    df['QUOTE_DATE'] = pd.to_datetime(df['QUOTE_DATE'])
    df['EXPIRE_DATE'] = pd.to_datetime(df['EXPIRE_DATE'])
    
    day_data = df[df['QUOTE_DATE'] == pd.to_datetime(target_date)].copy()
    S0 = day_data['UNDERLYING_LAST'].iloc[0]
    day_data['T'] = (day_data['EXPIRE_DATE'] - day_data['QUOTE_DATE']).dt.days / 365.25
    
    # --- CRITICAL CHANGE: USE CALLS ONLY ---
    calls = day_data.copy()
    calls['type'] = 'CALL' 
    
    # Pass ONLY Calls to the filter
    filtered = smart_filter_options(calls, S0, target_size=300)
    return filtered, S0, day_data

# ---------------------------------------------------------
# 4. IMPLIED DIVIDENDS (Valid for European SPX)
# ---------------------------------------------------------
def extract_dividends(day_data, S0, r_curve):
    log("Extracting Implied Yield (Put-Call Parity)...")
    tenors, q_rates = [], []
    for T, group in day_data.groupby('T'):
        if not (0.05 <= T <= 1.5): continue
        merged = pd.merge(group[['STRIKE','C_BID','C_ASK']], 
                          group[['STRIKE','P_BID','P_ASK']], on='STRIKE', suffixes=('_c','_p'))
        atm = merged[(merged['STRIKE'] > S0*0.99) & (merged['STRIKE'] < S0*1.01)].copy()
        if atm.empty: continue
        
        mid_c, mid_p = (atm['C_BID']+atm['C_ASK'])/2, (atm['P_BID']+atm['P_ASK'])/2
        r = r_curve.get_rate(T)
        lhs = (mid_c - mid_p + atm['STRIKE'] * np.exp(-r*T)) / S0
        q_est = -np.log(lhs[lhs > 0]) / T
        if not q_est.empty and 0 < q_est.median() < 0.05:
            tenors.append(T); q_rates.append(q_est.median())

    if not tenors: return SimpleYieldCurve([0.1, 30.0], [0.014, 0.014])
    return SimpleYieldCurve(tenors, q_rates)

# ---------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------
def main():
    r_curve = fetch_historical_rates_dynamic(TARGET_DATE)
    options_final, S0_hist, day_data = load_spx_txt(CSV_FILE_PATH, TARGET_DATE)
    q_curve = SimpleYieldCurve([0.1, 30.0], [0.0, 0.0]) #extract_dividends(day_data, S0_hist, r_curve)
    
    log(f"Backtest Ready: S0={S0_hist:.2f} | Options={len(options_final)} (SPX European)")

    options_scaled = []
    for opt in options_final:
        opt_obj = EuropeanOption(opt.strike/S0_hist, opt.maturity, opt.option_type)
        opt_obj.market_price = opt.market_price/S0_hist
        options_scaled.append(opt_obj)
    
    #use raw data instead
    cal = HestonCalibrator(S0_hist, r_curve, q_curve)
    init_guess = [3.0, 0.05, 0.3, -0.8, 0.1] 
    
    t0 = time.time()
    res = cal.calibrate(options_final, init_guess)
    
    log(f"Calibration Done ({time.time()-t0:.2f}s). RMSE: {np.sqrt(res['fun']):.6f}")
    print("\nCalibrated Parameters:")
    for k, v in res.items():
        if k not in ['success', 'fun']: print(f"  {k}: {v:.4f}")



if __name__ == "__main__":
    main()