import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List
from nelson_siegel_svensson.calibrate import calibrate_nss_ols

@dataclass
class MarketOption:
    strike: float
    maturity: float
    market_price: float
    option_type: str = "CALL" 
    bid: float = 0.0
    ask: float = 0.0

# --- PART 1: RATE ENGINE (NSS/FRED) ---
class NSSYieldCurve:
    def __init__(self, curve_fit):
        self.curve = curve_fit
    def get_rate(self, T: float) -> float:
        return float(self.curve(max(T, 1e-4)))

def fetch_treasury_rates_fred(date_str: str, api_key: str) -> NSSYieldCurve:
    """Fetches official yields. Looks back up to 5 days to handle weekends/holidays."""
    series_map = {
        1/12: "DGS1MO", 3/12: "DGS3MO", 6/12: "DGS6MO", 
        1.0: "DGS1", 2.0: "DGS2", 5.0: "DGS5", 10.0: "DGS10", 30.0: "DGS30"
    }
    
    def fetch_for_date(d_str):
        mats, yields = [], []
        for tenor, s_id in series_map.items():
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id={s_id}&api_key={api_key}&file_type=json&observation_start={d_str}&observation_end={d_str}"
            try:
                res = requests.get(url, timeout=5).json()
                val = res['observations'][0]['value']
                if val != '.':
                    mats.append(tenor)
                    yields.append(float(val) / 100.0)
            except: continue
        return mats, yields

    target_dt = datetime.strptime(date_str, "%Y-%m-%d")
    for i in range(6):
        lookup_date = (target_dt - timedelta(days=i)).strftime("%Y-%m-%d")
        maturities, yields = fetch_for_date(lookup_date)
        if maturities:
            curve_fit, _ = calibrate_nss_ols(np.array(maturities), np.array(yields))
            return NSSYieldCurve(curve_fit)
    
    raise ValueError("Could not fetch FRED rates for the last 5 days.")

# --- PART 2: SMART DIVIDEND CURVE ---
class ImpliedDividendCurve:
    """Extracts implied dividend yield term structure using Put-Call Parity."""
    def __init__(self, df: pd.DataFrame, S0: float, r_curve):
        self.yields = {}
        # Group by maturity and find ATM points
        for T in sorted(df['T'].unique()):
            if T < 0.005: continue
            subset = df[df['T'] == T]
            r = r_curve.get_rate(T)
            
            # Find row closest to ATM Forward (F = S*exp(rT))
            F_approx = S0 * np.exp(r * T)
            valid = subset.dropna(subset=['C_MID', 'P_MID'])
            if valid.empty: continue
            
            row = valid.loc[(valid['STRIKE'] - F_approx).abs().idxmin()]
            # Solve q: S*exp(-qT) = C - P + K*exp(-rT)
            rhs = row['C_MID'] - row['P_MID'] + row['STRIKE'] * np.exp(-r * T)
            
            if rhs > 0:
                self.yields[T] = -np.log(rhs / S0) / T

    def get_rate(self, T: float) -> float:
        mats = sorted(self.yields.keys())
        if len(mats) > 1:
            # Linear interpolation/extrapolation
            return float(np.interp(T, mats, [self.yields[m] for m in mats]))
        elif len(mats) == 1:
            return float(self.yields[mats[0]])
        return 0.015

# --- PART 3: DATA FETCHING ---
def fetch_raw_data(ticker_symbol: str) -> pd.DataFrame:
    """Fetches broad options data for curve construction. Scans deep for long tenors."""
    ticker = yf.Ticker(ticker_symbol)
    today, all_rows = datetime.now(), []
    
    # SCAN DEPTH: Increased to 120 to catch 1Y and 2Y monthly/quarterly contracts
    exp_list = ticker.options
    limit = min(len(exp_list), 120)
    
    print(f"Scanning {limit} expirations for dividend extraction...")
    for exp_str in exp_list[:limit]: 
        try:
            T = (datetime.strptime(exp_str, "%Y-%m-%d") - today).days / 365.25
            if T < 0.005: continue
            chain = ticker.option_chain(exp_str)
            for opt_type, data in [('CALL', chain.calls), ('PUT', chain.puts)]:
                for _, row in data.iterrows():
                    all_rows.append({
                        'T': T, 
                        'STRIKE': round(float(row['strike']), 2),
                        'type': opt_type, 
                        'bid': row['bid'], 
                        'ask': row['ask']
                    })
        except: continue
    
    df = pd.DataFrame(all_rows)
    df_c = df[df['type']=='CALL'].drop(columns='type').rename(columns={'bid':'C_BID','ask':'C_ASK'}).set_index(['T','STRIKE'])
    df_p = df[df['type']=='PUT'].drop(columns='type').rename(columns={'bid':'P_BID','ask':'P_ASK'}).set_index(['T','STRIKE'])
    
    full = df_c.join(df_p, how='inner').reset_index()
    full['C_MID'], full['P_MID'] = (full['C_BID']+full['C_ASK'])/2, (full['P_BID']+full['P_ASK'])/2
    return full

def fetch_options(ticker_symbol: str, S0: float, target_size: int = 300) -> List[MarketOption]:
    """Hybrid OTM Filter: Fetches liquid OTM options within a stable moneyness window."""
    ticker = yf.Ticker(ticker_symbol)
    today, candidates = datetime.now(), []
    exp_list = ticker.options
    limit = min(len(exp_list), 100)

    # Define Moneyness Bounds (e.g., 0.85 to 1.15)
    # This prevents the model from hitting rho boundaries trying to fit extreme wings.
    lower_k = S0 * 0.85
    upper_k = S0 * 1.15

    for exp_str in exp_list[:limit]:
        try:
            T = (datetime.strptime(exp_str, "%Y-%m-%d") - today).days / 365.25
            if not (0.04 <= T <= 2.2): continue
            chain = ticker.option_chain(exp_str)
            
            for opt_type, data, filter_func in [
                ('PUT', chain.puts, lambda k: (k < S0) & (k >= lower_k)), 
                ('CALL', chain.calls, lambda k: (k > S0) & (k <= upper_k))
            ]:
                # Apply the combined OTM + Moneyness filter
                otm = data[filter_func(data['strike'])].copy()
                
                for _, row in otm.iterrows():
                    mid = (row['bid'] + row['ask']) / 2
                    # Liquidity check (bid > 0.05% of spot)
                    if row['bid'] > S0 * 0.0005:
                        candidates.append(MarketOption(
                            row['strike'], T, mid, opt_type, row['bid'], row['ask']
                        ))
        except: continue
    
    # Deterministic sampling
    if len(candidates) > target_size:
        indices = np.linspace(0, len(candidates)-1, target_size, dtype=int)
        candidates = [candidates[i] for i in indices]
    return candidates

def get_market_implied_spot(ticker_symbol: str, r_curve) -> float:
    ticker = yf.Ticker(ticker_symbol)
    try: S_anchor = ticker.fast_info['last_price']
    except: return 0.0
    
    # Use near-term options to refine Spot via Put-Call Parity
    for exp_str in ticker.options[:3]:
        try:
            T = (datetime.strptime(exp_str, "%Y-%m-%d") - datetime.now()).days / 365.25
            chain = ticker.option_chain(exp_str)
            merged = pd.merge(chain.calls, chain.puts, on='strike')
            merged['C'], merged['P'] = (merged.bid_x + merged.ask_x)/2, (merged.bid_y + merged.ask_y)/2
            atm = merged[(merged.strike > S_anchor*0.98) & (merged.strike < S_anchor*1.02)]
            if atm.empty: continue
            r = r_curve.get_rate(T)
            # S = C - P + K*exp(-rT)
            return (atm.C - atm.P + atm.strike * np.exp(-r*T)).median()
        except: continue
    return S_anchor