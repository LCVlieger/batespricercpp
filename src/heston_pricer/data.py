import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class MarketOption:
    strike: float
    maturity: float
    market_price: float
    option_type: str = "CALL" 
    bid: float = 0.0
    ask: float = 0.0

# --- PART 2: THE CALIBRATION FETCHER (OTM ONLY) ---
def fetch_options(ticker_symbol: str, S0: float, target_size: int = 300) -> List[MarketOption]:
    """
    Fetches strictly OTM options based on the provided S0.
    """
    ticker = yf.Ticker(ticker_symbol)
    expirations = ticker.options
    today = datetime.now()
    
    all_candidates = []
    MIN_T, MAX_T = 0.04, 1
    PHI = 0.0005 

    print(f"Scanning chains for OTM Calibration Data (Spot Reference: {S0:.2f})...")

    for exp_str in expirations:
        try:
            d = datetime.strptime(exp_str, "%Y-%m-%d")
            T = (d - today).days / 365.25
            if not (MIN_T <= T <= MAX_T): continue

            chain = ticker.option_chain(exp_str)
            
            # Process Puts
            puts = chain.puts.copy()
            puts['type'] = 'PUT'
            # STRICT OTM FILTER: Strike < Spot
            puts = puts[puts['strike'] < S0] 
            
            # Process Calls
            calls = chain.calls.copy()
            calls['type'] = 'CALL'
            # STRICT OTM FILTER: Strike > Spot
            calls = calls[calls['strike'] > S0]
            
            combined = pd.concat([puts, calls])
            
            for _, row in combined.iterrows():
                K = row['strike']
                bid = row.get('bid', 0.0)
                ask = row.get('ask', 0.0)
                mid = (bid + ask) / 2.0
                
                # Filters
                if bid < S0 * PHI: continue # Ghost bid
                spread = (ask - bid) / mid
                if spread > 0.40: continue # Wide spread
                
                # Moneyness Filter (0.75 to 1.3)
                if not (0.75 < K/S0 < 1.25): continue

                all_candidates.append({
                    'strike': K, 'maturity': T, 'market_price': mid,
                    'spread_ratio': spread, 'type': row['type'],
                    'bid': bid, 'ask': ask
                })
        except: continue

    if not all_candidates: return []
    
    # --- SAMPLING STRATEGY ---
    df = pd.DataFrame(all_candidates)
    unique_maturities = sorted(df['maturity'].unique())
    selected_indices = set()
    target_per_date = max(4, target_size // len(unique_maturities))
    
    # Use SKEW_POWER > 1.0 to prioritize ATM options (Low Skew)
    SKEW_POWER = 2.0 
    
    for mat in unique_maturities:
        mat_slice = df[df['maturity'] == mat]
        
        for opt_type in ['PUT', 'CALL']:
            # Puts: Sort Ascending (Deep OTM -> ATM) -> We want end of list
            # Calls: Sort Ascending (ATM -> Deep OTM) -> We want start of list
            
            candidates = mat_slice[mat_slice['type'] == opt_type].sort_values('strike')
            count = len(candidates)
            if count == 0: continue
            
            n_need = target_per_date // 2
            
            if count <= n_need:
                selected_indices.update(candidates.index)
            else:
                u = np.linspace(0, 1, n_need)
                if opt_type == 'CALL':
                    # Calls start at ATM (index 0). We want index 0.
                    # Quadratic mapping: 0->0, 1->1. Clusters at 0.
                    skewed_u = u ** SKEW_POWER
                else:
                    # Puts end at ATM (index -1). We want index -1.
                    # Quadratic mapping: 0->0, 1->1. Clusters at 1.
                    skewed_u = 1 - (1 - u) ** SKEW_POWER
                
                idx_positions = (skewed_u * (count - 1)).astype(int)
                selected_indices.update(candidates.iloc[idx_positions].index)

    final_df = df.loc[list(selected_indices)].copy()
    
    return [
        MarketOption(r['strike'], r['maturity'], r['market_price'], r['type'], r['bid'], r['ask'])
        for _, r in final_df.iterrows()
    ]


# --- PART 1: THE DEDICATED SPOT FETCHER ---
def get_market_implied_spot(ticker_symbol: str, r_curve) -> float:
    ticker = yf.Ticker(ticker_symbol)
    
    try:
        # Use fast_info for a quick anchor
        S_anchor = ticker.fast_info.get('last_price') or ticker.history(period="1d")['Close'].iloc[-1]
    except:
        return 0.0

    expirations = ticker.options
    if not expirations: return S_anchor
    today = datetime.now()
    
    for exp_str in expirations[:5]:
        d = datetime.strptime(exp_str, "%Y-%m-%d")
        T = (d - today).days / 365.25
        if T < 0.005: continue 
        
        try:
            chain = ticker.option_chain(exp_str)
            calls, puts = chain.calls.copy(), chain.puts.copy()
            calls['price'] = (calls['bid'] + calls['ask']) / 2
            puts['price'] = (puts['bid'] + puts['ask']) / 2
            
            merged = pd.merge(calls, puts, on='strike', suffixes=('_c', '_p'))
            atm_slice = merged[(merged['strike'] > S_anchor * 0.99) & (merged['strike'] < S_anchor * 1.01)].copy()
            
            if atm_slice.empty: continue
            
            # --- USE THE REAL RATE HERE ---
            r = r_curve.get_rate(T) 
            discount = np.exp(-r * T)
            
            # S = C - P + K*exp(-rT)
            atm_slice.loc[:, 'S_imp'] = atm_slice['price_c'] - atm_slice['price_p'] + (atm_slice['strike'] * discount)
            
            return float(atm_slice['S_imp'].median())
        except:
            continue
    return S_anchor