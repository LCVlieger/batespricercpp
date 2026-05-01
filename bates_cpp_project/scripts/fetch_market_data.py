#!/usr/bin/env python3
"""
fetch_market_data.py — Fetch options data and dump to JSON for the C++ pricer.

Usage:
    python fetch_market_data.py --ticker ^SPX --fred-api-key YOUR_KEY
    python fetch_market_data.py --ticker AAPL --fred-api-key YOUR_KEY -o data/market_data.json
"""

import argparse
import json
import os
import sys
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, time as dt_time
from scipy.interpolate import interp1d
from concurrent.futures import ThreadPoolExecutor

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

BOX_SPREAD = 0.0045   # SPX box-spread adjustment for risk-free rate


# ─────────────────────────────────────────────────────────────
# Yield curve
# ─────────────────────────────────────────────────────────────

def fetch_treasury_rates_fred(date_str: str, api_key: str):
    """Fetch UST rates from FRED and fit NSS curve via OLS."""
    import requests

    series = {1/12: "DGS1MO", 3/12: "DGS3MO", 6/12: "DGS6MO", 1.0: "DGS1", 2.0: "DGS2"}
    target_dt = datetime.strptime(date_str, "%Y-%m-%d")

    for i in range(6):
        d = (target_dt - timedelta(days=i)).strftime("%Y-%m-%d")
        mats, yields = [], []
        for tenor, s_id in series.items():
            try:
                url = (f"https://api.stlouisfed.org/fred/series/observations?"
                       f"series_id={s_id}&api_key={api_key}&file_type=json"
                       f"&observation_start={d}&observation_end={d}")
                val = requests.get(url, timeout=5).json()['observations'][0]['value']
                if val != '.':
                    mats.append(tenor)
                    yields.append(float(val) / 100.0)
            except Exception:
                continue

        if len(mats) >= 3:
            try:
                from nelson_siegel_svensson.calibrate import calibrate_nss_ols
                curve_fit, _ = calibrate_nss_ols(np.array(mats), np.array(yields))
                return curve_fit, BOX_SPREAD
            except ImportError:
                # Fallback: linear interpolation
                interp = interp1d(mats, yields, kind='linear',
                                  bounds_error=False, fill_value=(yields[0], yields[-1]))
                return interp, BOX_SPREAD

    raise ValueError("FRED rates unavailable for date range.")


def fallback_flat_rate(rate: float = 0.045):
    """Constant rate curve when FRED is unavailable."""
    return lambda T: rate, 0.0


# ─────────────────────────────────────────────────────────────
# Time-to-maturity
# ─────────────────────────────────────────────────────────────

def calculate_time_to_maturity(expiry: datetime, ticker: str) -> float:
    now = datetime.now()
    am_settled = ["^SPX", "^NDX", "^VIX", "^RUT", "^GDAXI"]
    is_am = any(ticker.startswith(t) for t in am_settled)
    is_monthly = expiry.weekday() == 4 and 15 <= expiry.day <= 21

    set_time = dt_time(9, 30) if (is_am and is_monthly) else dt_time(16, 0)
    delta = datetime.combine(expiry.date(), set_time) - now
    return max(delta.total_seconds() / (365.25 * 24 * 3600), 1e-6)


# ─────────────────────────────────────────────────────────────
# Implied dividend curve
# ─────────────────────────────────────────────────────────────

def build_dividend_curve(raw_df, S0, r_curve, spread, ticker):
    """Build implied dividend yield term structure from put-call parity."""
    is_index = ticker.startswith("^") or ticker in ["SPX", "NDX", "RUT"]
    yields_map = {}

    fundamental_q = 0.0
    if not is_index:
        try:
            raw_q = yf.Ticker(ticker).info.get('dividendYield', 0) / 100
            fundamental_q = raw_q if 0 <= raw_q <= 0.15 else 0.0
        except Exception:
            pass

    for T in sorted(raw_df['T'].unique()):
        if is_index:
            sub = raw_df[raw_df['T'] == T]
            r = float(r_curve(max(T, 1e-4))) + spread
            parity = (sub['C_MID'] - sub['P_MID']) + sub['STRIKE'] * np.exp(-r * T)
            intercept = np.median(parity)
            q = -np.log(intercept / S0) / T if (T > 1e-4 and intercept > 0) else 0.0
            yields_map[T] = float(np.clip(q, -0.01, 0.06))
        else:
            yields_map[T] = fundamental_q

    mats = np.sort(list(yields_map.keys()))
    vals = np.array([yields_map[m] for m in mats])
    q_interp = interp1d(mats, vals, kind='linear',
                        bounds_error=False, fill_value=(vals[0], vals[-1]))
    return q_interp


# ─────────────────────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────────────────────

def fetch_raw_data(ticker_symbol):
    """Fetch raw option chain data from yfinance."""
    import pandas as pd

    t_obj = yf.Ticker(ticker_symbol)
    exps = getattr(t_obj, 'options', [])
    if not exps:
        return pd.DataFrame()

    targets = [0.019, 0.08, 0.17, 0.25, 0.5, 0.75, 1.0, 1.25]
    selected = {min(exps, key=lambda x: abs(
        calculate_time_to_maturity(datetime.strptime(x, "%Y-%m-%d"), ticker_symbol) - t))
        for t in targets}

    def fetch_one(exp_str):
        try:
            T = calculate_time_to_maturity(datetime.strptime(exp_str, "%Y-%m-%d"), ticker_symbol)
            chain = t_obj.option_chain(exp_str)
            df_c = chain.calls[['strike', 'bid', 'ask']].rename(
                columns={'strike': 'STRIKE', 'bid': 'bC', 'ask': 'aC'})
            df_p = chain.puts[['strike', 'bid', 'ask']].rename(
                columns={'strike': 'STRIKE', 'bid': 'bP', 'ask': 'aP'})
            full = df_c.merge(df_p, on='STRIKE')
            full['C_MID'] = (full['bC'] + full['aC']) / 2
            full['P_MID'] = (full['bP'] + full['aP']) / 2
            full['T'] = T
            return full
        except Exception:
            return None

    workers = 4 if ticker_symbol.startswith("^") else 1
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = [r for r in ex.map(fetch_one, sorted(list(selected))) if r is not None]

    import pandas as pd
    return pd.concat(res, ignore_index=True) if res else pd.DataFrame()


def get_spot_price(ticker_symbol, raw_df, r_curve, spread):
    """Market-implied spot via put-call parity regression, with fallbacks."""
    from sklearn.linear_model import LinearRegression

    t_obj = yf.Ticker(ticker_symbol)
    if not ticker_symbol.startswith("^"):
        try:
            p = t_obj.fast_info.get('last_price', 0)
            if p > 0:
                return float(p)
        except Exception:
            pass

    if raw_df is not None and not raw_df.empty:
        a_T = min(raw_df['T'].unique(), key=lambda x: abs(x - 0.0833))
        sub = raw_df[raw_df['T'] == a_T]
        if len(sub) > 5:
            reg = LinearRegression().fit(
                sub['STRIKE'].values.reshape(-1, 1),
                (sub['C_MID'] - sub['P_MID']).values)
            if abs(reg.coef_[0]) > 1e-5:
                r = float(r_curve(max(a_T, 1e-4))) + spread
                f_price = -reg.intercept_ / reg.coef_[0]
                q_base = 0.013 if ticker_symbol.startswith("^") else 0.0
                return float(f_price / np.exp((r - q_base) * a_T))

    return float(t_obj.history(period="1d")['Close'].iloc[-1])


def fetch_options(ticker_symbol, S0, target_size=300):
    """Download OTM options chain and filter by liquidity."""
    if np.isnan(S0):
        return []

    t_obj = yf.Ticker(ticker_symbol)
    exps = [e for e in getattr(t_obj, 'options', [])
            if 0.04 <= calculate_time_to_maturity(
                datetime.strptime(e, "%Y-%m-%d"), ticker_symbol) <= 1.3]

    def process(exp_str):
        try:
            T = calculate_time_to_maturity(datetime.strptime(exp_str, "%Y-%m-%d"), ticker_symbol)
            chain = t_obj.option_chain(exp_str)
            local = []
            for typ, data, f in [
                ('PUT', chain.puts, lambda k: k < S0 * 0.98),
                ('CALL', chain.calls, lambda k: k > S0 * 1.02)
            ]:
                sub = data[f(data['strike']) & (data['strike'] > S0 * 0.7) & (data['strike'] < S0 * 1.3)]
                for _, r in sub.iterrows():
                    mid = (r['bid'] + r['ask']) / 2
                    bid, ask = r['bid'], r['ask']
                    if mid > 0.05 and bid > 0 and (ask - bid) / max(mid, 0.01) < 0.25:
                        local.append({
                            'strike': float(r['strike']),
                            'maturity': float(T),
                            'market_price': float(mid),
                            'option_type': typ,
                            'bid': float(bid),
                            'ask': float(ask),
                        })
            return local
        except Exception:
            return []

    workers = 4 if ticker_symbol.startswith("^") else 1
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = [item for sub in ex.map(process, exps) for item in sub]

    if len(res) > target_size:
        indices = np.linspace(0, len(res) - 1, target_size, dtype=int)
        res = [res[i] for i in indices]

    return res


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch market data for Bates C++ pricer")
    parser.add_argument("--ticker", type=str, default="^SPX", help="Ticker symbol (e.g., ^SPX, AAPL)")
    parser.add_argument("--fred-api-key", type=str, default="", help="FRED API key for yield curve")
    parser.add_argument("-o", "--output", type=str, default="data/market_data.json", help="Output JSON path")
    parser.add_argument("--target-size", type=int, default=300, help="Target number of options")
    args = parser.parse_args()

    print(f"[1] Fetching data for {args.ticker}...")

    # Rate curve
    date_str = datetime.now().strftime("%Y-%m-%d")
    if args.fred_api_key:
        try:
            r_curve, spread = fetch_treasury_rates_fred(date_str, args.fred_api_key)
            print("  Rate curve: FRED NSS fit")
        except Exception as e:
            print(f"  FRED failed ({e}), using flat rate fallback")
            r_curve, spread = fallback_flat_rate()
    else:
        print("  No FRED API key provided, using flat rate (4.5%)")
        r_curve, spread = fallback_flat_rate()

    # Raw data for dividend curve and spot
    print(f"[2] Fetching raw option chain...")
    raw_df = fetch_raw_data(args.ticker)

    # Spot
    print(f"[3] Computing spot price...")
    S0 = get_spot_price(args.ticker, raw_df, r_curve, spread)
    print(f"  S0 = {S0:.2f}")

    # Dividend curve
    print(f"[4] Building dividend curve...")
    if not raw_df.empty:
        q_interp = build_dividend_curve(raw_df, S0, r_curve, spread, args.ticker)
    else:
        q_interp = lambda T: 0.0

    # Options
    print(f"[5] Fetching OTM options (target: {args.target_size})...")
    options = fetch_options(args.ticker, S0, target_size=args.target_size)
    print(f"  Found {len(options)} options")

    # Attach per-option r and q
    for o in options:
        T = o['maturity']
        o['r'] = float(r_curve(max(T, 1e-4))) + spread
        o['q'] = float(q_interp(max(T, 1e-4)))

    # Build rate/dividend curve snapshots for reference
    sample_tenors = [0.083, 0.25, 0.5, 1.0]
    rate_curve_snap = {f"{t}Y": float(r_curve(t)) + spread for t in sample_tenors}
    div_curve_snap = {f"{t}Y": float(q_interp(t)) for t in sample_tenors}

    # Assemble JSON
    output = {
        "ticker": args.ticker,
        "S0": S0,
        "fetch_date": date_str,
        "rate_curve": rate_curve_snap,
        "dividend_curve": div_curve_snap,
        "options": options,
    }

    # Write
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[DONE] Wrote {len(options)} options to {args.output}")


if __name__ == "__main__":
    main()
