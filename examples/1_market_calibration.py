import time
import os
import numpy as np
import yfinance as yf
from datetime import datetime
from heston_pricer.calibration import HestonCalibrator
from heston_pricer.data import (
    MarketOption, NSSYieldCurve, ImpliedDividendCurve, 
    fetch_treasury_rates_fred, get_market_implied_spot, 
    fetch_options, fetch_raw_data
)

FRED_API_KEY = os.getenv("FRED_API_KEY") 
TARGET_TICKER = "^SPX"

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def get_smart_risk_free_curve(date_str):
    if FRED_API_KEY:
        try: return fetch_treasury_rates_fred(date_str, FRED_API_KEY)
        except Exception as e: log(f"FRED Failed: {e}")
    class Flat:
        def get_rate(self, T): return 0.042
    return Flat()

def print_market_curves(r_curve, q_curve):
    print("\n" + "="*65)
    print(f"{'Tenor':<12} | {'Risk-Free (r)':<18} | {'Implied Div (q)':<18}")
    print("-" * 65)
    tenors = [1/12, 3/12, 6/12, 1.0, 2.0]
    labels = ["1 Month", "3 Month", "6 Month", "1 Year", "2 Year"]
    for t, label in zip(tenors, labels):
        r, q = r_curve.get_rate(t), q_curve.get_rate(t)
        print(f"{label:<12} | {r*100:>15.4f}% | {q*100:>15.4f}%")
    print("="*65 + "\n")

def main():
    target_date = datetime.now().strftime("%Y-%m-%d")
    r_curve = get_smart_risk_free_curve(target_date)
    
    S0_downloaded = yf.Ticker(TARGET_TICKER).fast_info['last_price']
    S0_implied = get_market_implied_spot(TARGET_TICKER, r_curve)
    log(f"Spot Check -> Downloaded: {S0_downloaded:.2f} | Implied: {S0_implied:.2f}")

    raw_df = fetch_raw_data(TARGET_TICKER)
    q_curve = ImpliedDividendCurve(raw_df, S0_implied, r_curve)
    
    # Check the Term Structure
    print_market_curves(r_curve, q_curve)
    
    options_raw = fetch_options(TARGET_TICKER, S0_implied, target_size=300)
    options_scaled = []
    for opt in options_raw:
        if opt.option_type == "PUT":
            r_T, q_T = r_curve.get_rate(opt.maturity), q_curve.get_rate(opt.maturity)
            opt.market_price += S0_implied*np.exp(-q_T*opt.maturity) - opt.strike*np.exp(-r_T*opt.maturity)
            opt.option_type = "CALL"
        opt.strike /= S0_implied; opt.market_price /= S0_implied
        options_scaled.append(opt)

    calibrator = HestonCalibrator(S0=1.0, r_curve=r_curve, q_curve=q_curve)
    res = calibrator.calibrate(options_scaled)
    
    print("\n" + "*"*20 + " FINAL HESTON PARAMETERS " + "*"*20)
    for p in ['v0', 'kappa', 'theta', 'xi', 'rho']:
        print(f"  {p:<10}: {res[p]:.6f}")
    print(f"  RMSE (Scaled): {res['rmse']:.6f}")
    print("*"*65)

if __name__ == "__main__":
    main()