import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from scipy.interpolate import PchipInterpolator

# Assuming these are your existing utility functions
from heston_pricer.data import (
    fetch_treasury_rates_fred, get_market_implied_spot, 
    fetch_options, fetch_raw_data
)
from heston_pricer.calibration import HestonCalibrator

# =================================================================
# 1. ROBUST DIVIDEND ENGINE (REGRESSION METHOD)
# =================================================================
class ImpliedDividendCurve:
    """
    Extracts implied dividends by regressing the entire strike cross-section.
    Solves: (C - P) = [exp(-rT) * F] - [exp(-rT)] * K
    """
    def __init__(self, df: pd.DataFrame, S0: float, r_curve):
        self.yields = {}
        for T in sorted(df['T'].unique()):
            if T < 0.005: continue
            
            # 1. Define the mask first using the original dataframe 'df'
            mask = (df['T'] == T) & (df['STRIKE'] > S0 * 0.85) & (df['STRIKE'] < S0 * 1.15)
            subset = df[mask].dropna(subset=['C_MID', 'P_MID'])
            
            # 2. Need at least a few points for a valid linear regression
            if len(subset) < 5: 
                continue
            
            r = r_curve.get_rate(T)
            X = subset['STRIKE'].values.reshape(-1, 1)
            y = (subset['C_MID'] - subset['P_MID']).values
            
            # 3. Linear Regression to find the Implied Forward
            reg = LinearRegression().fit(X, y)
            
            # Slope (coef_) is -exp(-rT). Intercept is exp(-rT) * F.
            # We use the absolute value of the slope to get the discount factor.
            implied_discount = -reg.coef_[0] 
            
            # Safety check: discount factor should be positive and reasonably near 1
            if implied_discount <= 0:
                continue
                
            implied_F = reg.intercept_ / implied_discount
            
            # 4. Solve for q: F = S0 * exp((r - q) * T) -> q = r - ln(F/S0)/T
            q = r - (np.log(implied_F / S0) / T)
            
            # Sanity Check: Keep q within realistic market bounds (-2% to 15%)
            if 0.15 > q > -0.02:
                self.yields[T] = q

            # Build the interpolator once during init
            mats = sorted(self.yields.keys())
            if len(mats) > 1:
                vals = [self.yields[m] for m in mats]
                self.interpolator = PchipInterpolator(mats, vals, extrapolate=True)
            else:
                self.interpolator = None

    def get_rate(self, T: float) -> float:
            if self.interpolator:
                # Pchip is much smoother than np.interp
                return float(self.interpolator(T))
            return self.yields.get(list(self.yields.keys())[0], 0.015) if self.yields else 0.015

# =================================================================
# 2. UPDATED MAIN EXECUTION
# =================================================================
def print_curves(r_curve, q_curve):
    print("\n" + "="*60)
    print(f"{'Tenor':<10} | {'Risk-Free (r)':<15} | {'Div Yield (q)':<15}")
    print("-" * 60)
    tenors = [(1/12, "1 Month"), (0.25, "3 Month"), (0.5, "6 Month"), (1.0, "1 Year"), (2.0, "2 Year")]
    for t, label in tenors:
        print(f"{label:<10} | {r_curve.get_rate(t)*100:>13.4f}% | {q_curve.get_rate(t)*100:>13.4f}%")
    print("="*60 + "\n")

def main():
    FRED_API_KEY = os.getenv("FRED_API_KEY")
    target_date = datetime.now().strftime("%Y-%m-%d")
    ticker = "^SPX"

    print(f"Initializing Robust Heston Calibration for {ticker}...")
    
    # 1. Fetch Yields and Spot
    r_curve = fetch_treasury_rates_fred(target_date, FRED_API_KEY)
    S0_actual = get_market_implied_spot(ticker, r_curve)
    
    # 2. Robust Dividend Curve via Regression
    print("Building Robust Dividend Surface...")
    raw_df = fetch_raw_data(ticker)
    q_curve = ImpliedDividendCurve(raw_df, S0_actual, r_curve)
    print_curves(r_curve, q_curve)
    
    # 3. Fetch Options for Calibration
    options_raw = fetch_options(ticker, S0_actual, target_size=250)
    
    options_processed = []
    print("Converting Surface to Synthetic Calls (using Robust q)...")
    
    for opt in options_raw:
        r_T = r_curve.get_rate(opt.maturity)
        q_T = q_curve.get_rate(opt.maturity)
        
        # Standardize everything to Call Prices
        if opt.option_type == "PUT":
            # Call = Put + S0*exp(-qT) - K*exp(-rT)
            price = opt.market_price + (S0_actual * np.exp(-q_T * opt.maturity) - 
                                        opt.strike * np.exp(-r_T * opt.maturity))
            opt.market_price = price
            opt.option_type = "CALL"
        
        options_processed.append(opt)

    print(f"Calibration starting on {len(options_processed)} options. S0: {S0_actual:.2f}")

    # 4. Calibration 
    # NOTE: Ensure your HestonCalibrator uses the Vectorized Albrecher pricer internally
    calibrator = HestonCalibrator(S0=S0_actual, r_curve=r_curve, q_curve=q_curve)
    
    # Suggesting differential_evolution for global search of rho/xi
    res = calibrator.calibrate(options_processed)
    
    # 5. Output
    print("\n" + "*"*20 + " CALIBRATION COMPLETE " + "*"*20)
    for p in ['v0', 'kappa', 'theta', 'xi', 'rho']:
        print(f"  {p:<10}: {res[p]:.6f}")
    
    print(f"  Final RMSE: {res['rmse']:.6f} index points")

if __name__ == "__main__":
    main()