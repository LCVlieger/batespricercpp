import os
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize

# Assuming these are available from your local module
from heston_pricer.data import (
    fetch_treasury_rates_fred, get_market_implied_spot, 
    fetch_options, fetch_raw_data, ImpliedDividendCurve
)

# ---------------------------------------------------------
# STABLE ANALYTICAL PRICER (NON-SCALED)
# ---------------------------------------------------------
class HestonAnalyticalPricer:
    @staticmethod
    def price_european_call_vectorized(S0, K, T, r, q, kappa, theta, xi, rho, v0):
        # High density grid for numerical stability with large index values
        N_grid, u_max = 250, 100.0
        du = u_max / N_grid
        u = np.linspace(1e-8, u_max, N_grid)[:, np.newaxis] 
        
        K, T, r, q = np.atleast_1d(K), np.atleast_1d(T), np.atleast_1d(r), np.atleast_1d(q)
        T_mat, r_mat, q_mat, K_mat = T[np.newaxis,:], r[np.newaxis,:], q[np.newaxis,:], K[np.newaxis,:]

        def get_cf(phi):
            xi_s = np.maximum(xi, 1e-6)
            d = np.sqrt((rho * xi_s * phi * 1j - kappa)**2 + xi_s**2 * (phi * 1j + phi**2))
            g = (kappa - rho * xi_s * phi * 1j - d) / (kappa - rho * xi_s * phi * 1j + d)
            
            exp_neg_dT = np.exp(-d * T_mat)
            
            # Albrecher (2007) stable formulation
            C = (1/xi_s**2) * ((1 - exp_neg_dT) / (1 - g * exp_neg_dT)) * (kappa - rho * xi_s * phi * 1j - d)
            D = (kappa * theta / xi_s**2) * ((kappa - rho * xi_s * phi * 1j - d) * T_mat - 
                2 * (np.log(1 - g * exp_neg_dT) - np.log(1 - g + 1e-15)))
            
            # Drift uses the actual S0 price
            drift = 1j * phi * np.log(S0 * np.exp((r_mat - q_mat) * T_mat))
            return np.exp(C * v0 + D + drift)

        cf_p1, cf_p2 = get_cf(u - 1j), get_cf(u)
        
        # Integration logic using raw K and S0
        int_p1 = np.real((np.exp(-1j * u * np.log(K_mat)) * cf_p1) / (1j * u * S0 * np.exp((r_mat - q_mat) * T_mat)))
        int_p2 = np.real((np.exp(-1j * u * np.log(K_mat)) * cf_p2) / (1j * u))
        
        P1 = 0.5 + (1/np.pi) * np.sum(int_p1 * du, axis=0)
        P2 = 0.5 + (1/np.pi) * np.sum(int_p2 * du, axis=0)
        
        price = S0 * np.exp(-q_mat * T_mat) * P1 - K_mat * np.exp(-r_mat * T_mat) * P2
        return np.nan_to_num(np.maximum(price.flatten(), 0.0), nan=0.0)

# ---------------------------------------------------------
# STABLE CALIBRATOR (NON-SCALED)
# ---------------------------------------------------------
class LocalHestonCalibrator:
    def __init__(self, S0, r_curve, q_curve):
        self.S0 = S0
        self.r_curve = r_curve
        self.q_curve = q_curve

    def calibrate(self, options):
        strikes = np.array([o.strike for o in options])
        maturities = np.array([o.maturity for o in options])
        market_prices = np.array([o.market_price for o in options])
        r_vec = np.array([self.r_curve.get_rate(t) for t in maturities])
        q_vec = np.array([self.q_curve.get_rate(t) for t in maturities])
        
        # Volatility parameters remain scale-invariant
        bounds = [(1e-4, 0.5), (0.1, 8.0), (1e-4, 0.5), (0.01, 2.0), (-0.95, 0.0)]
        x0 = [0.04, 2.0, 0.04, 0.4, -0.7]

        def objective(p):
            v0, kappa, theta, xi, rho = p
            try:
                model_p = HestonAnalyticalPricer.price_european_call_vectorized(
                    self.S0, strikes, maturities, r_vec, q_vec, kappa, theta, xi, rho, v0
                )
                return np.mean((model_p - market_prices)**2) # MSE
            except:
                return 1e12 # Penalty for large index points

        def callback(xk):
            mse = objective(xk)
            print(f"   [Step] RMSE: {np.sqrt(mse):.6f} pts | v0:{xk[0]:.4f} k:{xk[1]:.2f} th:{xk[2]:.4f} xi:{xk[3]:.3f} rho:{xk[4]:.2f}")

        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, callback=callback, 
                       tol=1e-9, options={'eps': 1e-3, 'maxiter':500})
        
        return {**dict(zip(['v0', 'kappa', 'theta', 'xi', 'rho'], res.x)), "rmse": np.sqrt(res.fun)}

# ---------------------------------------------------------
# MAIN LIVE EXECUTION (NON-SCALED)
# ---------------------------------------------------------
def main():
    FRED_API_KEY = os.getenv("FRED_API_KEY")
    target_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Initializing Live Feed (Raw Units) for {target_date}...")
    
    # 1. Fetch Yield Curves and Spot
    r_curve = fetch_treasury_rates_fred(target_date, FRED_API_KEY)
    S0_actual = get_market_implied_spot("^SPX", r_curve)
    
    # 2. Fetch Dividends and Raw Options
    raw_df = fetch_raw_data("^SPX")
    q_curve = ImpliedDividendCurve(raw_df, S0_actual, r_curve)
    
    options_raw = fetch_options("^SPX", S0_actual, target_size=200)
    options_processed = []
    
    # 3. Process Puts -> Calls using Raw Prices
    for opt in options_raw:
        r_T = r_curve.get_rate(opt.maturity)
        q_T = q_curve.get_rate(opt.maturity)
        
        # Working with raw prices, no scaling division
        k = opt.strike
        p = opt.market_price
        
        if opt.option_type == "PUT":
            # Synthetic Call = Put + S0*exp(-qT) - K*exp(-rT)
            p += (S0_actual * np.exp(-q_T * opt.maturity) - k * np.exp(-r_T * opt.maturity))
            opt.option_type = "CALL"
        
        opt.strike, opt.market_price = k, p
        options_processed.append(opt)

    print(f"\nMarket Spot: {S0_actual:.2f}")
    print(f"Ready to calibrate on {len(options_processed)} options.")

    # 4. Calibration using the actual Market Spot
    calibrator = LocalHestonCalibrator(S0=S0_actual, r_curve=r_curve, q_curve=q_curve)
    res = calibrator.calibrate(options_processed)
    
    # 5. Output Results
    print("\n" + "*"*20 + " CALIBRATION COMPLETE " + "*"*20)
    for p in ['v0', 'kappa', 'theta', 'xi', 'rho']:
        print(f"  {p:<10}: {res[p]:.6f}")
    
    print(f"  Final RMSE: {res['rmse']:.6f} index points")

if __name__ == "__main__":
    main()