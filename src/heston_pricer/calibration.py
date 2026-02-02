import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq
from scipy.stats import norm
from dataclasses import dataclass
from .analytics import HestonAnalyticalPricer
from typing import List, Dict
from scipy.interpolate import interp1d
from collections import defaultdict
import warnings

def implied_volatility(price, S, K, T, r, q, option_type="CALL"):
    if price <= 0: return 0.0
    intrinsic = max(K * np.exp(-r*T) - S * np.exp(-q*T), 0) if option_type == "PUT" else max(S * np.exp(-q*T) - K * np.exp(-r*T), 0)
    if price < intrinsic: return 0.0
    def bs_price(sigma):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        val = (K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)) if option_type == "PUT" else (S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        return val - price
    try: return brentq(bs_price, 0.001, 5.0)
    except: return 0.0

import numpy as np
from scipy.optimize import minimize
from heston_pricer.analytics import HestonAnalyticalPricer

class HestonCalibrator:
    """
    Calibrates Heston model parameters to market option prices using 
    L-BFGS-B optimization on Mean Squared Error (MSE).
    """
    
    def __init__(self, S0, r_curve, q_curve):
        """
        Parameters:
        -----------
        S0 : float
            Underlying Spot Price.
        r_curve : object
            Yield curve object with a .get_rate(T) method.
        q_curve : object
            Dividend curve object with a .get_rate(T) method.
        """
        self.S0 = S0
        self.r_curve = r_curve
        self.q_curve = q_curve

    def calibrate(self, options):
        """
        Run the calibration routine.

        Parameters:
        -----------
        options : list
            List of option objects (must have .strike, .maturity, .market_price).

        Returns:
        --------
        dict
            Dictionary containing calibrated parameters and final RMSE.
        """
        strikes = np.array([o.strike for o in options])
        maturities = np.array([o.maturity for o in options])
        market_prices = np.array([o.market_price for o in options])
        
        # Vectorize rate retrieval
        r_vec = np.array([self.r_curve.get_rate(t) for t in maturities])
        q_vec = np.array([self.q_curve.get_rate(t) for t in maturities])
        
        # Bounds: v0, kappa, theta, xi, rho
        # Note: Rho is bounded [-0.95, 0.0] for stability in equity markets
        bounds = [(1e-4, 0.5), (0.1, 8.0), (1e-4, 0.5), (0.01, 2.0), (-0.95, 0.0)]
        x0 = [0.04, 2.0, 0.04, 0.4, -0.7]

        def objective(p):
            v0, kappa, theta, xi, rho = p
            try:
                model_p = HestonAnalyticalPricer.price_european_call_vectorized(
                    self.S0, strikes, maturities, r_vec, q_vec, kappa, theta, xi, rho, v0
                )
                # Optimize Mean Squared Error (MSE) for smoother gradient
                return np.mean((model_p - market_prices)**2) 
            except Exception:
                return 1e12 # Penalty for large index points failure

        def callback(xk):
            mse = objective(xk)
            print(f"   [Step] RMSE: {np.sqrt(mse):.6f} pts | v0:{xk[0]:.4f} k:{xk[1]:.2f} th:{xk[2]:.4f} xi:{xk[3]:.3f} rho:{xk[4]:.2f}")

        # Switched to L-BFGS-B for better handling of parameter bounds and gradients
        res = minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds, 
            callback=callback, 
            tol=1e-9, 
            options={'eps': 1e-3, 'maxiter': 500}
        )
        
        return {
            **dict(zip(['v0', 'kappa', 'theta', 'xi', 'rho'], res.x)), 
            "rmse": np.sqrt(res.fun)
        }
    
class HestonCalibratorMC:
    def __init__(self, S0, r_curve, q_curve, n_paths=30000, n_steps=100):
        self.S0, self.r_curve, self.q_curve = S0, r_curve, q_curve
        self.n_paths, self.n_steps = n_paths, n_steps
        self.z_noise, self.maturity_batches = None, defaultdict(list)
        self.max_T, self.dt = 0.0, 0.0

    def _precompute_batches(self, options):
        self.maturity_batches.clear()
        if not options: return
        self.max_T = max(opt.maturity for opt in options)
        self.dt = self.max_T / self.n_steps
        for opt in options: self.maturity_batches[opt.maturity].append(opt)
        if self.z_noise is None:
            np.random.seed(42) 
            self.z_noise = np.random.normal(0, 1, (2, self.n_steps, self.n_paths))

    def get_prices(self, params):
        kappa, theta, xi, rho, v0 = params
        results = {}
        for T_target, opts in self.maturity_batches.items():
            r_T, q_T = self.r_curve.get_rate(T_target), self.q_curve.get_rate(T_target)
            steps = max(1, min(self.n_steps, int(round(T_target / self.dt))))
            env = MarketEnvironment(self.S0, r_T, q_T, kappa, theta, xi, rho, v0)
            process = HestonProcess(env)
            paths = process.generate_paths(T_target, self.n_paths, steps, self.z_noise[:, :steps, :])
            S_final = paths[:, -1]
            prices = [np.mean(np.maximum(opt.strike - S_final, 0) if opt.option_type == "PUT" else np.maximum(S_final - opt.strike, 0)) * np.exp(-r_T * T_target) for opt in opts]
            results[T_target] = prices
        return results

    def objective(self, params):
        model_prices_map = self.get_prices(params)
        total_error = 0.0
        for T, opts in self.maturity_batches.items():
            m_prices = model_prices_map[T]
            for i, opt in enumerate(opts):
                moneyness = np.log(opt.strike / self.S0)
                weight = 1.0 + 5.0 * (moneyness**2)
                total_error += weight * ((m_prices[i] - opt.market_price) / (opt.market_price + 1e-8))**2
        return total_error

    def calibrate(self, options, init_guess=None):
        self._precompute_batches(options)
        x0 = init_guess if init_guess else [2.0, 0.05, 0.3, -0.7, 0.04]
        bounds = [(0.1, 10.0), (0.001, 2.0), (0.01, 5.0), (-0.999, 0.0), (0.001, 2.0)]
        def callback(xk):
             print(f"   [MonteCarlo] k={xk[0]:.2f}, theta={xk[1]:.3f}, xi={xk[2]:.2f}, rho={xk[3]:.2f}, v0={xk[4]:.3f}", flush=True)
        result = minimize(self.objective, x0, method='L-BFGS-B', bounds=bounds, callback=callback, tol=1e-8, options={'eps':1e-2})
        final_map = self.get_prices(result.x)
        sse_iv, count = 0.0, 0
        for T, opts in self.maturity_batches.items():
            r_T, q_T, m_prices = self.r_curve.get_rate(T), self.q_curve.get_rate(T), final_map[T]
            for i, opt in enumerate(opts):
                iv_mkt = implied_volatility(opt.market_price, self.S0, opt.strike, opt.maturity, r_T, q_T, opt.option_type)
                iv_model = implied_volatility(m_prices[i], self.S0, opt.strike, opt.maturity, r_T, q_T, opt.option_type)
                if iv_mkt > 0 and iv_model > 0:
                    sse_iv += (iv_model - iv_mkt) ** 2
                    count += 1
        return {"kappa": result.x[0], "theta": result.x[1], "xi": result.x[2], "rho": result.x[3], "v0": result.x[4], "success": result.success, "fun": result.fun, "rmse_iv": np.sqrt(sse_iv / count) if count > 0 else 0.0}