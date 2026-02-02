import numpy as np
from scipy.stats import norm

class HestonAnalyticalPricer:
    @staticmethod
    def price_european_call_vectorized(S0, K, T, r, q, kappa, theta, xi, rho, v0):
        # 1. Type Prep & Setup
        K, T, r, q = np.atleast_1d(K), np.atleast_1d(T), np.atleast_1d(r), np.atleast_1d(q)
        N_grid, u_max = 200, 100.0
        du = u_max / N_grid
        u = np.linspace(1e-8, u_max, N_grid)[:, np.newaxis] 

        T_mat, r_mat, q_mat, K_mat = T[np.newaxis,:], r[np.newaxis,:], q[np.newaxis,:], K[np.newaxis,:]
        
        # 2. Characteristic Function with Hard Guards
        def get_cf(phi):
            xi_s = max(xi, 1e-7)
            # Use Albrecher Form
            d = np.sqrt((rho * xi_s * phi * 1j - kappa)**2 + xi_s**2 * (phi * 1j + phi**2))
            g = (kappa - rho * xi_s * phi * 1j - d) / (kappa - rho * xi_s * phi * 1j + d)
            
            # Guard the exponent to prevent 'inf' before it happens
            dT = d * T_mat
            exp_neg_dT = np.exp(-np.clip(dT.real, -50, 50) + 1j * dT.imag)
            
            denom = 1 - g * exp_neg_dT + 1e-15
            C = (1/xi_s**2) * ((1 - exp_neg_dT) / denom) * (kappa - rho * xi_s * phi * 1j - d)
            
            # Stable log handling
            log_val = np.log(np.maximum(1e-15, 1 - g * exp_neg_dT)) - np.log(np.maximum(1e-15, 1 - g))
            D = (kappa * theta / xi_s**2) * ((kappa - rho * xi_safe * phi * 1j - d) * T_mat - 2 * log_val)
            
            exponent = C * v0 + D
            # HARD CLIP: exp(700) is the limit for float64. 50 is plenty for a normalized price.
            return np.exp(np.clip(exponent.real, -50, 50) + 1j * exponent.imag)

        # 3. Lewis (2001) Integration - Single Integral Form (More stable)
        # We integrate phi = u - 0.5j
        phi = u - 0.5j
        cf = get_cf(phi)
        
        # F = S0 * exp((r-q)T) -> Forward price
        # Moneyness term: log(F/K)
        log_F_K = np.log(S0 / K_mat) + (r_mat - q_mat) * T_mat
        
        # The Lewis integrand is typically smoother for calibration
        integrand = np.real(np.exp(1j * u * log_F_K) * cf / (u**2 + 0.25))
        
        integral = np.sum(integrand * du, axis=0)
        
        # Price assembly
        price = S0 * np.exp(-q_mat * T_mat) - (np.sqrt(S0 * K_mat) * np.exp(-(r_mat + q_mat) * T_mat / 2) / np.pi) * integral
        
        return np.nan_to_num(np.maximum(price.flatten(), 0.0), nan=0.0, posinf=1e6)

    @staticmethod
    def price_european_call(S0, K, T, r, q, kappa, theta, xi, rho, v0):
        return float(HestonAnalyticalPricer.price_european_call_vectorized(S0, K, T, r, q, kappa, theta, xi, rho, v0)[0])

    @staticmethod
    def price_european_put(S0, K, T, r, q, kappa, theta, xi, rho, v0):
        call = HestonAnalyticalPricer.price_european_call(S0, K, T, r, q, kappa, theta, xi, rho, v0)
        return call - S0 * np.exp(-q * T) + K * np.exp(-r * T)