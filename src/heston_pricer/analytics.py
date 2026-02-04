from scipy.stats import norm
import numpy as np
from scipy.optimize import brentq


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
class BatesAnalyticalPricer:
    @staticmethod
    def price_vectorized(S0, K, T, r, q, types, kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j):
        # 1. UPGRADED RESOLUTION: 4000 points up to u=2500
        N_grid, u_max = 4000, 2500.0 
        u = np.linspace(0, u_max, N_grid + 1)[:, np.newaxis]
        # We start at 0 but the integrand for P2 at u=0 is 0.5 (limit logic)
        u[0] = 1e-10 
        
        K, T, r, q = np.atleast_1d(K), np.atleast_1d(T), np.atleast_1d(r), np.atleast_1d(q)
        T_mat, r_mat, q_mat, K_mat = T[np.newaxis,:], r[np.newaxis,:], q[np.newaxis,:], K[np.newaxis,:]
        F_mat = S0 * np.exp((r_mat - q_mat) * T_mat)

        def get_cf(phi):
            xi_s = np.maximum(xi, 1e-6)
            d = np.sqrt((rho * xi_s * phi * 1j - kappa)**2 + xi_s**2 * (phi * 1j + phi**2))
            g = (kappa - rho * xi_s * phi * 1j - d) / (kappa - rho * xi_s * phi * 1j + d)
            e_neg_dT = np.exp(-d * T_mat)
            C = (1/xi_s**2) * ((1 - e_neg_dT) / (1 - g * e_neg_dT)) * (kappa - rho * xi_s * phi * 1j - d)
            # Log Trap Fix
            D = (kappa * theta / xi_s**2) * ((kappa - rho * xi_s * phi * 1j - d) * T_mat - 
                2 * np.log((1 - g * e_neg_dT) / (1 - g + 1e-15)))
            
            # Bates Jumps
            k_bar = np.exp(mu_j + 0.5 * sigma_j**2) - 1
            e_i_phi_J = np.exp(1j * phi * mu_j - 0.5 * sigma_j**2 * phi**2)
            jump_part = lamb * T_mat * (e_i_phi_J - 1 - 1j * phi * k_bar)
            return np.exp(C * v0 + D + 1j * phi * np.log(F_mat) + jump_part)

        cf_p1, cf_p2 = get_cf(u - 1j), get_cf(u)
        
        # Integrands
        int_p1 = np.real((np.exp(-1j * u * np.log(K_mat)) * cf_p1) / (1j * u * F_mat))
        int_p2 = np.real((np.exp(-1j * u * np.log(K_mat)) * cf_p2) / (1j * u))

        # 2. SIMPSON'S RULE WEIGHTS (Much more accurate for steep slopes)
        weights = np.ones(N_grid + 1)
        weights[1:-1:2] = 4
        weights[2:-2:2] = 2
        weights = (weights * (u_max / N_grid) / 3.0)[:, np.newaxis]

        # Integration
        is_otm_call = (K_mat > F_mat)
        P1_sum = np.sum(int_p1 * weights, axis=0)
        P2_sum = np.sum(int_p2 * weights, axis=0)
        
        P1 = np.where(is_otm_call, 0.5 + (1/np.pi) * P1_sum, 0.5 - (1/np.pi) * P1_sum)
        P2 = np.where(is_otm_call, 0.5 + (1/np.pi) * P2_sum, 0.5 - (1/np.pi) * P2_sum)
        
        price_otm = np.where(is_otm_call,
                             S0 * np.exp(-q_mat * T_mat) * P1 - K_mat * np.exp(-r_mat * T_mat) * P2,
                             K_mat * np.exp(-r_mat * T_mat) * P2 - S0 * np.exp(-q_mat * T_mat) * P1)
        
        # requested type conversion logic (unchanged)
        res = price_otm.flatten()
        is_put_req = (types == "PUT")
        is_itm_req = (is_put_req != (~is_otm_call.flatten()))
        if np.any(is_itm_req):
            adj = S0 * np.exp(-q * T) - K * np.exp(-r * T)
            res = np.where(is_put_req, np.where(~is_otm_call.flatten(), res, res - adj),
                                       np.where(is_otm_call.flatten(), res, res + adj))
        return np.nan_to_num(np.maximum(res, 0.0))
    
    @staticmethod
    def price_european_call(S0, K, T, r, q, kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j):
        return float(BatesAnalyticalPricer.price_european_call_vectorized(
            S0, K, T, r, q, kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j)[0])

    @staticmethod
    def price_european_put(S0, K, T, r, q, kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j):
        call = BatesAnalyticalPricer.price_european_call(
            S0, K, T, r, q, kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j)
        return call - S0 * np.exp(-q * T) + K * np.exp(-r * T)