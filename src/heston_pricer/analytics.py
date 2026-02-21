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
import numpy as np

class BatesAnalyticalPricer:
    @staticmethod
    def price_vectorized(S0, K, T, r, q, types, kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j, silent=False):
        """
        Analytical pricer for the Bates model. 
        Only prints if numerical integrity is compromised.
        """
        #if not silent:
            # Heartbeat to confirm engine is active
            #print(f"--- Bates Engine Active: xi={xi:.4f}, v0={v0:.4f}, lamb={lamb:.4f} ---")

        # 1. INTEGRATION SETUP
        N_grid, u_limit =  296, 10000 #  8000, 10000.0 
        u_linear = np.linspace(0, 1, N_grid + 1)
        u = (u_linear**2 * u_limit)[:, np.newaxis]
        u[0] = 1e-12 
        du = np.diff(u, axis=0) 
        
        K, T, r, q = np.atleast_1d(K), np.atleast_1d(T), np.atleast_1d(r), np.atleast_1d(q)
        T_mat, r_mat, q_mat, K_mat = T[np.newaxis,:], r[np.newaxis,:], q[np.newaxis,:], K[np.newaxis,:]
        F_mat = S0 * np.exp((r_mat - q_mat) * T_mat)

        def get_cf(phi):
            xi_s = np.maximum(xi, 1e-6)
            
            # Complex component d (Matches LaTeX exactly)
            d = np.sqrt((kappa - rho * xi_s * phi * 1j)**2 + xi_s**2 * (phi * 1j + phi**2))
            
            if np.any(np.isnan(d)):
                print(f"CRITICAL: NaN in 'd' (Complex Square Root). Parameters likely out of bounds.")

            g = (kappa - rho * xi_s * phi * 1j - d) / (kappa - rho * xi_s * phi * 1j + d)
            e_neg_dT = np.exp(-d * T_mat)
            
            # D matches LaTeX D(i Phi, T) - Multiplies v0
            D = ((kappa - rho * xi_s * phi * 1j - d) / xi_s**2) * ((1 - e_neg_dT) / (1 - g * e_neg_dT))
            
            # C matches LaTeX C(i Phi, T) - Independent term including drift
            log_arg = (1 - g * e_neg_dT) / (1 - g + 1e-15)
            C = phi * 1j * (r_mat - q_mat) * T_mat + (kappa * theta / xi_s**2) * ((kappa - rho * xi_s * phi * 1j - d) * T_mat - 2 * np.log(log_arg))
            
            # Merton Jump Component
            k_bar = np.exp(mu_j + 0.5 * sigma_j**2) - 1
            jump_part = lamb * T_mat * (np.exp(1j * phi * mu_j - 0.5 * sigma_j**2 * phi**2) - 1 - 1j * phi * k_bar)
            
            # CF for X_T = log(S_T / S_0)
            cf_XT = np.exp(C + D * v0 + jump_part)
            
            if np.any(np.isnan(cf_XT)):
                 print(f"CRITICAL: NaN in Characteristic Function. Log or Division by Zero error.")
            
            return cf_XT

        # Evaluate CF for log-returns
        cf_p1, cf_p2 = get_cf(u - 1j), get_cf(u)
        
        # Integration mapping 1:1 to your LaTeX P1 and P2 formulas
        int_p1 = np.real((cf_p1 * np.exp(-1j * u * np.log(K_mat / S0) - (r_mat - q_mat) * T_mat)) / (1j * u))
        int_p2 = np.real((cf_p2 * np.exp(-1j * u * np.log(K_mat / S0))) / (1j * u))
        # 2. NUMERICAL INTEGRITY CHECKS
        tail_magnitude = np.max(np.abs(int_p2[-1, :]))
        if tail_magnitude > 1e-4:
            print(f"WARNING: Truncation Error {tail_magnitude:.2e}. Integral hasn't converged at u=10000.")

        is_otm_call = (K_mat > F_mat)
        P1_sum = np.sum(0.5 * (int_p1[:-1, :] + int_p1[1:, :]) * du, axis=0)
        P2_sum = np.sum(0.5 * (int_p2[:-1, :] + int_p2[1:, :]) * du, axis=0)
        
        P1_val = 0.5 + (1/np.pi) * P1_sum
        P2_val = 0.5 + (1/np.pi) * P2_sum

        if np.any((P1_val < -0.01) | (P1_val > 1.01)):
            print(f"WARNING: Probability Violation (P={np.mean(P1_val):.2f}). Integration is likely oscillating/aliasing.")

        # 3. PRICING LOGIC
        P1 = np.where(is_otm_call, P1_val, 1.0 - P1_val)
        P2 = np.where(is_otm_call, P2_val, 1.0 - P2_val)
        
        price_otm = np.where(is_otm_call,
                             S0 * np.exp(-q_mat * T_mat) * P1 - K_mat * np.exp(-r_mat * T_mat) * P2,
                             K_mat * np.exp(-r_mat * T_mat) * P2 - S0 * np.exp(-q_mat * T_mat) * P1)
        
        price_otm = np.maximum(price_otm.flatten(), 0.0)

        # 4. REQUESTED TYPE CONVERSION
        res = price_otm
        is_put_req = (np.array([t.upper() for t in types]) == "PUT")
        is_itm_req = (is_put_req != (~is_otm_call.flatten()))
        if np.any(is_itm_req):
            adj = S0 * np.exp(-q * T) - K * np.exp(-r * T)
            res = np.where(is_put_req, np.where(~is_otm_call.flatten(), res, res - adj),
                                       np.where(is_otm_call.flatten(), res, res + adj))
        
        return np.nan_to_num(np.maximum(res, 0.0))

    @staticmethod
    def price_european_call_vectorized(S0, K, T, r, q, kappa, theta, xi, rho, v0, 
                                       lamb, mu_j, sigma_j, silent=True):
        K = np.atleast_1d(K)
        types = np.array(["CALL"] * len(K))
        return BatesAnalyticalPricer.price_vectorized(
            S0, K, T, r, q, types, kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j, silent=silent
        )

    @staticmethod
    def price_european_call(S0, K, T, r, q, kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j, silent=True):
        res = BatesAnalyticalPricer.price_european_call_vectorized(
            S0, K, T, r, q, kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j, silent=silent)
        return float(res[0])

    @staticmethod
    def price_european_put(S0, K, T, r, q, kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j, silent=True):
        res = BatesAnalyticalPricer.price_vectorized(
            S0, np.atleast_1d(K), np.atleast_1d(T), np.atleast_1d(r), np.atleast_1d(q), 
            np.array(["PUT"]), kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j, silent=silent
        )
        return float(res[0])
    

import numpy.polynomial.legendre as leg

class BatesAnalyticalPricerFast:
    @staticmethod
    def price_vectorized(S0, K, T, r, q, types, kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j, silent=True):
        """
        Ultra-fast Analytical pricer for the Bates model.
        Uses Gauss-Legendre Quadrature and Maturity-based Characteristic Function caching.
        """
        K = np.atleast_1d(K)
        T = np.atleast_1d(T)
        r = np.atleast_1d(r)
        q = np.atleast_1d(q)
        types = np.atleast_1d(types)
        
        # 1. GAUSS-LEGENDRE QUADRATURE SETUP
        # 150 nodes is extremely accurate for smooth characteristic functions
        N_nodes = 126
        u_max = 300.0  # Truncation limit for the integral
        
        nodes, weights = leg.leggauss(N_nodes)
        # Map nodes from [-1, 1] to [0, u_max]
        u = 0.5 * u_max * (nodes + 1)
        du = 0.5 * u_max * weights
        
        # Open-quadrature naturally avoids u=0, but we add a safeguard
        u = np.maximum(u, 1e-12)
        
        # 2. GROUP BY MATURITY (CACHING ENGINE)
        # We only evaluate the Characteristic Function once per unique T.
        unique_T_indices = {}
        for i, t_val in enumerate(T):
            if t_val not in unique_T_indices:
                unique_T_indices[t_val] = []
            unique_T_indices[t_val].append(i)
            
        prices = np.zeros(len(K))
        xi_s = np.maximum(xi, 1e-6)
        k_bar = np.exp(mu_j + 0.5 * sigma_j**2) - 1
        
        # Characteristic Function Logic (Isolated for caching)
        def eval_cf(phi, t_val, r_val, q_val):
            d = np.sqrt((kappa - rho * xi_s * phi * 1j)**2 + xi_s**2 * (phi * 1j + phi**2))
            
            if not silent and np.any(np.isnan(d)):
                print("CRITICAL: NaN in 'd' (Complex Square Root).")

            g = (kappa - rho * xi_s * phi * 1j - d) / (kappa - rho * xi_s * phi * 1j + d)
            e_neg_dT = np.exp(-d * t_val)
            
            D = ((kappa - rho * xi_s * phi * 1j - d) / xi_s**2) * ((1 - e_neg_dT) / (1 - g * e_neg_dT))
            
            log_arg = (1 - g * e_neg_dT) / (1 - g + 1e-15)
            C = phi * 1j * (r_val - q_val) * t_val + (kappa * theta / xi_s**2) * ((kappa - rho * xi_s * phi * 1j - d) * t_val - 2 * np.log(log_arg))
            
            jump_part = lamb * t_val * (np.exp(1j * phi * mu_j - 0.5 * sigma_j**2 * phi**2) - 1 - 1j * phi * k_bar)
            
            return np.exp(C + D * v0 + jump_part)

        # 3. EVALUATE PER MATURITY AND BROADCAST ACROSS STRIKES
        for t_val, idxs in unique_T_indices.items():
            r_val = r[idxs[0]]
            q_val = q[idxs[0]]
            
            # THE MASSIVE SPEEDUP: CF evaluated exactly twice per maturity slice!
            cf_p1 = eval_cf(u - 1j, t_val, r_val, q_val)
            cf_p2 = eval_cf(u, t_val, r_val, q_val)
            
            K_slice = K[idxs]
            F_slice = S0 * np.exp((r_val - q_val) * t_val)
            is_otm_call = (K_slice > F_slice)
            
            # Reshape u arrays to broadcast against strikes (N_nodes, N_strikes)
            u_col = u[:, np.newaxis]
            du_col = du[:, np.newaxis]
            log_K_S = np.log(K_slice / S0)[np.newaxis, :]
            
            # P1 Integral (Vectorized over K)
            term_p1 = (cf_p1[:, np.newaxis] * np.exp(-1j * u_col * log_K_S - (r_val - q_val) * t_val)) / (1j * u_col)
            P1_sum = np.sum(np.real(term_p1) * du_col, axis=0)
            P1_val = 0.5 + (1 / np.pi) * P1_sum
            
            # P2 Integral (Vectorized over K)
            term_p2 = (cf_p2[:, np.newaxis] * np.exp(-1j * u_col * log_K_S)) / (1j * u_col)
            P2_sum = np.sum(np.real(term_p2) * du_col, axis=0)
            P2_val = 0.5 + (1 / np.pi) * P2_sum
            
            # Calculate Out-Of-The-Money base price
            P1 = np.where(is_otm_call, P1_val, 1.0 - P1_val)
            P2 = np.where(is_otm_call, P2_val, 1.0 - P2_val)
            
            price_otm = np.where(is_otm_call,
                                 S0 * np.exp(-q_val * t_val) * P1 - K_slice * np.exp(-r_val * t_val) * P2,
                                 K_slice * np.exp(-r_val * t_val) * P2 - S0 * np.exp(-q_val * t_val) * P1)
            
            price_otm = np.maximum(price_otm, 0.0)
            
            # Convert back to requested types via Put-Call Parity
            types_slice = types[idxs]
            is_put_req = np.array([t.upper() == "PUT" for t in types_slice])
            
            adj = S0 * np.exp(-q_val * t_val) - K_slice * np.exp(-r_val * t_val)
            res = np.copy(price_otm)
            
            res = np.where(is_put_req, 
                           np.where(~is_otm_call, res, res - adj),
                           np.where(is_otm_call, res, res + adj))
                           
            prices[idxs] = res

        return np.nan_to_num(np.maximum(prices, 0.0))

    @staticmethod
    def price_european_call_vectorized(S0, K, T, r, q, kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j, silent=True):
        K = np.atleast_1d(K)
        types = np.array(["CALL"] * len(K))
        return BatesAnalyticalPricerFast.price_vectorized(
            S0, K, T, r, q, types, kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j, silent=silent
        )

    @staticmethod
    def price_european_call(S0, K, T, r, q, kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j, silent=True):
        res = BatesAnalyticalPricerFast.price_european_call_vectorized(
            S0, K, T, r, q, kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j, silent=silent)
        return float(res[0])

    @staticmethod
    def price_european_put(S0, K, T, r, q, kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j, silent=True):
        res = BatesAnalyticalPricerFast.price_vectorized(
            S0, np.atleast_1d(K), np.atleast_1d(T), np.atleast_1d(r), np.atleast_1d(q), 
            np.array(["PUT"]), kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j, silent=silent
        )
        return float(res[0])
