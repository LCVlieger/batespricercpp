import numpy as np
from scipy.stats import norm

class BlackScholesPricer:
    @staticmethod
    def price_european_call(S0, K, T, r, sigma):
        if T < 1e-6: return max(S0 - K, 0.0)
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
import numpy as np
from scipy.stats import norm

class HestonAnalyticalPricer:
    @staticmethod
    def price_european_call_vectorized(S0, K, T, r, q, kappa, theta, xi, rho, v0):
        K, T, r, q = np.atleast_1d(K), np.atleast_1d(T), np.atleast_1d(r), np.atleast_1d(q)
        N_grid, u_max = 128, 100.0
        du = u_max / N_grid
        u = np.linspace(1e-8, u_max, N_grid)[:, np.newaxis] 
        T_mat, r_mat, q_mat, K_mat = T[np.newaxis, :], r[np.newaxis, :], q[np.newaxis, :], K[np.newaxis, :]

        def get_cf(phi):
            d = np.sqrt((rho * xi * phi * 1j - kappa)**2 + xi**2 * (phi * 1j + phi**2))
            g = (kappa - rho * xi * phi * 1j - d) / (kappa - rho * xi * phi * 1j + d)
            exp_neg_dT = np.exp(-d * T_mat)
            C = (1/xi**2) * ((1 - exp_neg_dT) / (1 - g * exp_neg_dT)) * (kappa - rho * xi * phi * 1j - d)
            
            # Epsilon adds stability if g is exactly 1
            val_num, val_denom = 1 - g * exp_neg_dT, 1 - g + 1e-12
            D = (kappa * theta / xi**2) * ((kappa - rho * xi * phi * 1j - d) * T_mat - 2 * (np.log(val_num) - np.log(val_denom)))
            drift = 1j * phi * np.log(S0 * np.exp((r_mat - q_mat) * T_mat))
            return np.exp(C * v0 + D + drift)

        cf_p1, cf_p2 = get_cf(u - 1j), get_cf(u)
        int_p1 = np.real((np.exp(-1j * u * np.log(K_mat)) * cf_p1) / (1j * u * S0 * np.exp((r_mat - q_mat) * T_mat)))
        int_p2 = np.real((np.exp(-1j * u * np.log(K_mat)) * cf_p2) / (1j * u))
        P1 = 0.5 + (1/np.pi) * np.sum(int_p1 * du, axis=0)
        P2 = 0.5 + (1/np.pi) * np.sum(int_p2 * du, axis=0)
        price = S0 * np.exp(-q_mat * T_mat) * P1 - K_mat * np.exp(-r_mat * T_mat) * P2
        return np.maximum(price.flatten(), 0.0)

    @staticmethod
    def price_european_call(S0, K, T, r, q, kappa, theta, xi, rho, v0):
        return float(HestonAnalyticalPricer.price_european_call_vectorized(S0, K, T, r, q, kappa, theta, xi, rho, v0)[0])
    
    @staticmethod
    def price_european_put(S0, K, T, r, q, kappa, theta, xi, rho, v0):
        call = HestonAnalyticalPricer.price_european_call(S0, K, T, r, q, kappa, theta, xi, rho, v0)
        return call - S0 * np.exp(-q * T) + K * np.exp(-r * T)