import numpy as np
from scipy.optimize import minimize, brentq
from scipy.stats import norm
from dataclasses import dataclass
from typing import List, Dict
from scipy.interpolate import interp1d
from collections import defaultdict

# --- DATA STRUCTURES ---
@dataclass
class MarketOption:
    strike: float
    maturity: float
    market_price: float
    option_type: str = "CALL" 

class SimpleYieldCurve:
    def __init__(self, tenors: List[float], rates: List[float]):
        self.tenors = tenors
        self.rates = rates
        if len(tenors) == 1:
            self.curve = lambda t: rates[0]
        else:
            self.curve = interp1d(tenors, rates, kind='linear', fill_value="extrapolate")

    def get_rate(self, T: float) -> float:
        if T < 1e-5: return float(self.rates[0]) if self.tenors else 0.0
        return float(self.curve(T))

    def to_dict(self):
        return {"tenors": self.tenors, "rates": self.rates}

class HestonCalibrator:
    def __init__(self, S0: float, r_curve, q_curve):
        self.S0 = S0
        self.r_curve = r_curve
        self.q_curve = q_curve

    def calibrate(self, options: List, init_guess: List[float] = None) -> Dict:
        # 1. PREPARE DATA VECTORS
        strikes = np.array([opt.strike for opt in options])
        maturities = np.array([opt.maturity for opt in options])
        market_prices = np.array([opt.market_price for opt in options])
        
        # Boolean mask for Puts
        is_put = np.array([opt.option_type == "PUT" for opt in options], dtype=bool)

        # Pre-calculate rates for every option
        r_vec = np.array([self.r_curve.get_rate(t) for t in maturities])
        q_vec = np.array([self.q_curve.get_rate(t) for t in maturities])

        # Bounds: kappa, theta, xi, rho, v0
        # Note: xi (vol-of-vol) replaces sigma
        bounds = [(0.01, 15.0), (0.001, 1.0), (0.01, 5.0), (-0.999, 0.999), (0.001, 1.0)]
        x0 = init_guess if init_guess else [2.0, 0.04, 0.5, -0.7, 0.04]

        # 2. VECTORIZED OBJECTIVE FUNCTION
        def objective(params):
            kappa, theta, xi, rho, v0 = params
            
            # A. Calculate CALL prices for everyone first
            model_calls = _heston_price_vectorized(
                self.S0, strikes, maturities, r_vec, q_vec,
                kappa, theta, xi, rho, v0
            )

            # B. Apply Put-Call Parity where needed
            if np.any(is_put):
                put_prices = (
                    model_calls[is_put] 
                    - self.S0 * np.exp(-q_vec[is_put] * maturities[is_put]) 
                    + strikes[is_put] * np.exp(-r_vec[is_put] * maturities[is_put])
                )
                model_final = model_calls.copy()
                model_final[is_put] = put_prices
            else:
                model_final = model_calls

            # C. Minimization Metric (MSE)
            mse = np.mean((model_final - market_prices)**2)
            return mse

        def callback(xk):
             print(f"   [Analytical] k={xk[0]:.2f}, theta={xk[1]:.3f}, xi={xk[2]:.2f}, rho={xk[3]:.2f}, v0={xk[4]:.3f}", flush=True)

        print("Starting Vectorized Price Minimization (Albrecher Stable Form)...")
        result = minimize(
            objective, x0, method='SLSQP', bounds=bounds, callback=callback, tol=1e-9
        )

        return {
            "kappa": float(result.x[0]), 
            "theta": float(result.x[1]), 
            "xi": float(result.x[2]),
            "rho": float(result.x[3]), 
            "v0": float(result.x[4]),
            "success": bool(result.success),
            "fun": float(result.fun)
        }

# --- HELPER: VECTORIZED ALBRECHER INTEGRATION ---
def _heston_price_vectorized(S0, K, T, r, q, kappa, theta, xi, rho, v0):
    """
    Vectorized implementation of Albrecher (2007) stable forms.
    Uses broadcasting to calculate prices for all options in parallel.
    """
    # 1. Integration Grid setup
    # We use a fixed grid for vectorization speed, but fine enough for accuracy.
    # Shape: (Grid_Size,)
    u_max = 200.0  # Sufficient for most maturities
    n_grid = 128   # 128 points usually balances speed/accuracy for calibration
    u = np.linspace(1e-5, u_max, n_grid)
    du = u[1] - u[0]

    # 2. Reshape Inputs for Broadcasting
    # u becomes (Grid, 1)
    # Option params become (1, Num_Options)
    u_vec = u[:, np.newaxis]       # Shape: (N, 1)
    K_vec = K[np.newaxis, :]       # Shape: (1, M)
    T_vec = T[np.newaxis, :]       # Shape: (1, M)
    r_vec = r[np.newaxis, :]       # Shape: (1, M)
    q_vec = q[np.newaxis, :]       # Shape: (1, M)

    # 3. Define the Characteristic Function Logic (Vectorized)
    def compute_integrand(is_P1):
        # If P1, evaluate at (u - i), effectively. 
        # We adapt the helper logic to vectorized arrays.
        
        # Adjust u for the specific probability measure
        # P1 corresponds to evaluating phi(u - i)
        # P2 corresponds to evaluating phi(u)
        u_calc = u_vec - 1j if is_P1 else u_vec
        
        # Albrecher Stable d
        # d = sqrt((rho*xi*u*i - kappa)^2 + xi^2(u*i + u^2))
        # Note: u_calc is complex for P1, so we handle the algebra carefully.
        # The term (u * 1j + u**2) matches the user's snippet logic.
        
        d = np.sqrt((rho * xi * u_calc * 1j - kappa)**2 + xi**2 * (u_calc * 1j + u_calc**2))
        
        # Stable g
        g = (kappa - rho * xi * u_calc * 1j - d) / (kappa - rho * xi * u_calc * 1j + d)
        
        # Characteristic Function Exponents
        # C term
        C = (1/xi**2) * (1 - np.exp(-d * T_vec)) / (1 - g * np.exp(-d * T_vec)) * \
            (kappa - rho * xi * u_calc * 1j - d)
            
        # D term (Split Logarithm for stability)
        val_num = 1 - g * np.exp(-d * T_vec)
        val_denom = 1 - g
        
        D = (kappa * theta / xi**2) * \
            ((kappa - rho * xi * u_calc * 1j - d) * T_vec - 2 * (np.log(val_num) - np.log(val_denom)))
            
        # Drift Term
        drift_term = 1j * u_calc * np.log(S0 * np.exp((r_vec - q_vec) * T_vec))
        
        # Full Characteristic Function value
        phi = np.exp(C * v0 + D + drift_term)
        
        # Final Integrand assembly
        # integrand = Re[ exp(-i*u*ln(K)) * phi / (i*u) ]
        # Note: We divide by (i * u_vec) from the grid, NOT u_calc.
        # For P1, the snippet divides by (i*u * S0*exp(r-q)T).
        
        numerator = np.exp(-1j * u_vec * np.log(K_vec)) * phi
        
        if is_P1:
             denominator = 1j * u_vec * S0 * np.exp((r_vec - q_vec) * T_vec)
        else:
             denominator = 1j * u_vec
             
        return np.real(numerator / denominator)

    # 4. Perform Vectorized Integration (Trapezoidal Rule)
    integrand_1 = compute_integrand(is_P1=True)
    integrand_2 = compute_integrand(is_P1=False)

    # Sum along axis 0 (the grid axis)
    P1 = 0.5 + (1 / np.pi) * np.trapz(integrand_1, x=u, axis=0)
    P2 = 0.5 + (1 / np.pi) * np.trapz(integrand_2, x=u, axis=0)

    # 5. Final Call Price
    call_price = (S0 * np.exp(-q * T) * P1) - (K * np.exp(-r * T) * P2)
    
    # Ensure non-negative (numerical noise can cause -1e-16)
    return np.maximum(call_price, 0.0)

# --- UTILS (Keep these for validation in main) ---
def implied_volatility(price: float, S: float, K: float, T: float, r: float, q: float, option_type: str = "CALL") -> float:
    if price <= 0: return 0.0
    if option_type == "PUT":
        intrinsic = max(K * np.exp(-r*T) - S * np.exp(-q*T), 0)
    else:
        intrinsic = max(S * np.exp(-q*T) - K * np.exp(-r*T), 0)
    if price < intrinsic: return 0.0

    def bs_price(sigma):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "PUT":
             val = (K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1))
        else:
             val = (S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        return val - price
    
    try:
        return brentq(bs_price, 0.001, 5.0)
    except:
        return 0.0

# --- MONTE CARLO CALIBRATOR ---
class HestonCalibratorMC:
    def __init__(self, S0: float, r_curve: SimpleYieldCurve, q_curve: SimpleYieldCurve, n_paths: int = 30000, n_steps: int = 100):
        self.S0 = S0
        self.r_curve = r_curve
        self.q_curve = q_curve # Now a Curve object
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.z_noise = None 
        
        # Batching containers
        self.maturity_batches = defaultdict(list)
        self.max_T = 0.0
        self.dt = 0.0

    def _precompute_batches(self, options: List[MarketOption]):
        """Organizes options by maturity to handle term structures."""
        self.maturity_batches.clear()
        if not options: return
        self.max_T = max(opt.maturity for opt in options)
        self.dt = self.max_T / self.n_steps
        
        # 1. Group options
        for opt in options:
            self.maturity_batches[opt.maturity].append(opt)
            
        # 2. Pre-generate ONE giant noise block (Brownian Bridge consistency)
        if self.z_noise is None:
            np.random.seed(42) 
            self.z_noise = np.random.normal(0, 1, (2, self.n_steps, self.n_paths))

    def get_prices(self, params: List[float]) -> Dict[float, List[float]]:
        """Returns map: {maturity: [price_opt1, price_opt2...]}"""
        kappa, theta, xi, rho, v0 = params
        
        results = {}
        
        # Loop over unique maturities
        for T_target, opts in self.maturity_batches.items():
            
            # A. Get rate/div for this specific maturity
            r_T = self.r_curve.get_rate(T_target)
            q_T = self.q_curve.get_rate(T_target)
            
            # B. Determine steps needed for this T
            steps_needed = int(round(T_target / self.dt))
            if steps_needed < 1: steps_needed = 1
            if steps_needed > self.n_steps: steps_needed = self.n_steps
            
            # C. Setup Environment & Process
            env = MarketEnvironment(self.S0, r_T, q_T)
            process = HestonProcess(env)
            process.market.kappa = kappa
            process.market.theta = theta
            process.market.xi = xi
            process.market.rho = rho
            process.market.v0 = v0
            
            # D. Simulation
            # Slice noise to match time-steps
            noise_slice = self.z_noise[:, :steps_needed, :]
            paths = process.generate_paths(
                T=T_target, n_paths=self.n_paths, n_steps=steps_needed, noise=noise_slice
            )
            S_final = paths[:, -1]
            
            # E. Pricing
            prices = []
            for opt in opts:
                if opt.option_type == "PUT":
                    payoff = np.maximum(opt.strike - S_final, 0.0)
                else:
                    payoff = np.maximum(S_final - opt.strike, 0.0)
                
                # Discount using the specific rate r_T
                price = np.mean(payoff) * np.exp(-r_T * T_target)
                prices.append(price)
                
            results[T_target] = prices
            
        return results

    def objective(self, params):
        kappa, theta, xi, rho, v0 = params
        
        penalty = 0.0
        if 2 * kappa * theta < xi**2:
            penalty += 0.0 * ((xi**2 - 2 * kappa * theta)**2)

        # Get prices for all maturities
        model_prices_map = self.get_prices(params)
        
        total_error = 0.0
        
        # Match Analytical Weighting Logic
        for T, opts in self.maturity_batches.items():
            m_prices = model_prices_map[T]
            
            for i, opt in enumerate(opts):
                model_p = m_prices[i]
                
                moneyness = np.log(opt.strike / self.S0)
                wing_weight = 1.0 + 5.0 * (moneyness**2)
                
                relative_error = (model_p - opt.market_price) / (opt.market_price + 1e-8)
                total_error += wing_weight * (relative_error**2)

        return total_error + penalty

    def calibrate(self, options: List[MarketOption], init_guess: List[float] = None) -> Dict:
        self._precompute_batches(options)
        x0 = init_guess if init_guess else [2.0, 0.05, 0.3, -0.7, 0.04]
        bounds = [(0.1, 10.0), (0.001, 2.0), (0.01, 5.0), (-0.999, 0.0), (0.001, 2.0)]
        
        def callback(xk):
             print(f"   [MonteCarlo] k={xk[0]:.2f}, theta={xk[1]:.3f}, xi={xk[2]:.2f}, rho={xk[3]:.2f}, v0={xk[4]:.3f}", flush=True)
        
        result = minimize(
            self.objective, x0, method='L-BFGS-B', bounds=bounds, 
            callback=callback, 
            tol=1e-5, options={'ftol': 1e-5, 'eps': 1e-5, 'maxiter': 50}
        )

        # Final IV Statistics
        final_map = self.get_prices(result.x)
        sse_iv, count = 0.0, 0
        
        for T, opts in self.maturity_batches.items():
            r_T = self.r_curve.get_rate(T)
            q_T = self.q_curve.get_rate(T) # Look up Q for this T
            m_prices = final_map[T]
            for i, opt in enumerate(opts):
                iv_mkt = implied_volatility(opt.market_price, self.S0, opt.strike, opt.maturity, r_T, q_T, opt.option_type)
                iv_model = implied_volatility(m_prices[i], self.S0, opt.strike, opt.maturity, r_T, q_T, opt.option_type)
                
                if iv_mkt > 0 and iv_model > 0:
                    sse_iv += (iv_model - iv_mkt) ** 2
                    count += 1
        
        return {
            "kappa": result.x[0], "theta": result.x[1], "xi": result.x[2],
            "rho": result.x[3], "v0": result.x[4], 
            "success": result.success, 
            "fun": result.fun, 
            "rmse_iv": np.sqrt(sse_iv / count) if count > 0 else 0.0
        }