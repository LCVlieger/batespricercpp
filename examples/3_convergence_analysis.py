import time
import numpy as np
import pandas as pd
import math
from heston_pricer.models.mc_kernels import generate_heston_paths
from heston_pricer.analytics import HestonAnalyticalPricer

def heston_pure_python(S0, r, q, v0, kappa, theta, xi, rho, T, n_paths, n_steps):
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    c1, c2 = rho, math.sqrt(1 - rho**2)
    
    final_prices = []
    for _ in range(n_paths):
        s_t, v_t = S0, v0
        for _ in range(n_steps):
            z1, z2 = np.random.normal(), np.random.normal()
            zv = c1 * z1 + c2 * z2
            
            v_pos = max(v_t, 0.0)
            dv = kappa * (theta - v_pos) * dt + xi * math.sqrt(v_pos) * sqrt_dt * zv
            v_t += dv
            
            drift = (r - q - 0.5 * v_pos) * dt
            diffusion = math.sqrt(v_pos) * sqrt_dt * z1
            s_t *= math.exp(drift + diffusion)
        final_prices.append(s_t)
        
    return np.array(final_prices)

def heston_numpy_vectorized(S0, r, q, v0, kappa, theta, xi, rho, T, n_paths, n_steps):
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    
    v_t = np.full(n_paths, v0)
    s_t = np.full(n_paths, S0)
    
    Z1 = np.random.normal(size=(n_steps, n_paths))
    Zv = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(size=(n_steps, n_paths))
    
    for t in range(n_steps):
        v_pos = np.maximum(v_t, 0.0)
        sq_v = np.sqrt(v_pos)
        
        dv = kappa * (theta - v_pos) * dt + xi * sq_v * sqrt_dt * Zv[t]
        v_t += dv
        s_t *= np.exp((r - q - 0.5 * v_pos) * dt + sq_v * sqrt_dt * Z1[t])
        
    return s_t

def main():
    S0, r, q, T = 100.0, 0.05, 0.0, 1.0
    v0, kappa, theta, xi, rho = 0.04, 1.0, 0.04, 0.5, -0.7
    N, M = 2_000_000, 252
    
    print(f"[{pd.Timestamp.now().time()}] Benchmarking Kernels (N={N}, Steps={M})")

    # 1. Python (Estimated)
    t0 = time.time()
    heston_pure_python(S0, r, q, v0, kappa, theta, xi, rho, T, 50_000, M)
    t_py = (time.time() - t0) * (N / 50_000)

    # 2. NumPy Vectorized
    t0 = time.time()
    heston_numpy_vectorized(S0, r, q, v0, kappa, theta, xi, rho, T, N, M)
    t_np = time.time() - t0

    # 3. Numba JIT
    generate_heston_paths(S0, r, q, v0, kappa, theta, xi, rho, T, 10, M) # Warmup
    t0 = time.time()
    generate_heston_paths(S0, r, q, v0, kappa, theta, xi, rho, T, N, M)
    t_numba = time.time() - t0

    results = [
        {"Engine": "Python (Est)", "Time": t_py, "Rel": 1.0},
        {"Engine": "NumPy", "Time": t_np, "Rel": t_py/t_np},
        {"Engine": "Numba", "Time": t_numba, "Rel": t_py/t_numba}
    ]
    print(pd.DataFrame(results).set_index("Engine").to_string(float_format="{:.2f}".format))

    # Validation
    p_ana = HestonAnalyticalPricer.price_european_call(S0, 100, T, r, q, kappa, theta, xi, rho, v0)
    paths = generate_heston_paths(S0, r, q, v0, kappa, theta, xi, rho, T, 200_000, 100)
    p_mc = np.mean(np.maximum(paths[:, -1] - 100, 0)) * np.exp(-r*T)
    
    print(f"\nConvergence Check: |Ana - MC| = {abs(p_ana - p_mc):.4f}")

if __name__ == "__main__":
    main()