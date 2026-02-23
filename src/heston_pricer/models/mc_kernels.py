import numpy as np
from numba import jit
import math
# --- KERNELS ---

@jit(nopython=True, cache=True, fastmath=True)
def generate_paths_kernel(S0: float, r: float, q: float, sigma: float, 
                          T: float, n_paths: int, n_steps: int) -> np.ndarray:
    """
    Classical GBM Black-Scholes kernel. Includes Antithetic Sampling.
    """
    dt = T / n_steps
    half_paths = n_paths // 2
    
    # Variance reduction: Antithetic variates [Z, -Z]
    Z = np.concatenate((np.random.standard_normal((half_paths, n_steps)), 
                        -np.random.standard_normal((half_paths, n_steps))), axis=0)
    
    drift = (r - q - 0.5 * sigma**2) * dt
    diff = sigma * np.sqrt(dt)
    
    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0
    
    for i in range(n_paths):
        log_ret = 0.0
        for j in range(n_steps):
            log_ret += drift + diff * Z[i, j]
            prices[i, j + 1] = S0 * np.exp(log_ret)
            
    return prices

@jit(nopython=True, cache=True, fastmath=True)
def generate_heston_paths(S0, r, q, v0, kappa, theta, xi, rho, T, n_paths, n_steps):
    """
    Heston Model simulation using a truncated Euler discretization.
    FIXED: Ensures Asset update uses v_t (not v_{t+1}) to align with Ito calculus.
    """
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    c1, c2 = rho, np.sqrt(1 - rho**2)
    
    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0
    
    curr_v = np.full(n_paths, v0)
    curr_s = np.full(n_paths, S0, dtype=np.float64)
    
    for j in range(n_steps):
        Z1 = np.random.standard_normal(n_paths)
        Z2 = np.random.standard_normal(n_paths)
        Zv = c1 * Z1 + c2 * Z2
        
        # CORRECTED: Capture v_t before updating to v_{t+1}
        v_t = np.maximum(curr_v, 0.0)
        
        # Update Asset using v_t
        curr_s *= np.exp((r - q - 0.5 * v_t) * dt + np.sqrt(v_t) * sqrt_dt * Z1)

        # Update Variance using v_t (Euler)
        curr_v += kappa * (theta - v_t) * dt + xi * np.sqrt(v_t) * sqrt_dt * Zv
        
        prices[:, j + 1] = curr_s
        
    return prices

@jit(nopython=True, cache=True, fastmath=True)
def generate_heston_paths_crn(S0, r, q, v0, kappa, theta, xi, rho, T, n_paths, n_steps, noise_matrix):
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    c1, c2 = rho, np.sqrt(1 - rho**2)
    
    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0
    curr_v = np.full(n_paths, v0)
    curr_s = np.full(n_paths, S0)
    
    for j in range(n_steps):
        Z1 = noise_matrix[0, j]
        Z2 = noise_matrix[1, j]
        Zv = c1 * Z1 + c2 * Z2
        
        # --- FIX: Capture v_t BEFORE updating ---
        v_t = np.maximum(curr_v, 0.0) 
        
        # Update Asset using v_t (Standard Euler)
        # drift = (r - q - 0.5 * v_t) * dt
        # diffusion = sqrt(v_t) * sqrt(dt) * Z1
        curr_s *= np.exp((r - q - 0.5 * v_t) * dt + np.sqrt(v_t) * sqrt_dt * Z1)
        
        # Update Variance using v_t
        curr_v += kappa * (theta - v_t) * dt + xi * np.sqrt(v_t) * sqrt_dt * Zv
        
        prices[:, j + 1] = curr_s
        
    return prices

@jit(nopython=True, cache=True, fastmath=True)
def generate_bates_paths(S0, r, q, v0, kappa, theta, xi, rho, lamb, mu_j, sigma_j, T, n_paths, n_steps):
    """
    Bates Model (Heston + Merton Jump Diffusion) simulation.
    Uses Poisson-Gaussian jumps.
    """
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    c1, c2 = rho, np.sqrt(1 - rho**2)
    
    # Expected relative jump size (compensator for drift)
    k = np.exp(mu_j + 0.5 * sigma_j**2) - 1
    drift_correction = lamb * k
    
    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0
    
    curr_v = np.full(n_paths, v0)
    curr_s = np.full(n_paths, S0)
    
    for j in range(n_steps):
        Z1 = np.random.standard_normal(n_paths)
        Z2 = np.random.standard_normal(n_paths)
        Zv = c1 * Z1 + c2 * Z2
        
        # --- JUMP COMPONENT ---
        # Number of jumps in this step (Poisson)
        # Note: For small lambda*dt, this is mostly 0 or 1.
        N_jumps = np.random.poisson(lamb * dt, n_paths)
        
        # Total Log-Jump size: Sum of N normals -> N(N*mu, N*sigma^2)
        # We simulate this by scaling a single standard normal by sqrt(N)
        Z_jump = np.random.standard_normal(n_paths)
        
        # Avoid sqrt(0) warnings/errors implicitly by multiplication (or masking if strictly needed)
        # N_jumps is integer, so we cast to float for sqrt
        jump_magnitude = N_jumps * mu_j + np.sqrt(N_jumps) * sigma_j * Z_jump
        
        # --- HESTON UPDATE ---
        v_t = np.maximum(curr_v, 0.0)
        
        # Asset Update:
        # drift = r - q - 0.5*v_t - lambda*k
        drift_term = (r - q - 0.5 * v_t - drift_correction) * dt
        diff_term = np.sqrt(v_t) * sqrt_dt * Z1
        
        curr_s *= np.exp(drift_term + diff_term + jump_magnitude)
        
        # Variance Update (Standard Heston)
        curr_v += kappa * (theta - v_t) * dt + xi * np.sqrt(v_t) * sqrt_dt * Zv
        
        prices[:, j + 1] = curr_s
        
    return prices
@jit(nopython=True, cache=True, fastmath=True)
def generate_bates_paths_crn(S0, r, q, v0, kappa, theta, xi, rho, lamb, mu_j, sigma_j, T, n_paths, n_steps, noise_matrix):
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    c1, c2 = rho, np.sqrt(1 - rho**2)
    
    # Compensator for Log-Normal Jumps: E[e^J] - 1
    # This MUST match the jump magnitude distribution below exactly
    k_bar = np.exp(mu_j + 0.5 * sigma_j**2) - 1
    drift_correction = lamb * k_bar
    
    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0
    
    curr_v = np.full(n_paths, v0, dtype=np.float64)
    curr_s = np.full(n_paths, S0, dtype=np.float64)
    
    lam_dt = lamb * dt
    exp_neg_lam = np.exp(-lam_dt)
    
    for j in range(n_steps):
        Z1_row = noise_matrix[0, j, :n_paths] 
        Z2_row = noise_matrix[1, j, :n_paths]
        U_jump_row = noise_matrix[2, j, :n_paths] 
        Z_size_row = noise_matrix[3, j, :n_paths]
        
        # 1. Update Variance (Standard Euler)
        v_t = np.maximum(curr_v, 0.0)
        Zv_row = c1 * Z1_row + c2 * Z2_row
        curr_v += kappa * (theta - v_t) * dt + xi * np.sqrt(v_t) * sqrt_dt * Zv_row
        
        # 2. Update Asset (Log-Euler)
        # drift = r - q - 0.5*v - lambda*k
        drift_base = (r - q - 0.5 * v_t - drift_correction) * dt 
        diffusion = np.sqrt(v_t) * sqrt_dt * Z1_row
        
        # 3. Jump Process
        for i in range(n_paths):
            # Inverse Transform Sampling for Poisson
            u = U_jump_row[i]
            n_jumps = 0
            p_pois = exp_neg_lam
            s_pois = p_pois
            
            # This loop is unbiased
            while u > s_pois and n_jumps < 10:
                n_jumps += 1
                p_pois *= lam_dt / n_jumps
                s_pois += p_pois
            
            jump_mag = 0.0
            if n_jumps > 0:
                # Sum of N normals: N(N*mu, N*sigma^2)
                # Correctly scaled by sqrt(N)
                jump_mag = n_jumps * mu_j + np.sqrt(float(n_jumps)) * sigma_j * Z_size_row[i]
            
            # Exponentiate
            curr_s[i] *= np.exp(drift_base[i] + diffusion[i] + jump_mag)
        
        prices[:, j + 1] = curr_s
        
    return prices
@jit(nopython=True, cache=True, fastmath=True)
def generate_bates_qe_slices_crn(S0, v0, kappa, theta, xi, rho, lamb, mu_j, sigma_j, 
                                 dt, n_paths, n_steps, maturity_step_idxs, noise_matrix):
    n_maturities = len(maturity_step_idxs)
    prices_at_maturities = np.zeros((n_paths, n_maturities), dtype=np.float64)
    
    k_bar = math.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    drift_base = -lamb * k_bar * dt
    
    exp_k = math.exp(-kappa * dt)
    inv_kappa = 1.0 / kappa
    c1 = xi**2 * exp_k * inv_kappa * (1.0 - exp_k)
    c2 = theta * xi**2 * 0.5 * inv_kappa * (1.0 - exp_k)**2
    
    rho_inv_xi = rho / xi
    sqrt_1_min_rho2 = math.sqrt(1.0 - rho**2)
    
    curr_v = np.full(n_paths, v0, dtype=np.float64)
    curr_log_s = np.full(n_paths, math.log(S0), dtype=np.float64)
    
    lam_dt = lamb * dt
    exp_neg_lam = math.exp(-lam_dt)
    
    next_mat_idx = 0
    target_step = maturity_step_idxs[next_mat_idx]
    
    for j in range(n_steps):
        Z_v_row = noise_matrix[0, j, :]
        Z_x_row = noise_matrix[1, j, :]
        Z_jump_row = noise_matrix[2, j, :]
        U_v_row = noise_matrix[3, j, :]
        U_jump_row = noise_matrix[4, j, :]
        
        for i in range(n_paths):
            vt = curr_v[i]
            
            # --- 1. VARIANCE UPDATE (QE) ---
            m = theta + (vt - theta) * exp_k
            s2 = vt * c1 + c2
            psi = s2 / (m**2)
            v_next = 0.0
            
            if psi <= 1.5:
                b2 = 2.0 / psi - 1.0 + math.sqrt(2.0 / psi * (2.0 / psi - 1.0))
                a = m / (1.0 + b2)
                b = math.sqrt(b2)
                v_next = a * (b + Z_v_row[i])**2
            else:
                p = (psi - 1.0) / (psi + 1.0)
                beta = (1.0 - p) / m
                u = U_v_row[i]
                if u > p:
                    v_next = math.log((1.0 - p) / (1.0 - u)) / beta
                else:
                    v_next = 0.0
            curr_v[i] = v_next
            
            # --- 2. ASSET UPDATE ---
            V_int = 0.5 * (vt + v_next) * dt 
            log_s_drift = drift_base - 0.5 * V_int + rho_inv_xi * (v_next - vt - kappa * theta * dt + kappa * V_int)
            log_s_diff = sqrt_1_min_rho2 * math.sqrt(V_int) * Z_x_row[i]
            
            # --- 3. JUMPS ---
            u_j = U_jump_row[i]
            n_jumps = 0
            p_pois = exp_neg_lam
            s_pois = p_pois
            
            while u_j > s_pois and n_jumps < 10:
                n_jumps += 1
                p_pois *= lam_dt / n_jumps
                s_pois += p_pois
                
            jump_mag = 0.0
            if n_jumps > 0:
                jump_mag = n_jumps * mu_j + math.sqrt(float(n_jumps)) * sigma_j * Z_jump_row[i]
            
            curr_log_s[i] += log_s_drift + log_s_diff + jump_mag

        # --- 4. MATURITY SLICING (FIXED WITH WHILE LOOP) ---
        current_step = j + 1
        
        # This will exhaust all maturities that fall on this exact time step
        while next_mat_idx < n_maturities and current_step == target_step:
            for i in range(n_paths):
                prices_at_maturities[i, next_mat_idx] = math.exp(curr_log_s[i])
            
            next_mat_idx += 1
            if next_mat_idx < n_maturities:
                target_step = maturity_step_idxs[next_mat_idx]

    return prices_at_maturities