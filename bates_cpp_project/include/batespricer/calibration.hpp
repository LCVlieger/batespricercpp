#pragma once

#include <vector>
#include "types.hpp"

namespace batespricer {

// ─────────────────────────────────────────────────────────────
// Calibration configuration
// ─────────────────────────────────────────────────────────────
struct CalibrationConfig {
    double tol       = 1e-8;
    double eps       = 1e-4;    // finite-diff step for numerical gradient
    int    max_iter  = 500;
    double sigma_cap = 2.0;     // spread-weight capping (μ + σ_cap·σ)
    bool   verbose   = true;
};

// ─────────────────────────────────────────────────────────────
// Parameter bounds for the 8 Bates parameters
// ─────────────────────────────────────────────────────────────
struct ParamBounds {
    double kappa_lo = 0.1,    kappa_hi = 10.0;
    double theta_lo = 0.001,  theta_hi = 0.5;
    double xi_lo    = 0.01,   xi_hi    = 5.0;
    double rho_lo   = -0.99,  rho_hi   = 0.0;
    double v0_lo    = 0.001,  v0_hi    = 0.5;
    double lamb_lo  = 0.0,    lamb_hi  = 5.0;
    double mu_j_lo  = -0.5,   mu_j_hi  = 0.5;
    double sigma_j_lo = 0.01, sigma_j_hi = 0.5;
};

// ─────────────────────────────────────────────────────────────
// Spread-weighted L-BFGS-B calibration
//   Uses the fast Gauss-Legendre analytical pricer internally
// ─────────────────────────────────────────────────────────────
CalibrationResult calibrate(
    double S0,
    const std::vector<MarketOption>& options,
    const CalibrationConfig& cfg = {},
    const ParamBounds& bounds = {});

} // namespace batespricer
