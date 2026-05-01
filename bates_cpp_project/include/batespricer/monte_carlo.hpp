#pragma once

#include <vector>
#include "types.hpp"

namespace batespricer {

// ─────────────────────────────────────────────────────────────
// Monte Carlo configuration
// ─────────────────────────────────────────────────────────────
struct MCConfig {
    int      n_paths    = 100000;
    int      n_steps    = 252;
    unsigned seed       = 42;
    bool     antithetic = true;
};

// ─────────────────────────────────────────────────────────────
// Full-Truncation Euler scheme  (Lord et al. 2010)
//   Returns matrix [n_paths × (n_steps+1)] of spot prices
// ─────────────────────────────────────────────────────────────
std::vector<std::vector<double>> generate_bates_paths_ft(
    double S0, double r, double q,
    const BatesParams& p, double T,
    const MCConfig& cfg);

// ─────────────────────────────────────────────────────────────
// Quadratic-Exponential scheme  (Andersen 2008)
//   Returns terminal spot at each maturity slice [n_paths × n_mats]
// ─────────────────────────────────────────────────────────────
std::vector<std::vector<double>> generate_bates_qe_slices(
    double S0, const BatesParams& p,
    double dt, int total_steps,
    const std::vector<int>& maturity_step_indices,
    const MCConfig& cfg);

// ─────────────────────────────────────────────────────────────
// Price a set of vanilla options via Monte Carlo
// ─────────────────────────────────────────────────────────────
std::vector<double> price_mc(
    double S0,
    const std::vector<MarketOption>& options,
    const BatesParams& p,
    const MCConfig& cfg);

} // namespace batespricer
