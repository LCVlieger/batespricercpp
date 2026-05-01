#include "batespricer/monte_carlo.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

#ifdef BATESPRICER_USE_OPENMP
#include <omp.h>
#endif

namespace batespricer {

// ═════════════════════════════════════════════════════════════
// Poisson variate via inverse-CDF  (capped at 10 jumps)
// ═════════════════════════════════════════════════════════════

static int poisson_icdf(double lam_dt, double u) {
    double en_lam = std::exp(-lam_dt);
    int nj = 0;
    double pp = en_lam;
    double sp = pp;
    while (u > sp && nj < 10) {
        ++nj;
        pp *= lam_dt / nj;
        sp += pp;
    }
    return nj;
}

// ═════════════════════════════════════════════════════════════
// Full-Truncation Euler  (Lord et al. 2010)
// ═════════════════════════════════════════════════════════════

std::vector<std::vector<double>> generate_bates_paths_ft(
    double S0, double r, double q,
    const BatesParams& p, double T,
    const MCConfig& cfg)
{
    const int n_paths = cfg.n_paths;
    const int n_steps = cfg.n_steps;
    const double dt     = T / n_steps;
    const double sqrt_dt = std::sqrt(dt);
    const double c1 = p.rho;
    const double c2 = std::sqrt(1.0 - p.rho * p.rho);
    const double kb = std::exp(p.mu_j + 0.5 * p.sigma_j * p.sigma_j) - 1.0;
    const double drift_corr = p.lamb * kb;
    const double lam_dt = p.lamb * dt;

    // Output: paths[path_idx][step_idx],  step 0 = S0
    std::vector<std::vector<double>> paths(n_paths, std::vector<double>(n_steps + 1, S0));

    #ifdef BATESPRICER_USE_OPENMP
    #pragma omp parallel
    #endif
    {
        #ifdef BATESPRICER_USE_OPENMP
        int tid = omp_get_thread_num();
        #else
        int tid = 0;
        #endif

        std::mt19937_64 rng(cfg.seed + tid * 1000);
        std::normal_distribution<double> ndist(0.0, 1.0);
        std::uniform_real_distribution<double> udist(0.0, 1.0);

        #ifdef BATESPRICER_USE_OPENMP
        #pragma omp for schedule(static)
        #endif
        for (int i = 0; i < n_paths; ++i) {
            // For antithetic: paths [0, half) get +Z, [half, n_paths) get -Z
            const double sign = (cfg.antithetic && i >= n_paths / 2) ? -1.0 : 1.0;

            // Seed per-path for reproducibility when using antithetics
            std::mt19937_64 path_rng(cfg.seed + i);
            std::normal_distribution<double> pn(0.0, 1.0);
            std::uniform_real_distribution<double> pu(0.0, 1.0);

            double cv = p.v0;
            double cs = S0;

            for (int j = 0; j < n_steps; ++j) {
                double Z1 = sign * pn(path_rng);
                double Z2 = sign * pn(path_rng);
                double Uj = pu(path_rng);
                double Zj = sign * pn(path_rng);

                double vt = std::max(cv, 0.0);

                // Poisson jump count
                int nj = poisson_icdf(lam_dt, Uj);
                double jump_m = 0.0;
                if (nj > 0) {
                    jump_m = nj * p.mu_j + std::sqrt(static_cast<double>(nj)) * p.sigma_j * Zj;
                }

                // Spot update (log-space)
                cs *= std::exp((r - q - 0.5 * vt - drift_corr) * dt
                               + std::sqrt(vt) * sqrt_dt * Z1 + jump_m);

                // Variance update (full truncation)
                cv += p.kappa * (p.theta - vt) * dt
                    + p.xi * std::sqrt(vt) * sqrt_dt * (c1 * Z1 + c2 * Z2);

                paths[i][j + 1] = cs;
            }
        }
    }

    return paths;
}

// ═════════════════════════════════════════════════════════════
// Quadratic-Exponential Scheme  (Andersen 2008)
//   Returns terminal spot at each maturity slice
// ═════════════════════════════════════════════════════════════

std::vector<std::vector<double>> generate_bates_qe_slices(
    double S0, const BatesParams& p,
    double dt, int total_steps,
    const std::vector<int>& maturity_step_indices,
    const MCConfig& cfg)
{
    const int n_paths = cfg.n_paths;
    const int n_mats  = static_cast<int>(maturity_step_indices.size());

    const double kb    = std::exp(p.mu_j + 0.5 * p.sigma_j * p.sigma_j) - 1.0;
    const double exk   = std::exp(-p.kappa * dt);
    const double c1_qe = p.xi * p.xi * exk / p.kappa * (1.0 - exk);
    const double c2_qe = p.theta * p.xi * p.xi * 0.5 / p.kappa * (1.0 - exk) * (1.0 - exk);
    const double r_inv_xi = p.rho / p.xi;
    const double s1r2  = std::sqrt(1.0 - p.rho * p.rho);
    const double lam_dt = p.lamb * dt;
    const double en_lam = std::exp(-p.lamb * dt);

    // Output: slices[path][mat_idx]
    std::vector<std::vector<double>> slices(n_paths, std::vector<double>(n_mats, 0.0));

    #ifdef BATESPRICER_USE_OPENMP
    #pragma omp parallel
    #endif
    {
        #ifdef BATESPRICER_USE_OPENMP
        #pragma omp for schedule(static)
        #endif
        for (int i = 0; i < n_paths; ++i) {
            double sign = (cfg.antithetic && i >= n_paths / 2) ? -1.0 : 1.0;

            std::mt19937_64 rng(cfg.seed + i);
            std::normal_distribution<double> nd(0.0, 1.0);
            std::uniform_real_distribution<double> ud(0.0, 1.0);

            double cv   = p.v0;
            double cl_s = std::log(S0);
            int next_mi = 0;

            for (int j = 0; j < total_steps; ++j) {
                double Zv = sign * nd(rng);
                double Zx = sign * nd(rng);
                double Zj = sign * nd(rng);
                double Uv = ud(rng);
                double Uj = ud(rng);

                double vt = cv;

                // QE variance step
                double m  = p.theta + (vt - p.theta) * exk;
                double s2 = vt * c1_qe + c2_qe;
                double psi = s2 / (m * m);
                double vn;

                if (psi <= 1.5) {
                    double b2 = 2.0 / psi - 1.0 + std::sqrt(2.0 / psi * (2.0 / psi - 1.0));
                    vn = (m / (1.0 + b2)) * (std::sqrt(b2) + Zv) * (std::sqrt(b2) + Zv);
                } else {
                    double pp = (psi - 1.0) / (psi + 1.0);
                    if (Uv > pp) {
                        vn = std::log((1.0 - pp) / (1.0 - Uv)) / ((1.0 - pp) / m);
                    } else {
                        vn = 0.0;
                    }
                }
                cv = vn;

                // Log-spot update
                double vi = 0.5 * (vt + vn) * dt;
                double ls_dr = -p.lamb * kb * dt - 0.5 * vi
                    + r_inv_xi * (vn - vt - p.kappa * p.theta * dt + p.kappa * vi);
                double ls_df = s1r2 * std::sqrt(std::max(vi, 0.0)) * Zx;

                // Jump
                int nj = poisson_icdf(lam_dt, Uj);
                double jm = 0.0;
                if (nj > 0) {
                    jm = nj * p.mu_j + std::sqrt(static_cast<double>(nj)) * p.sigma_j * Zj;
                }

                cl_s += ls_dr + ls_df + jm;

                // Record at maturity slices
                if (next_mi < n_mats && (j + 1) == maturity_step_indices[next_mi]) {
                    slices[i][next_mi] = std::exp(cl_s);
                    ++next_mi;
                    // Handle multiple maturities at the same step (unlikely but safe)
                    while (next_mi < n_mats && maturity_step_indices[next_mi] == (j + 1)) {
                        slices[i][next_mi] = std::exp(cl_s);
                        ++next_mi;
                    }
                }
            }
        }
    }

    return slices;
}

// ═════════════════════════════════════════════════════════════
// Price a set of vanilla options via MC (Full-Truncation Euler)
// ═════════════════════════════════════════════════════════════

std::vector<double> price_mc(
    double S0,
    const std::vector<MarketOption>& options,
    const BatesParams& p,
    const MCConfig& cfg)
{
    if (options.empty()) return {};

    // Determine T_max and dt
    double T_max = 0.0;
    for (auto& o : options) T_max = std::max(T_max, o.maturity);

    const int n_steps = cfg.n_steps;
    const double dt = T_max / n_steps;

    // Map each option to a time-step index
    std::vector<int> t_idxs(options.size());
    for (size_t i = 0; i < options.size(); ++i) {
        t_idxs[i] = std::max(1, std::min(static_cast<int>(std::round(options[i].maturity / dt)), n_steps));
    }

    // Generate full paths
    double r0 = options[0].r, q0 = options[0].q;
    auto paths = generate_bates_paths_ft(S0, r0, q0, p, T_max, cfg);

    // Martingale correction: normalise each time slice
    for (int j = 0; j <= n_steps; ++j) {
        double sum = 0.0;
        for (int i = 0; i < cfg.n_paths; ++i) sum += paths[i][j];
        double mean_s = sum / cfg.n_paths;
        if (mean_s > 1e-12) {
            double scale = S0 / mean_s;
            for (int i = 0; i < cfg.n_paths; ++i) paths[i][j] *= scale;
        }
    }

    // Price each option
    std::vector<double> prices(options.size(), 0.0);

    for (size_t oi = 0; oi < options.size(); ++oi) {
        double K = options[oi].strike;
        double T = options[oi].maturity;
        double r = options[oi].r;
        double q = options[oi].q;
        bool is_call = (options[oi].option_type == "CALL");
        int t_idx = t_idxs[oi];

        double drift_adj = std::exp((r - q) * T);
        double disc = std::exp(-r * T);

        double payoff_sum = 0.0;
        for (int i = 0; i < cfg.n_paths; ++i) {
            double s_final = paths[i][t_idx] * drift_adj;
            double val = is_call ? (s_final - K) : (K - s_final);
            if (val > 0.0) payoff_sum += val;
        }

        prices[oi] = (payoff_sum / cfg.n_paths) * disc;
    }

    return prices;
}

} // namespace batespricer
