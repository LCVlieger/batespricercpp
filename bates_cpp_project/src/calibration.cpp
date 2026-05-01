#include "batespricer/calibration.hpp"
#include "batespricer/analytics.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <vector>

#include <Eigen/Core>
#include <LBFGSB.h>

namespace batespricer {

// ═════════════════════════════════════════════════════════════
// Spread-based weight calculation
// ═════════════════════════════════════════════════════════════

static std::vector<double> compute_weights(
    const std::vector<MarketOption>& options, double sigma_cap)
{
    std::vector<double> w(options.size());
    for (size_t i = 0; i < options.size(); ++i) {
        double spread = std::max(std::abs(options[i].ask - options[i].bid), 0.01);
        w[i] = 1.0 / spread;
    }

    // Cap at mean + sigma_cap * std
    double sum  = std::accumulate(w.begin(), w.end(), 0.0);
    double mean = sum / w.size();
    double sq_sum = 0.0;
    for (auto v : w) sq_sum += (v - mean) * (v - mean);
    double std_dev = std::sqrt(sq_sum / w.size());
    double cap = mean + sigma_cap * std_dev;

    for (auto& v : w) v = std::min(v, cap);
    return w;
}

// ═════════════════════════════════════════════════════════════
// Objective functor for LBFGSpp
//   f(x) = sqrt( mean( (model_price - mkt_price)^2 * w^2 ) )
//   Gradient via central finite differences
// ═════════════════════════════════════════════════════════════

class BatesObjective {
public:
    BatesObjective(
        double S0,
        const std::vector<MarketOption>& options,
        const std::vector<double>& weights,
        double fd_eps,
        bool verbose)
        : S0_(S0), options_(options), weights_(weights),
          fd_eps_(fd_eps), verbose_(verbose), eval_count_(0) {}

    // Operator required by LBFGSpp: returns f(x) and fills grad
    double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
        ++eval_count_;

        // Evaluate objective at current point
        double fx = evaluate(x);

        // Central finite differences for gradient
        const int n = static_cast<int>(x.size());
        grad.resize(n);

        for (int k = 0; k < n; ++k) {
            Eigen::VectorXd x_plus  = x;
            Eigen::VectorXd x_minus = x;
            x_plus[k]  += fd_eps_;
            x_minus[k] -= fd_eps_;

            double f_plus  = evaluate(x_plus);
            double f_minus = evaluate(x_minus);
            grad[k] = (f_plus - f_minus) / (2.0 * fd_eps_);
        }

        if (verbose_ && eval_count_ % 5 == 0) {
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "[Iter " << eval_count_ << "] W-Obj: " << fx
                      << " | k:" << x[0] << " th:" << x[1]
                      << " xi:" << x[2] << " rho:" << x[3]
                      << " v0:" << x[4] << " | L:" << x[5]
                      << " muJ:" << x[6] << " sJ:" << x[7] << std::endl;
        }

        return fx;
    }

    int eval_count() const { return eval_count_; }

private:
    double evaluate(const Eigen::VectorXd& x) {
        BatesParams p;
        p.kappa   = x[0];
        p.theta   = x[1];
        p.xi      = x[2];
        p.rho     = x[3];
        p.v0      = x[4];
        p.lamb    = x[5];
        p.mu_j    = x[6];
        p.sigma_j = x[7];

        std::vector<double> model_prices;
        try {
            model_prices = price_analytical_fast(S0_, options_, p);
        } catch (...) {
            return 1e12;
        }

        // Check for NaN or negative prices
        for (auto v : model_prices) {
            if (std::isnan(v) || v < 0.0) return 1e10;
        }

        // Weighted RMSE
        double sum_sq = 0.0;
        for (size_t i = 0; i < options_.size(); ++i) {
            double diff = (model_prices[i] - options_[i].market_price) * weights_[i];
            sum_sq += diff * diff;
        }

        return std::sqrt(sum_sq / options_.size());
    }

    double S0_;
    const std::vector<MarketOption>& options_;
    const std::vector<double>& weights_;
    double fd_eps_;
    bool verbose_;
    int eval_count_;
};

// ═════════════════════════════════════════════════════════════
// Public calibration entry point
// ═════════════════════════════════════════════════════════════

CalibrationResult calibrate(
    double S0,
    const std::vector<MarketOption>& options,
    const CalibrationConfig& cfg,
    const ParamBounds& bounds)
{
    CalibrationResult result;

    if (options.empty()) {
        std::cerr << "[calibrate] No options provided." << std::endl;
        return result;
    }

    // Compute weights
    auto weights = compute_weights(options, cfg.sigma_cap);

    // Set up LBFGSpp solver
    LBFGSpp::LBFGSBParam<double> param;
    param.epsilon    = cfg.tol;
    param.max_iterations = cfg.max_iter;
    param.m = 10;  // L-BFGS memory size

    LBFGSpp::LBFGSBSolver<double> solver(param);

    // Initial guess (same as Python)
    Eigen::VectorXd x(8);
    x << 1.5, 0.25, 0.6, -0.2, 0.21, 0.5, -0.05, 0.2;

    // Bounds
    Eigen::VectorXd lb(8), ub(8);
    lb << bounds.kappa_lo, bounds.theta_lo, bounds.xi_lo, bounds.rho_lo,
          bounds.v0_lo, bounds.lamb_lo, bounds.mu_j_lo, bounds.sigma_j_lo;
    ub << bounds.kappa_hi, bounds.theta_hi, bounds.xi_hi, bounds.rho_hi,
          bounds.v0_hi, bounds.lamb_hi, bounds.mu_j_hi, bounds.sigma_j_hi;

    // Objective functor
    BatesObjective objective(S0, options, weights, cfg.eps, cfg.verbose);

    // Run optimisation
    double fx = 0.0;
    int niter = 0;
    try {
        niter = solver.minimize(objective, x, fx, lb, ub);
        result.converged = true;
    } catch (const std::exception& e) {
        std::cerr << "[calibrate] Optimisation failed: " << e.what() << std::endl;
        result.converged = false;
    }

    // Extract calibrated parameters
    result.params.kappa   = x[0];
    result.params.theta   = x[1];
    result.params.xi      = x[2];
    result.params.rho     = x[3];
    result.params.v0      = x[4];
    result.params.lamb    = x[5];
    result.params.mu_j    = x[6];
    result.params.sigma_j = x[7];
    result.weighted_obj   = fx;
    result.iterations     = niter;

    // Compute final RMSE (unweighted, in dollars)
    auto final_prices = price_analytical_fast(S0, options, result.params);
    double rmse_sum = 0.0;
    for (size_t i = 0; i < options.size(); ++i) {
        double d = final_prices[i] - options[i].market_price;
        rmse_sum += d * d;
    }
    result.rmse = std::sqrt(rmse_sum / options.size());

    return result;
}

} // namespace batespricer
