#include "batespricer/analytics.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace batespricer {

// ═════════════════════════════════════════════════════════════
// Helpers
// ═════════════════════════════════════════════════════════════

static constexpr double PI = 3.14159265358979323846;

// Standard-normal CDF via the complementary error function
static double norm_cdf(double x) {
    return 0.5 * std::erfc(-x * std::sqrt(0.5));
}

// Standard-normal PDF
static double norm_pdf(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * PI);
}

// ═════════════════════════════════════════════════════════════
// Gauss-Legendre nodes and weights on [-1, 1]
//   Golub-Welsch-style recurrence for moderate N
// ═════════════════════════════════════════════════════════════

static void gauss_legendre(int N, std::vector<double>& nodes, std::vector<double>& weights) {
    nodes.resize(N);
    weights.resize(N);

    for (int i = 0; i < N; ++i) {
        // Initial guess (Tricomi approximation)
        double z = std::cos(PI * (i + 0.75) / (N + 0.5));
        double pp = 0.0;

        // Newton-Raphson refinement
        for (int iter = 0; iter < 100; ++iter) {
            double p1 = 1.0, p2 = 0.0;
            for (int j = 0; j < N; ++j) {
                double p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1.0);
            }
            pp = N * (z * p1 - p2) / (z * z - 1.0);
            double z1 = z;
            z -= p1 / pp;
            if (std::abs(z - z1) < 1e-15) break;
        }
        nodes[i]   = z;
        weights[i] = 2.0 / ((1.0 - z * z) * pp * pp);
    }
}

// ═════════════════════════════════════════════════════════════
// Bates Characteristic Function  (Albrecher 2007)
// ═════════════════════════════════════════════════════════════

std::complex<double> bates_cf(
    double phi_re, double T, double r, double q,
    const BatesParams& p)
{
    const std::complex<double> I(0.0, 1.0);
    const std::complex<double> phi(phi_re, 0.0);

    double xs = std::max(p.xi, 1e-6);

    std::complex<double> d = std::sqrt(
        (p.kappa - p.rho * xs * phi * I) * (p.kappa - p.rho * xs * phi * I)
        + xs * xs * (phi * I + phi * phi));

    std::complex<double> g = (p.kappa - p.rho * xs * phi * I - d)
                           / (p.kappa - p.rho * xs * phi * I + d);

    std::complex<double> et = std::exp(-d * T);

    std::complex<double> D = ((p.kappa - p.rho * xs * phi * I - d) / (xs * xs))
                           * ((1.0 - et) / (1.0 - g * et));

    std::complex<double> C = phi * I * (r - q) * T
        + (p.kappa * p.theta / (xs * xs))
        * ((p.kappa - p.rho * xs * phi * I - d) * T
           - 2.0 * std::log((1.0 - g * et) / (1.0 - g + 1e-15)));

    // Jump component (Merton log-normal)
    double kb = std::exp(p.mu_j + 0.5 * p.sigma_j * p.sigma_j) - 1.0;
    std::complex<double> jump = p.lamb * T
        * (std::exp(I * phi * p.mu_j - 0.5 * p.sigma_j * p.sigma_j * phi * phi)
           - 1.0 - I * phi * kb);

    return std::exp(C + D * p.v0 + jump);
}

// Internal: evaluate CF for a complex-valued phi argument
static std::complex<double> bates_cf_complex(
    std::complex<double> phi, double T, double r, double q,
    const BatesParams& p)
{
    const std::complex<double> I(0.0, 1.0);
    double xs = std::max(p.xi, 1e-6);

    std::complex<double> d = std::sqrt(
        (p.kappa - p.rho * xs * phi * I) * (p.kappa - p.rho * xs * phi * I)
        + xs * xs * (phi * I + phi * phi));

    std::complex<double> g = (p.kappa - p.rho * xs * phi * I - d)
                           / (p.kappa - p.rho * xs * phi * I + d);

    std::complex<double> et = std::exp(-d * T);

    std::complex<double> D_val = ((p.kappa - p.rho * xs * phi * I - d) / (xs * xs))
                               * ((1.0 - et) / (1.0 - g * et));

    std::complex<double> C_val = phi * I * (r - q) * T
        + (p.kappa * p.theta / (xs * xs))
        * ((p.kappa - p.rho * xs * phi * I - d) * T
           - 2.0 * std::log((1.0 - g * et) / (1.0 - g + 1e-15)));

    double kb = std::exp(p.mu_j + 0.5 * p.sigma_j * p.sigma_j) - 1.0;
    std::complex<double> jump = p.lamb * T
        * (std::exp(I * phi * p.mu_j - 0.5 * p.sigma_j * p.sigma_j * phi * phi)
           - 1.0 - I * phi * kb);

    return std::exp(C_val + D_val * p.v0 + jump);
}

// ═════════════════════════════════════════════════════════════
// Naive Midpoint Pricer   (N=296, u_limit=10 000)
// ═════════════════════════════════════════════════════════════

std::vector<double> price_analytical(
    double S0,
    const std::vector<MarketOption>& options,
    const BatesParams& p)
{
    const int N_grid  = 296;
    const double u_limit = 10000.0;
    const std::complex<double> I(0.0, 1.0);

    // Build quadratically-spaced grid: u = linspace(0,1,N+1)^2 * u_limit
    std::vector<double> u(N_grid + 1);
    for (int i = 0; i <= N_grid; ++i) {
        double t = static_cast<double>(i) / N_grid;
        u[i] = t * t * u_limit;
    }
    u[0] = 1e-12;

    // du[k] = u[k+1] - u[k]
    std::vector<double> du(N_grid);
    for (int k = 0; k < N_grid; ++k) du[k] = u[k + 1] - u[k];

    // Group options by maturity for CF caching
    std::unordered_map<double, std::vector<int>> t_groups;
    for (int i = 0; i < static_cast<int>(options.size()); ++i) {
        t_groups[options[i].maturity].push_back(i);
    }

    std::vector<double> prices(options.size(), 0.0);

    for (auto& [T, idxs] : t_groups) {
        double r_val = options[idxs[0]].r;
        double q_val = options[idxs[0]].q;

        // Evaluate CF at all grid points for this maturity
        std::vector<std::complex<double>> cf1(N_grid + 1), cf2(N_grid + 1);
        for (int k = 0; k <= N_grid; ++k) {
            std::complex<double> u_c(u[k], 0.0);
            cf1[k] = bates_cf_complex(u_c - I, T, r_val, q_val, p);
            cf2[k] = bates_cf_complex(u_c,      T, r_val, q_val, p);
        }

        for (int idx : idxs) {
            double K = options[idx].strike;
            double log_KS = std::log(K / S0);

            // Trapezoidal integration
            double sum1 = 0.0, sum2 = 0.0;
            for (int k = 0; k < N_grid; ++k) {
                std::complex<double> u1(u[k], 0.0), u2(u[k + 1], 0.0);

                auto f1_lo = (cf1[k]     * std::exp(-I * u1 * log_KS - (r_val - q_val) * T)) / (I * u1);
                auto f1_hi = (cf1[k + 1] * std::exp(-I * u2 * log_KS - (r_val - q_val) * T)) / (I * u2);
                sum1 += 0.5 * (f1_lo.real() + f1_hi.real()) * du[k];

                auto f2_lo = (cf2[k]     * std::exp(-I * u1 * log_KS)) / (I * u1);
                auto f2_hi = (cf2[k + 1] * std::exp(-I * u2 * log_KS)) / (I * u2);
                sum2 += 0.5 * (f2_lo.real() + f2_hi.real()) * du[k];
            }

            double P1_raw = 0.5 + sum1 / PI;
            double P2_raw = 0.5 + sum2 / PI;

            double F = S0 * std::exp((r_val - q_val) * T);
            bool is_otm = (K > F);

            double P1 = is_otm ? P1_raw : 1.0 - P1_raw;
            double P2 = is_otm ? P2_raw : 1.0 - P2_raw;

            double pr = is_otm
                ? S0 * std::exp(-q_val * T) * P1 - K * std::exp(-r_val * T) * P2
                : K * std::exp(-r_val * T) * P2 - S0 * std::exp(-q_val * T) * P1;

            // Put-call parity adjustment if needed
            bool is_put = (options[idx].option_type == "PUT");
            if (is_put && !is_otm) {
                // already priced as ITM put (OTM call formula gives put)
            } else if (is_put && is_otm) {
                double adj = S0 * std::exp(-q_val * T) - K * std::exp(-r_val * T);
                pr -= adj;
            } else if (!is_put && !is_otm) {
                double adj = S0 * std::exp(-q_val * T) - K * std::exp(-r_val * T);
                pr += adj;
            }

            prices[idx] = std::max(pr, 0.0);
            if (std::isnan(prices[idx])) prices[idx] = 0.0;
        }
    }

    return prices;
}

// ═════════════════════════════════════════════════════════════
// Fast Gauss-Legendre Pricer   (N=300, u_max=20 000)
// ═════════════════════════════════════════════════════════════

std::vector<double> price_analytical_fast(
    double S0,
    const std::vector<MarketOption>& options,
    const BatesParams& p)
{
    const int N_nodes  = 300;
    const double u_max = 20000.0;
    const std::complex<double> I(0.0, 1.0);

    // Compute Gauss-Legendre nodes and weights on [-1, 1]
    std::vector<double> gl_nodes, gl_weights;
    gauss_legendre(N_nodes, gl_nodes, gl_weights);

    // Transform to [0, u_max]:  u = 0.5*u_max*(node+1),  w = 0.5*u_max*weight
    std::vector<double> u(N_nodes), w(N_nodes);
    for (int k = 0; k < N_nodes; ++k) {
        u[k] = std::max(0.5 * u_max * (gl_nodes[k] + 1.0), 1e-12);
        w[k] = 0.5 * u_max * gl_weights[k];
    }

    // Group by maturity for CF caching
    std::unordered_map<double, std::vector<int>> t_groups;
    for (int i = 0; i < static_cast<int>(options.size()); ++i) {
        t_groups[options[i].maturity].push_back(i);
    }

    std::vector<double> prices(options.size(), 0.0);

    for (auto& [T, idxs] : t_groups) {
        double r_val = options[idxs[0]].r;
        double q_val = options[idxs[0]].q;

        // Evaluate CF once for all N nodes at this maturity
        std::vector<std::complex<double>> cf1(N_nodes), cf2(N_nodes);
        for (int k = 0; k < N_nodes; ++k) {
            std::complex<double> u_c(u[k], 0.0);
            cf1[k] = bates_cf_complex(u_c - I, T, r_val, q_val, p);
            cf2[k] = bates_cf_complex(u_c,      T, r_val, q_val, p);
        }

        for (int idx : idxs) {
            double K = options[idx].strike;
            double log_KS = std::log(K / S0);

            double sum1 = 0.0, sum2 = 0.0;
            for (int k = 0; k < N_nodes; ++k) {
                std::complex<double> u_c(u[k], 0.0);

                auto f1 = (cf1[k] * std::exp(-I * u_c * log_KS - (r_val - q_val) * T)) / (I * u_c);
                sum1 += f1.real() * w[k];

                auto f2 = (cf2[k] * std::exp(-I * u_c * log_KS)) / (I * u_c);
                sum2 += f2.real() * w[k];
            }

            double P1_raw = 0.5 + sum1 / PI;
            double P2_raw = 0.5 + sum2 / PI;

            double F = S0 * std::exp((r_val - q_val) * T);
            bool is_otm = (K > F);

            double P1 = is_otm ? P1_raw : 1.0 - P1_raw;
            double P2 = is_otm ? P2_raw : 1.0 - P2_raw;

            double pr = is_otm
                ? S0 * std::exp(-q_val * T) * P1 - K * std::exp(-r_val * T) * P2
                : K * std::exp(-r_val * T) * P2 - S0 * std::exp(-q_val * T) * P1;

            // Put-call parity adjustment
            bool is_put = (options[idx].option_type == "PUT");
            if (is_put && is_otm) {
                pr -= (S0 * std::exp(-q_val * T) - K * std::exp(-r_val * T));
            } else if (!is_put && !is_otm) {
                pr += (S0 * std::exp(-q_val * T) - K * std::exp(-r_val * T));
            }

            prices[idx] = std::max(pr, 0.0);
            if (std::isnan(prices[idx])) prices[idx] = 0.0;
        }
    }

    return prices;
}

// ═════════════════════════════════════════════════════════════
// Black-Scholes Implied Volatility  (Brent's method)
// ═════════════════════════════════════════════════════════════

double implied_volatility(
    double price, double S, double K, double T,
    double r, double q, const std::string& option_type)
{
    if (price <= 0.0 || T <= 1e-8) return 0.0;

    double df_q = std::exp(-q * T);
    double df_r = std::exp(-r * T);

    bool is_put = (option_type == "PUT");
    double intrinsic = is_put
        ? std::max(K * df_r - S * df_q, 0.0)
        : std::max(S * df_q - K * df_r, 0.0);
    if (price < intrinsic) return 0.0;

    // BS pricing error as function of sigma
    auto bs_err = [&](double sigma) -> double {
        double sqrtT = std::sqrt(T);
        double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
        double d2 = d1 - sigma * sqrtT;
        double bs;
        if (is_put) {
            bs = K * df_r * norm_cdf(-d2) - S * df_q * norm_cdf(-d1);
        } else {
            bs = S * df_q * norm_cdf(d1) - K * df_r * norm_cdf(d2);
        }
        return bs - price;
    };

    // Brent's method on [1e-4, 5.0]
    double a = 1e-4, b = 5.0;
    double fa = bs_err(a), fb = bs_err(b);

    if (fa * fb > 0.0) return 0.0;  // no root bracketed

    if (std::abs(fa) < std::abs(fb)) { std::swap(a, b); std::swap(fa, fb); }

    double c = a, fc = fa;
    bool mflag = true;
    double d_val = 0.0, s = 0.0;

    for (int iter = 0; iter < 100; ++iter) {
        if (std::abs(fb) < 1e-12) return b;
        if (std::abs(b - a) < 1e-14) return b;

        if (std::abs(fa - fc) > 1e-15 && std::abs(fb - fc) > 1e-15) {
            // Inverse quadratic interpolation
            s = a * fb * fc / ((fa - fb) * (fa - fc))
              + b * fa * fc / ((fb - fa) * (fb - fc))
              + c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            s = b - fb * (b - a) / (fb - fa);  // Secant
        }

        bool cond1 = !(s > (3 * a + b) / 4.0 && s < b) && !(s < (3 * a + b) / 4.0 && s > b);
        bool cond2 = mflag && std::abs(s - b) >= std::abs(b - c) / 2.0;
        bool cond3 = !mflag && std::abs(s - b) >= std::abs(c - d_val) / 2.0;
        bool cond4 = mflag && std::abs(b - c) < 1e-14;
        bool cond5 = !mflag && std::abs(c - d_val) < 1e-14;

        if (cond1 || cond2 || cond3 || cond4 || cond5) {
            s = (a + b) / 2.0;
            mflag = true;
        } else {
            mflag = false;
        }

        double fs = bs_err(s);
        d_val = c;
        c = b; fc = fb;

        if (fa * fs < 0.0) { b = s; fb = fs; }
        else               { a = s; fa = fs; }

        if (std::abs(fa) < std::abs(fb)) {
            std::swap(a, b); std::swap(fa, fb);
        }
    }

    return b;
}

} // namespace batespricer
