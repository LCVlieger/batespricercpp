#pragma once

#include "types.hpp"
#include <complex>
#include <string>
#include <vector>

namespace batespricer {

// ─────────────────────────────────────────────────────────────
// Bates characteristic function  (Albrecher 2007 formulation)
// ─────────────────────────────────────────────────────────────
std::complex<double> bates_cf(double phi, double T, double r, double q,
                              const BatesParams &p);

// ─────────────────────────────────────────────────────────────
// Semi-analytical pricing — naive midpoint quadrature
//   N_grid = 296,  u_limit = 10 000
// ─────────────────────────────────────────────────────────────
std::vector<double> price_analytical(double S0,
                                     const std::vector<MarketOption> &options,
                                     const BatesParams &p);

// ─────────────────────────────────────────────────────────────
// Semi-analytical pricing — Gauss-Legendre with maturity caching
//   N_nodes = 300,  u_max = 20 000
// ─────────────────────────────────────────────────────────────
std::vector<double>
price_analytical_fast(double S0, const std::vector<MarketOption> &options,
                      const BatesParams &p);

// ─────────────────────────────────────────────────────────────
// Black-Scholes implied volatility via Brent's method
// ─────────────────────────────────────────────────────────────
double implied_volatility(double price, double S, double K, double T, double r,
                          double q, const std::string &option_type);

} // namespace batespricer
