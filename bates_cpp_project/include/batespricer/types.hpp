#pragma once

#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace batespricer {

// ─────────────────────────────────────────────────────────────
// Option type enum
// ─────────────────────────────────────────────────────────────
enum class OptionType { CALL, PUT };

// ─────────────────────────────────────────────────────────────
// Single market-observed option with mid price and bid-ask data
// ─────────────────────────────────────────────────────────────
struct MarketOption {
    double strike       = 0.0;
    double maturity     = 0.0;
    double market_price = 0.0;
    std::string option_type = "CALL";   // "CALL" or "PUT"
    double bid    = 0.0;
    double ask    = 0.0;
    double spread = 0.0;
    double r      = 0.0;               // risk-free rate at this maturity
    double q      = 0.0;               // dividend yield at this maturity
};

// ─────────────────────────────────────────────────────────────
// Bates model parameters (8 free parameters)
// ─────────────────────────────────────────────────────────────
struct BatesParams {
    double kappa   = 1.5;      // Mean-reversion speed
    double theta   = 0.25;     // Long-run variance
    double xi      = 0.6;      // Vol-of-vol
    double rho     = -0.2;     // Spot-vol correlation
    double v0      = 0.21;     // Initial variance
    double lamb    = 0.5;      // Jump intensity (λ)
    double mu_j    = -0.05;    // Mean jump size (log-space)
    double sigma_j = 0.2;      // Jump-size volatility

    // Pack into a flat array (for optimiser interface)
    std::vector<double> to_vector() const {
        return {kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j};
    }

    // Unpack from a flat array
    static BatesParams from_vector(const std::vector<double>& v) {
        return {v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]};
    }
};

// ─────────────────────────────────────────────────────────────
// Calibration output
// ─────────────────────────────────────────────────────────────
struct CalibrationResult {
    BatesParams params;
    double weighted_obj = 0.0;
    double rmse         = 0.0;
    int    iterations   = 0;
    bool   converged    = false;
};

// ─────────────────────────────────────────────────────────────
// Top-level container parsed from market_data.json
// ─────────────────────────────────────────────────────────────
struct MarketData {
    std::string ticker;
    double S0 = 0.0;
    std::string fetch_date;
    std::vector<MarketOption> options;
};

// ─────────────────────────────────────────────────────────────
// JSON deserialisation (nlohmann ADL convention)
// ─────────────────────────────────────────────────────────────
inline void from_json(const nlohmann::json& j, MarketOption& o) {
    j.at("strike").get_to(o.strike);
    j.at("maturity").get_to(o.maturity);
    j.at("market_price").get_to(o.market_price);
    j.at("option_type").get_to(o.option_type);
    j.at("bid").get_to(o.bid);
    j.at("ask").get_to(o.ask);
    o.spread = o.ask - o.bid;
    if (j.contains("r")) j.at("r").get_to(o.r);
    if (j.contains("q")) j.at("q").get_to(o.q);
}

inline void from_json(const nlohmann::json& j, MarketData& md) {
    j.at("ticker").get_to(md.ticker);
    j.at("S0").get_to(md.S0);
    j.at("fetch_date").get_to(md.fetch_date);
    j.at("options").get_to(md.options);
}

} // namespace batespricer
