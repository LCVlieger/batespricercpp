#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include "batespricer/types.hpp"
#include "batespricer/analytics.hpp"
#include "batespricer/monte_carlo.hpp"
#include "batespricer/calibration.hpp"

namespace fs = std::filesystem;

// ═════════════════════════════════════════════════════════════
// Utility: print a horizontal rule
// ═════════════════════════════════════════════════════════════
static void hr() {
    std::cout << std::string(72, '-') << "\n";
}

// ═════════════════════════════════════════════════════════════
// Utility: locate market_data.json
// ═════════════════════════════════════════════════════════════
static std::string find_data_file(int argc, char* argv[]) {
    // 1. CLI argument
    if (argc >= 2) return argv[1];
    // 2. Default relative path
    if (fs::exists("data/market_data.json")) return "data/market_data.json";
    // 3. One level up
    if (fs::exists("../data/market_data.json")) return "../data/market_data.json";
    return "";
}

// ═════════════════════════════════════════════════════════════
// Main entry point
// ═════════════════════════════════════════════════════════════
int main(int argc, char* argv[]) {
    std::cout << "\n";
    hr();
    std::cout << "  BATES MODEL PRICER  (C++)\n";
    std::cout << "  Stochastic Volatility Jump-Diffusion\n";
    hr();
    std::cout << "\n";

    // ── 1. Locate and parse JSON ─────────────────────────────
    std::string data_path = find_data_file(argc, argv);
    if (data_path.empty()) {
        std::cerr << "[ERROR] Cannot find market_data.json.\n"
                  << "  Usage: batespricer_cpp [path/to/market_data.json]\n"
                  << "  Or place the file at data/market_data.json\n";
        return 1;
    }

    std::cout << "[1] Loading market data from: " << data_path << "\n";

    std::ifstream ifs(data_path);
    if (!ifs.is_open()) {
        std::cerr << "[ERROR] Cannot open file: " << data_path << "\n";
        return 1;
    }

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(ifs);
    } catch (const nlohmann::json::parse_error& e) {
        std::cerr << "[ERROR] JSON parse error: " << e.what() << "\n";
        return 1;
    }

    batespricer::MarketData md;
    try {
        md = j.get<batespricer::MarketData>();
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] JSON schema error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "  Ticker:       " << md.ticker << "\n"
              << "  Spot (S0):    " << md.S0 << "\n"
              << "  Fetch date:   " << md.fetch_date << "\n"
              << "  # Options:    " << md.options.size() << "\n\n";

    if (md.options.empty()) {
        std::cerr << "[ERROR] No options in data file.\n";
        return 1;
    }

    // ── 2. Calibrate ────────────────────────────────────────
    std::cout << "[2] Calibrating Bates model (L-BFGS-B, Gauss-Legendre pricer)...\n\n";

    batespricer::CalibrationConfig cal_cfg;
    cal_cfg.verbose = true;

    auto result = batespricer::calibrate(md.S0, md.options, cal_cfg);

    std::cout << "\n";
    hr();
    std::cout << "  CALIBRATION RESULT\n";
    hr();

    auto& p = result.params;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  kappa   = " << p.kappa   << "\n"
              << "  theta   = " << p.theta   << "\n"
              << "  xi      = " << p.xi      << "\n"
              << "  rho     = " << p.rho     << "\n"
              << "  v0      = " << p.v0      << "\n"
              << "  lambda  = " << p.lamb    << "\n"
              << "  mu_j    = " << p.mu_j    << "\n"
              << "  sigma_j = " << p.sigma_j << "\n";
    std::cout << "\n"
              << "  Weighted obj: " << result.weighted_obj << "\n"
              << "  RMSE ($):     " << result.rmse << "\n"
              << "  Iterations:   " << result.iterations << "\n"
              << "  Converged:    " << (result.converged ? "YES" : "NO") << "\n\n";

    // ── 3. Re-price with calibrated params (analytical) ─────
    std::cout << "[3] Re-pricing all options (analytical)...\n";

    auto prices_ana = batespricer::price_analytical_fast(md.S0, md.options, result.params);

    // Show first 10 options as a sample
    int n_show = std::min(static_cast<int>(md.options.size()), 10);
    std::cout << "\n";
    std::cout << std::setw(10) << "Strike"
              << std::setw(8)  << "T"
              << std::setw(6)  << "Type"
              << std::setw(12) << "Market"
              << std::setw(12) << "Model"
              << std::setw(12) << "Error"
              << "\n";
    hr();

    for (int i = 0; i < n_show; ++i) {
        auto& o = md.options[i];
        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(10) << o.strike
                  << std::setw(8)  << std::setprecision(3) << o.maturity
                  << std::setw(6)  << o.option_type
                  << std::setw(12) << std::setprecision(2) << o.market_price
                  << std::setw(12) << prices_ana[i]
                  << std::setw(12) << std::setprecision(4) << (prices_ana[i] - o.market_price)
                  << "\n";
    }
    if (static_cast<int>(md.options.size()) > n_show) {
        std::cout << "  ... (" << md.options.size() - n_show << " more options)\n";
    }

    // ── 4. MC verification on a small subset ────────────────
    std::cout << "\n[4] Monte Carlo verification (5 sample options, 100k paths)...\n";

    int n_mc_sample = std::min(static_cast<int>(md.options.size()), 5);
    std::vector<batespricer::MarketOption> mc_subset(
        md.options.begin(), md.options.begin() + n_mc_sample);

    batespricer::MCConfig mc_cfg;
    mc_cfg.n_paths = 100000;
    mc_cfg.n_steps = 252;

    auto prices_mc = batespricer::price_mc(md.S0, mc_subset, result.params, mc_cfg);

    std::cout << "\n"
              << std::setw(10) << "Strike"
              << std::setw(8)  << "T"
              << std::setw(6)  << "Type"
              << std::setw(12) << "Analytical"
              << std::setw(12) << "MC"
              << std::setw(12) << "Diff"
              << "\n";
    hr();

    for (int i = 0; i < n_mc_sample; ++i) {
        auto& o = mc_subset[i];
        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(10) << o.strike
                  << std::setw(8)  << std::setprecision(3) << o.maturity
                  << std::setw(6)  << o.option_type
                  << std::setw(12) << std::setprecision(2) << prices_ana[i]
                  << std::setw(12) << prices_mc[i]
                  << std::setw(12) << std::setprecision(4) << (prices_mc[i] - prices_ana[i])
                  << "\n";
    }

    std::cout << "\n";
    hr();
    std::cout << "  Done.\n";
    hr();
    std::cout << "\n";

    return 0;
}
