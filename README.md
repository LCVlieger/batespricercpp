# batespricer — C++

C++ port of [batespricer](https://github.com/LCVlieger/batespricer). Calibrates the Bates stochastic volatility jump-diffusion model to live options data and prices vanillas using both semi-analytical and Monte Carlo methods. The original Python repo has the full feature set (exotics, Greeks, notebooks); this repository increases computation speed utilizing C++.

## Overview

Market data is pulled by a small Python script (`yfinance` + FRED yield curves) and written to JSON. Everything else, that is, Fourier inversion, quadrature, path simulation, optimisation, runs in C++17.

```
Python (fetch)  →  market_data.json  →  C++ (calibrate + price)
```

**Semi-analytical pricer** — Albrecher characteristic function, Gauss-Legendre quadrature, maturity-bucketed CF caching.

**Monte Carlo pricer** — Full-truncation Euler (Lord et al. 2010), parallelised with OpenMP when available.

**Calibrator** — Minimises spread-weighted squared residuals over 8 Bates parameters via L-BFGS-B. Calibrates to ~300 OTM options per asset.

## Structure

```
bates_cpp_project/
├── CMakeLists.txt
├── data/
│   └── market_data.json
├── include/batespricer/
│   ├── types.hpp            # BatesParams, MarketOption, MarketData, JSON parsing
│   ├── analytics.hpp        # Fourier pricer interface
│   ├── monte_carlo.hpp      # MC pricer interface
│   └── calibration.hpp      # L-BFGS-B calibrator interface
├── src/
│   ├── main.cpp             # CLI: load → calibrate → price → MC verify
│   ├── analytics.cpp        # CF evaluation, GL quadrature, batch pricing
│   ├── monte_carlo.cpp      # Path generation, payoff evaluation
│   └── calibration.cpp      # Objective, bounds, solver wrapper
└── scripts/
    └── fetch_market_data.py # Options chains + yield curve → JSON
```

## Dependencies

Fetched automatically via CMake `FetchContent`:
- [nlohmann/json](https://github.com/nlohmann/json) — JSON I/O
- [Eigen 3.4](https://eigen.tuxfamily.org/) — linear algebra
- [LBFGSpp](https://github.com/yixuan/LBFGSpp) — L-BFGS-B solver
- OpenMP (optional) — MC parallelism

## Build

```bash
git clone https://github.com/LCVlieger/batespricercpp
cd batespricercpp/bates_cpp_project

# fetch market data (needs python + yfinance/numpy/scipy)
python scripts/fetch_market_data.py

# build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# run
./build/batespricer_cpp
```

The binary calibrates the model, re-prices every option analytically, and cross-checks a subset against Monte Carlo.

## Report

See [batespricer.pdf](batespricer.pdf) for methodology and results. The table below is from the Python reference run (T=1, K=1.05·S₀, B=0.8·S₀):

| | | SPX | | | AAPL | | |
|---|---|---|---|---|---|---|---|
| **Product** | **Price** | **Δ** | **Γ** | **Price** | **Δ** | **Γ** |  **V**_var |
| European call | $406.02 | 0.633 | 0.0004 | $28.03 | 0.607 | 0.0060 | 97.50 |
| Down-Out call | $393.46 | 0.636 | 0.0004 | $26.38 | 0.630 | 0.0046 | 70.93 |
| Down-In call | $12.56 | −0.003 | 0.0000 | $1.65 | −0.023 | 0.0014 | 26.57 |
| Asian call | $134.46 | 0.466 | 0.0009 | $12.32 | 0.490 | 0.0112 | 85.89 |

## References

- Bates (1996). Jumps and stochastic volatility. *Review of Financial Studies*.
- Albrecher et al. (2007). The little Heston trap. *Wilmott Magazine*.
- Kilin (2007). Accelerating the calibration of stochastic volatility models. *CPQF Working Paper*.
- Andersen (2008). Efficient simulation of the Heston stochastic volatility model. *J. Comp. Finance*.
- Lord et al. (2010). A comparison of biased simulation schemes. *Quantitative Finance*.
- Gatheral (2006). *The Volatility Surface*. Wiley.
