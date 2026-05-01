# `batespricer` — C++ Implementation

High-performance C++ reimplementation of the Bates stochastic volatility jump-diffusion pricer. Calibrates to real-time options data, prices vanillas via semi-analytical Fourier inversion and Monte Carlo simulation, and verifies results across both engines.

> This is the C++ companion to the full-featured Python package [`batespricer`](https://github.com/LCVlieger/batespricer), which additionally supports exotic pricing (Asian, barrier), Greeks, and interactive notebooks. See the [report](batespricer.pdf) for the complete methodology and results.

## What this repo does

All compute-intensive work — characteristic function evaluation, Gauss-Legendre quadrature, Monte Carlo path simulation, and L-BFGS-B calibration — is implemented in strictly-typed C++17. A thin Python script handles market-data retrieval (yield curves, options chains via `yfinance`), serialising it to JSON for the C++ engine.

**Pipeline:** &ensp; Python (data fetch) &ensp;→&ensp; `market_data.json` &ensp;→&ensp; C++ (calibrate + price)

## Pricing and calibration

Two pricing methods are implemented:

- **Semi-analytical** — Fourier inversion of the Albrecher (2007) characteristic function with Gauss-Legendre quadrature and maturity-based CF caching.
- **Monte Carlo** — Full-truncation Euler scheme (Lord et al., 2010) with optional OpenMP parallelism.

Calibration minimises spread-weighted squared residuals via L-BFGS-B (using [LBFGSpp](https://github.com/yixuan/LBFGSpp) backed by Eigen). We calibrate to ~300 OTM options per asset. Results under the calibrated Bates model from the Python reference (T = 1, K = 1.05 · S₀, B = 0.8 · S₀):

| | | SPX ($6,923) | | | | AAPL ($278) | | |
|---|---|---|---|---|---|---|---|---|
| **Product** | **Price** | **Δ** | **Γ** | **V**_var | **Price** | **Δ** | **Γ** | **V**_var |
| European call | $406.02 | 0.633 | 0.0004 | 3141.7 | $28.03 | 0.607 | 0.0060 | 97.50 |
| Down-Out call | $393.46 | 0.636 | 0.0004 | 2759.4 | $26.38 | 0.630 | 0.0046 | 70.93 |
| Down-In call | $12.56 | −0.003 | 0.0000 |  382.3 | $1.65 | −0.023 | 0.0014 | 26.57 |
| Asian call | $134.46 | 0.466 | 0.0009 | 2842.5 | $12.32 | 0.490 | 0.0112 | 85.89 |

## Project structure

```
bates_cpp_project/
├── CMakeLists.txt                  # CMake build (FetchContent for deps)
├── data/
│   └── market_data.json            # Options chain + yield curve (from Python)
├── include/batespricer/
│   ├── types.hpp                   # BatesParams, MarketOption, MarketData
│   ├── analytics.hpp               # Fourier pricer (Gauss-Legendre + CF cache)
│   ├── monte_carlo.hpp             # Full-truncation MC pricer (OpenMP)
│   └── calibration.hpp             # L-BFGS-B calibrator (Eigen + LBFGSpp)
├── src/
│   ├── main.cpp                    # CLI entry point: load → calibrate → price → verify
│   ├── analytics.cpp               # Albrecher CF, quadrature, fast batch pricer
│   ├── monte_carlo.cpp             # Path generation, payoff evaluation
│   └── calibration.cpp             # Objective function, bounds, solver wrapper
└── scripts/
    └── fetch_market_data.py        # Downloads live options data → JSON
```

## Dependencies

All C++ dependencies are fetched automatically by CMake (`FetchContent`):

| Library | Role |
|---|---|
| [nlohmann/json](https://github.com/nlohmann/json) | JSON parsing (market data I/O) |
| [Eigen 3.4](https://eigen.tuxfamily.org/) | Linear algebra (optimiser backend) |
| [LBFGSpp](https://github.com/yixuan/LBFGSpp) | L-BFGS-B constrained optimisation |
| OpenMP *(optional)* | Monte Carlo path parallelism |

## Build and run

```bash
git clone https://github.com/LCVlieger/batespricercpp
cd batespricercpp/bates_cpp_project
```

**Fetch live market data** (requires Python with `yfinance`, `numpy`, `scipy`):
```bash
python scripts/fetch_market_data.py
```

**Build the C++ pricer:**
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

**Run:**
```bash
./build/batespricer_cpp              # auto-detects data/market_data.json
# or
./build/batespricer_cpp path/to/market_data.json
```

The executable calibrates the 8 Bates parameters, re-prices every option analytically, and cross-validates a subset against Monte Carlo.

## References

- **Bates, D.S. (1996).** Jumps and stochastic volatility. *Review of Financial Studies*.
- **Albrecher, H. et al. (2007).** The little Heston trap. *Wilmott Magazine*.
- **Kilin, F. (2007).** Accelerating the calibration of stochastic volatility models. *CPQF Working Paper*.
- **Andersen, L. (2008).** Efficient simulation of the Heston stochastic volatility model. *J. Computational Finance*.
- **Lord, R. et al. (2010).** A comparison of biased simulation schemes. *Quantitative Finance*.
- **Gatheral, J. (2006).** *The Volatility Surface*. Wiley.
