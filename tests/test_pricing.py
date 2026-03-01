import pytest
import numpy as np
from batespricer.analytics import BatesAnalyticalPricer, BatesAnalyticalPricerFast, implied_volatility
from batespricer.instruments import (
    EuropeanOption, AsianOption, BarrierOption,
    OptionType, BarrierType
)
from batespricer.market import MarketEnvironment

# --- Test parameters ---
SPX_PARAMS = dict(kappa=1.441, theta=0.051, xi=0.771, rho=-0.715, v0=0.020,
                  lamb=0.147, mu_j=-0.192, sigma_j=0.208)

AAPL_PARAMS = dict(kappa=1.472, theta=0.082, xi=0.628, rho=-0.451, v0=0.051,
                   lamb=0.269, mu_j=-0.184, sigma_j=0.236)

# --- Analytical pricer tests ---
class TestAnalyticalPricers:
    # Analytical pricers should agree on European option prices.
    def test_naive_vs_fast_spx_call(self):
        S0, K, T, r, q = 6923.5, 7000.0, 0.5, 0.041, 0.0054
        p = SPX_PARAMS

        naive = BatesAnalyticalPricer.price_vectorized(
            S0, np.array([K]), np.array([T]), np.array([r]), np.array([q]),
            np.array(["CALL"]), p['kappa'], p['theta'], p['xi'], p['rho'],
            p['v0'], p['lamb'], p['mu_j'], p['sigma_j']
        )[0]

        fast = BatesAnalyticalPricerFast.price_vectorized(
            S0, np.array([K]), np.array([T]), np.array([r]), np.array([q]),
            np.array(["CALL"]), p['kappa'], p['theta'], p['xi'], p['rho'],
            p['v0'], p['lamb'], p['mu_j'], p['sigma_j']
        )[0]

        assert abs(naive - fast) < 2.00, f"Naive ({naive:.4f}) vs Fast ({fast:.4f})"

    def test_naive_vs_fast_aapl_put(self):
        S0, K, T, r, q = 278.12, 260.0, 0.25, 0.041, 0.0037
        p = AAPL_PARAMS

        naive = BatesAnalyticalPricer.price_vectorized(
            S0, np.array([K]), np.array([T]), np.array([r]), np.array([q]),
            np.array(["PUT"]), p['kappa'], p['theta'], p['xi'], p['rho'],
            p['v0'], p['lamb'], p['mu_j'], p['sigma_j']
        )[0]

        fast = BatesAnalyticalPricerFast.price_vectorized(
            S0, np.array([K]), np.array([T]), np.array([r]), np.array([q]),
            np.array(["PUT"]), p['kappa'], p['theta'], p['xi'], p['rho'],
            p['v0'], p['lamb'], p['mu_j'], p['sigma_j']
        )[0]

        assert abs(naive - fast) < 0.50, f"Naive ({naive:.4f}) vs Fast ({fast:.4f})"

    def test_vectorized_multiple_strikes(self):
        # Pricer handles a batch of strikes and maturities.
        S0, r, q = 6923.5, 0.041, 0.0054
        p = SPX_PARAMS
        K = np.array([6500, 6800, 7000, 7200])
        T = np.array([0.25, 0.25, 0.5, 0.5])
        r_v, q_v = np.full(4, r), np.full(4, q)
        types = np.array(["CALL", "CALL", "CALL", "CALL"])

        prices = BatesAnalyticalPricerFast.price_vectorized(
            S0, K, T, r_v, q_v, types,
            p['kappa'], p['theta'], p['xi'], p['rho'],
            p['v0'], p['lamb'], p['mu_j'], p['sigma_j']
        )

        assert len(prices) == 4
        assert all(p >= 0 for p in prices)
        # Deep ITM should be more expensive than OTM
        assert prices[0] > prices[2], "K=6500 call should be worth more than K=7000"

    def test_put_call_parity_analytical(self):
        """C - P = S*exp(-qT) - K*exp(-rT) for European options."""
        S0, K, T, r, q = 6923.5, 7000.0, 0.5, 0.041, 0.0054
        p = SPX_PARAMS

        args = (S0, np.array([K]), np.array([T]), np.array([r]), np.array([q]))
        kwargs = dict(kappa=p['kappa'], theta=p['theta'], xi=p['xi'], rho=p['rho'],
                      v0=p['v0'], lamb=p['lamb'], mu_j=p['mu_j'], sigma_j=p['sigma_j'])

        call = BatesAnalyticalPricerFast.price_vectorized(*args, np.array(["CALL"]), **kwargs)[0]
        put = BatesAnalyticalPricerFast.price_vectorized(*args, np.array(["PUT"]), **kwargs)[0]

        parity = S0 * np.exp(-q * T) - K * np.exp(-r * T)
        assert abs((call - put) - parity) < 1.0, (
            f"C-P = {call - put:.4f}, S*exp(-qT) - K*exp(-rT) = {parity:.4f}"
        )

# --- Implied volatility tests ---
class TestImpliedVolatility:

    def test_round_trip(self):
        """BS price → implied vol → should recover the input vol."""
        from scipy.stats import norm
        S, K, T, r, q, sigma = 100, 100, 1.0, 0.05, 0.0, 0.20
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        bs_price = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

        iv = implied_volatility(bs_price, S, K, T, r, q, "CALL")
        assert abs(iv - sigma) < 0.001, f"IV ({iv:.6f}) should match input sigma ({sigma})"

    def test_zero_price_returns_zero(self):
        assert implied_volatility(0.0, 100, 100, 1.0, 0.05, 0.0) == 0.0

    def test_negative_price_returns_zero(self):
        assert implied_volatility(-1.0, 100, 100, 1.0, 0.05, 0.0) == 0.0

# --- Instrument payoff tests ---
class TestInstrumentPayoffs:

    def test_european_call_payoff(self):
        option = EuropeanOption(K=100, T=1.0, option_type=OptionType.CALL)
        paths = np.array([[90, 95, 110], [90, 105, 80]])  # 2 paths, 3 steps
        payoffs = option.payoff(paths)
        np.testing.assert_array_equal(payoffs, [10, 0])

    def test_european_put_payoff(self):
        option = EuropeanOption(K=100, T=1.0, option_type=OptionType.PUT)
        paths = np.array([[90, 95, 110], [90, 105, 80]])
        payoffs = option.payoff(paths)
        np.testing.assert_array_equal(payoffs, [0, 20])

    def test_asian_call_payoff(self):
        option = AsianOption(K=100, T=1.0, option_type=OptionType.CALL)
        paths = np.array([[100, 90, 110, 120]])  # S0=100, then [90,110,120]
        payoff = option.payoff(paths)[0]
        expected = max(np.mean([90, 110, 120]) - 100, 0)  # avg=106.67, payoff=6.67
        assert abs(payoff - expected) < 0.01

    def test_barrier_doc_knocked_out(self):
        option = BarrierOption(K=100, T=1.0, barrier=80, barrier_type=BarrierType.DOWN_AND_OUT,
                               option_type=OptionType.CALL)
        paths = np.array([[100, 90, 75, 110]])  # hits barrier at 75
        payoff = option.payoff(paths)[0]
        assert payoff == 0.0, "Should be knocked out"

    def test_barrier_doc_survives(self):
        option = BarrierOption(K=100, T=1.0, barrier=80, barrier_type=BarrierType.DOWN_AND_OUT,
                               option_type=OptionType.CALL)
        paths = np.array([[100, 95, 85, 120]])  # never hits 80
        payoff = option.payoff(paths)[0]
        assert payoff == 20.0, "Should survive and pay max(120-100, 0) = 20"

    def test_barrier_decomposition(self):
        """DOC + DIC = European payoff for any path."""
        paths = np.array([[100, 90, 75, 110], [100, 95, 85, 120]])
        K, B = 100, 80

        eu = EuropeanOption(K, 1.0, OptionType.CALL).payoff(paths)
        doc = BarrierOption(K, 1.0, B, BarrierType.DOWN_AND_OUT, OptionType.CALL).payoff(paths)
        dic = BarrierOption(K, 1.0, B, BarrierType.DOWN_AND_IN, OptionType.CALL).payoff(paths)

        np.testing.assert_array_almost_equal(doc + dic, eu,
            err_msg="DOC + DIC should equal European payoff")

# --- MarketEnvironment tests ---
class TestMarketEnvironment:

    def test_default_parameters(self):
        m = MarketEnvironment(S0=100, r=0.05)
        assert m.S0 == 100
        assert m.r == 0.05
        assert m.lamb == 0.0  # no jumps by default

    def test_bates_parameters(self):
        m = MarketEnvironment(S0=100, r=0.05, **SPX_PARAMS)
        assert m.kappa == 1.441
        assert m.lamb == 0.147
        assert m.mu_j == -0.192