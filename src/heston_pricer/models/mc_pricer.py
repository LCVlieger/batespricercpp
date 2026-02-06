import numpy as np
from dataclasses import dataclass, replace
from typing import Dict, Tuple
from ..instruments import Option
from .process import StochasticProcess
from ..calibration import implied_volatility

@dataclass
class PricingResult:
    price: float
    std_error: float
    conf_interval_95: tuple[float, float]

class MonteCarloPricer:
    def __init__(self, process: StochasticProcess):
        self.process = process

    def price(self, option: Option, n_paths: int = 10000, n_steps: int = 100, **kwargs) -> PricingResult:
        paths = self.process.generate_paths(option.T, n_paths, n_steps, **kwargs)
        payoffs = option.payoff(paths)
    
        discount_factor = np.exp(-self.process.market.r * option.T)
        discounted_payoffs = payoffs * discount_factor
    
        mean_price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(n_paths)
        
        return PricingResult(
            price=mean_price,
            std_error=std_error,
            conf_interval_95=(mean_price - 1.96 * std_error, mean_price + 1.96 * std_error)
        )

    def compute_greeks(self, option: Option, n_paths: int = 10000, n_steps: int = 252, bump_ratio: float = 0.01, seed: int = 42) -> Dict[str, float]:
        """
        Computes Greeks using finite differences with Common Random Numbers (CRN).
        Adapted to handle variable noise channels (Heston=2, Bates=4).
        """
        original_market = self.process.market
        original_S0 = original_market.S0
        epsilon_s = original_S0 * bump_ratio
        epsilon_v = 0.001 
        
        # Determine dimensions needed
        n_channels = getattr(self.process, 'noise_channels', 2)
        
        rng = np.random.default_rng(seed)
        
        # Initialize Noise Tensor
        # If Bates (4 channels): 
        #   Ch 0,1,3: Gaussian (Asset, Vol, JumpSize)
        #   Ch 2: Uniform (JumpTrigger)
        if n_channels == 4:
            Z_CRN = np.zeros((4, n_steps, n_paths))
            Z_CRN[0] = rng.standard_normal((n_steps, n_paths)) # Asset
            Z_CRN[1] = rng.standard_normal((n_steps, n_paths)) # Vol
            Z_CRN[2] = rng.random((n_steps, n_paths))          # Jump Prob (Uniform)
            Z_CRN[3] = rng.standard_normal((n_steps, n_paths)) # Jump Size
        else:
            # Default Heston/BS (All Gaussian)
            Z_CRN = rng.standard_normal((n_channels, n_steps, n_paths))
        
        # 1. Base Price
        res_curr = self.price(option, n_paths, n_steps, noise=Z_CRN)
        
        # 2. Delta & Gamma (Bump S0)
        self.process.market = replace(original_market, S0 = original_S0 + epsilon_s)
        res_up = self.price(option, n_paths, n_steps, noise=Z_CRN)
        
        self.process.market = replace(original_market, S0 = original_S0 - epsilon_s)
        res_down = self.price(option, n_paths, n_steps, noise=Z_CRN)
        
        # 3. Vega (Bump v0)
        self.process.market = replace(original_market, v0 = original_market.v0 + epsilon_v, S0=original_S0)
        res_vega = self.price(option, n_paths, n_steps, noise=Z_CRN)
        
        # Restore market
        self.process.market = original_market

        delta = (res_up.price - res_down.price) / (2 * epsilon_s)
        gamma = (res_up.price - 2 * res_curr.price + res_down.price) / (epsilon_s ** 2)
        vega = (res_vega.price - res_curr.price) / epsilon_v
        
        return {
            "price": res_curr.price,
            "delta": delta,
            "gamma": gamma,
            "vega_v0": vega
        }