from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from ..market import MarketEnvironment
from .mc_kernels import generate_paths_kernel, generate_heston_paths, generate_heston_paths_crn, generate_bates_paths, generate_bates_paths_crn

class StochasticProcess(ABC):
    def __init__(self, market: MarketEnvironment):
        self.market = market

    @abstractmethod
    def generate_paths(self, T: float, n_paths: int, n_steps: int) -> np.ndarray:
        pass

class BlackScholesProcess(StochasticProcess):
    def generate_paths(self, T: float, n_paths: int, n_steps: int) -> np.ndarray:
        return generate_paths_kernel(
            self.market.S0, self.market.r, self.market.q, self.market.sigma,
            T, n_paths, n_steps
        )

class HestonProcess(StochasticProcess):
    def generate_paths(self, T: float, n_paths: int, n_steps: int, noise=None) -> np.ndarray:
        args = (
            self.market.S0, self.market.r, self.market.q,
            self.market.v0, self.market.kappa, self.market.theta,
            self.market.xi, self.market.rho, T, n_paths, n_steps
        )
        # common random numbers 
        if noise is not None:
            return generate_heston_paths_crn(*args, noise)
        return generate_heston_paths(*args)
    
class BatesProcess(StochasticProcess):
    @property
    def noise_channels(self) -> int:
        return 4 # Asset, Vol, JumpProb, JumpSize

    def generate_paths(self, T: float, n_paths: int, n_steps: int, noise=None) -> np.ndarray:
        # Assumes market has Bates params. If not, defaults provided for safety.
        lamb = getattr(self.market, 'lamb', 0.0)
        mu_j = getattr(self.market, 'mu_j', 0.0)
        sigma_j = getattr(self.market, 'sigma_j', 0.0)
        
        args = (
            self.market.S0, self.market.r, self.market.q,
            self.market.v0, self.market.kappa, self.market.theta,
            self.market.xi, self.market.rho, 
            lamb, mu_j, sigma_j,
            T, n_paths, n_steps
        )
        
        if noise is not None:
            return generate_bates_paths_crn(*args, noise)
        return generate_bates_paths(*args)