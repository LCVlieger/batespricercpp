from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from dataclasses import dataclass

class OptionType(Enum):
    CALL = 1
    PUT = -1

class Option(ABC):
    def __init__(self, K: float, T: float, option_type: OptionType):
        self.K = K
        self.T = T
        self.option_type = option_type

    @abstractmethod
    def payoff(self, prices: np.ndarray) -> np.ndarray:
        pass

class EuropeanOption(Option):
    def __init__(self, strike, maturity, option_type, price=None):
        self.strike = strike
        self.maturity = maturity
        self.option_type = option_type
        self.price = price
    def payoff(self, prices: np.ndarray) -> np.ndarray:
        S_T = prices[:, -1]
        phi = self.option_type.value
        return np.maximum(phi * (S_T - self.K), 0)
    
class AsianOption(Option):
    def payoff(self, prices: np.ndarray) -> np.ndarray:
        # Average across time steps (excluding t=0)
        average_price = np.mean(prices[:, 1:], axis=1)
        phi = self.option_type.value
        return np.maximum(phi * (average_price - self.K), 0)
    
class BarrierType(Enum):
    DOWN_AND_OUT = 1
    DOWN_AND_IN = 2
    UP_AND_OUT = 3
    UP_AND_IN = 4

class BarrierOption(Option):
    def __init__(self, K: float, T: float, barrier: float, barrier_type: BarrierType, option_type: OptionType):
        super().__init__(K, T, option_type)
        self.barrier = barrier
        self.barrier_type = barrier_type

    def payoff(self, prices: np.ndarray) -> np.ndarray:
        S_T = prices[:, -1]
        phi = self.option_type.value
        intrinsic_payoff = np.maximum(phi * (S_T - self.K), 0)
        
        path_min = np.min(prices, axis=1)
        path_max = np.max(prices, axis=1)
        
        # Barrier conditions below
        if self.barrier_type == BarrierType.DOWN_AND_OUT:
            active_mask = path_min > self.barrier
            return intrinsic_payoff * active_mask
            
        elif self.barrier_type == BarrierType.DOWN_AND_IN:
            active_mask = path_min <= self.barrier
            return intrinsic_payoff * active_mask
            
        elif self.barrier_type == BarrierType.UP_AND_OUT:
            active_mask = path_max < self.barrier
            return intrinsic_payoff * active_mask
            
        elif self.barrier_type == BarrierType.UP_AND_IN:
            active_mask = path_max >= self.barrier
            return intrinsic_payoff * active_mask
            
        return intrinsic_payoff
    
    
@dataclass
class MarketOption:
    strike: float
    maturity: float
    market_price: float
    option_type: str = OptionType.CALL