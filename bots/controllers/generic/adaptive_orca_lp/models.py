"""
Data types for the Adaptive Orca LP controller.

Everything in this module is pure data: no network access, no Gateway calls and no
Hummingbot runtime dependencies. It is shared by the controller, the pure strategy
components (volatility / range / economics) and the offline simulator.
"""
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict


class MarketRegime(Enum):
    """Volatility regime derived from ``current_volatility / baseline_volatility``."""
    CALM = "CALM"
    NORMAL = "NORMAL"
    HIGH_VOL = "HIGH_VOL"
    EXTREME = "EXTREME"


class StrategyState(Enum):
    """
    Controller-level state.

    This is a *view* over the LP executor lifecycle (``LPExecutorStates``) plus the
    controller's own bookkeeping, not a competing state machine: the executor still
    owns the on-chain position, and every transition below is derived from executor
    state, the regime, and pending controller intents.
    """
    NO_POSITION = "NO_POSITION"      # No liquidity deployed, free to open
    ACTIVE = "ACTIVE"                # An LP position is deployed and being monitored
    REBALANCING = "REBALANCING"      # Close requested / in flight, new range not open yet
    COLLECTING = "COLLECTING"        # A collect-fees transaction is in flight
    PAUSED = "PAUSED"                # Extreme volatility (or its reopen cooldown)
    ERROR = "ERROR"                  # Executor/Gateway failure, backing off before retry


class StrategyAction(Enum):
    """The decision taken on the current control cycle."""
    HOLD = "HOLD"
    OPEN = "OPEN"
    REBALANCE = "REBALANCE"
    COLLECT_FEES = "COLLECT_FEES"
    CLOSE = "CLOSE"


class VolatilitySnapshot(BaseModel):
    """Output of :class:`VolatilityEstimator` for one control cycle."""
    samples: int
    volatility: Optional[Decimal] = None
    baseline_volatility: Optional[Decimal] = None
    volatility_ratio: Optional[Decimal] = None

    @property
    def ready(self) -> bool:
        return self.volatility is not None and self.baseline_volatility is not None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PriceRange(BaseModel):
    """
    A candidate LP range in *price* space.

    Prices - not ticks. Gateway (and the Orca Whirlpools SDK behind it) is what
    converts these bounds into tick indexes aligned to the pool's tick spacing; the
    realized, tick-aligned bounds are read back from the position afterwards.
    """
    lower_price: Decimal
    upper_price: Decimal
    width_pct: Decimal   # half-width as a fraction of price, e.g. Decimal("0.0125")

    @property
    def width(self) -> Decimal:
        return self.upper_price - self.lower_price

    def contains(self, price: Decimal) -> bool:
        return self.lower_price <= price <= self.upper_price

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PositionSnapshot(BaseModel):
    """
    Everything the strategy needs to know about the live position on one cycle.

    Built either from an LP executor's ``custom_info`` or, after a restart, from a
    Gateway ``CLMMPositionInfo`` for an adopted on-chain position.
    """
    position_address: Optional[str] = None
    current_price: Decimal
    lower_price: Decimal
    upper_price: Decimal
    lower_tick: Optional[int] = None
    upper_tick: Optional[int] = None
    base_amount: Decimal = Decimal("0")
    quote_amount: Decimal = Decimal("0")
    base_fee: Decimal = Decimal("0")
    quote_fee: Decimal = Decimal("0")
    created_at: Optional[float] = None
    adopted: bool = False

    @property
    def is_valid(self) -> bool:
        return self.upper_price > self.lower_price > Decimal("0") and self.current_price > Decimal("0")

    @property
    def range_width(self) -> Decimal:
        return self.upper_price - self.lower_price

    @property
    def distance_from_lower(self) -> Decimal:
        return self.current_price - self.lower_price

    @property
    def distance_from_upper(self) -> Decimal:
        return self.upper_price - self.current_price

    @property
    def lower_ratio(self) -> Decimal:
        """Distance to the lower bound as a fraction of the full range width."""
        if self.range_width <= 0:
            return Decimal("0")
        return self.distance_from_lower / self.range_width

    @property
    def upper_ratio(self) -> Decimal:
        """Distance to the upper bound as a fraction of the full range width."""
        if self.range_width <= 0:
            return Decimal("0")
        return self.distance_from_upper / self.range_width

    @property
    def in_range(self) -> bool:
        return self.lower_price <= self.current_price <= self.upper_price

    @property
    def liquidity_quote(self) -> Decimal:
        """Deployed liquidity valued in quote currency."""
        return self.base_amount * self.current_price + self.quote_amount

    @property
    def uncollected_fees_quote(self) -> Decimal:
        return self.base_fee * self.current_price + self.quote_fee

    model_config = ConfigDict(arbitrary_types_allowed=True)


class RebalanceEvaluation(BaseModel):
    """Result of the economic rebalance filter. Deterministic and fully loggable."""
    expected_benefit: Decimal
    estimated_cost: Decimal
    required_benefit: Decimal
    should_rebalance: bool
    reason: str
    current_range_fees: Decimal = Decimal("0")
    new_range_fees: Decimal = Decimal("0")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class FeeCollectionEvaluation(BaseModel):
    """Result of the fee-collection economic check."""
    fees_available: Decimal
    estimated_cost: Decimal
    required_fees: Decimal
    should_collect: bool
    reason: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PositionPlan(BaseModel):
    """
    A fully-costed intent to open a position, produced by the async data pass and
    consumed by the synchronous action pass.
    """
    price_range: PriceRange
    base_amount: Decimal
    quote_amount: Decimal
    reference_price: Decimal
    regime: MarketRegime
    timestamp: float

    @property
    def notional_quote(self) -> Decimal:
        return self.base_amount * self.reference_price + self.quote_amount

    model_config = ConfigDict(arbitrary_types_allowed=True)
