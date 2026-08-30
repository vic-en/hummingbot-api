"""
Economic models for the Adaptive Orca LP controller.

Two decisions live here and nowhere else:

1. Is re-centering the range worth the transactions it costs?
2. Are the uncollected fees worth a collect-fees transaction?

Both are deliberately simple, deterministic and isolated so they can be replaced by
better estimators without touching the controller.
"""
from decimal import Decimal
from typing import Dict, Optional

from .models import FeeCollectionEvaluation, PositionSnapshot, PriceRange, RebalanceEvaluation


class ExpectedFeeModel:
    """
    A first-order estimate of the fees a CLMM position earns over a forecast horizon.

        utilization    = clamp(2 * distance_to_nearest_boundary / range_width, 0, 1)
        concentration  = clamp(reference_range_pct / half_width_pct, 0, cap)
        expected_fees  = expected_hourly_volume * pool_fee_rate * forecast_hours
                         * utilization * concentration

    Two notes on what this is and is not.

    ``utilization`` is the complement of the raw "1 - distance/width" form: that form
    peaks when the price sits *on* a boundary, which is precisely when a CLMM
    position has stopped earning and is fully converted into one token. Orienting it
    the other way makes it 1.0 at the centre of the range and 0.0 at the boundary,
    which is what a fee-capture weight has to look like. It is a crude proxy for
    "how much of the horizon will be spent in range", not a probability.

    ``concentration`` captures the whole point of a CLMM: the same capital spread
    over a narrower range is denser liquidity and takes a larger share of the swap
    flow that crosses it. Without it a wider range would dominate on every metric and
    the adaptive/fixed comparison would be meaningless. It is normalised against
    ``reference_range_pct``, the half-width at which ``expected_hourly_volume`` is
    quoted, and capped so a degenerate range cannot produce an unbounded benefit.

    ``expected_hourly_volume`` is the volume expected to route *through this
    position's range* per hour, already net of the position's share of the pool - not
    the pool's total volume. It must be calibrated per pool and position size; the
    default is a placeholder, not a measurement.
    """

    def __init__(self,
                 pool_fee_rate: Decimal = Decimal("0.003"),
                 expected_hourly_volume: Decimal = Decimal("100000"),
                 forecast_hours: Decimal = Decimal("1"),
                 reference_range_pct: Decimal = Decimal("0.01"),
                 max_concentration_multiplier: Decimal = Decimal("10")):
        if pool_fee_rate < 0:
            raise ValueError("pool_fee_rate cannot be negative")
        if expected_hourly_volume < 0:
            raise ValueError("expected_hourly_volume cannot be negative")
        if forecast_hours <= 0:
            raise ValueError("forecast_hours must be positive")
        if reference_range_pct <= 0:
            raise ValueError("reference_range_pct must be positive")
        if max_concentration_multiplier <= 0:
            raise ValueError("max_concentration_multiplier must be positive")

        self.pool_fee_rate = pool_fee_rate
        self.expected_hourly_volume = expected_hourly_volume
        self.forecast_hours = forecast_hours
        self.reference_range_pct = reference_range_pct
        self.max_concentration_multiplier = max_concentration_multiplier

    @staticmethod
    def utilization(price: Decimal, lower_price: Decimal, upper_price: Decimal) -> Decimal:
        """1.0 with the price centred in the range, 0.0 at (or outside) a boundary."""
        range_width = upper_price - lower_price
        if range_width <= 0:
            return Decimal("0")
        distance_to_nearest = min(price - lower_price, upper_price - price)
        ratio = Decimal("2") * distance_to_nearest / range_width
        if ratio <= 0:
            return Decimal("0")
        if ratio >= 1:
            return Decimal("1")
        return ratio

    def concentration(self, half_width_pct: Decimal) -> Decimal:
        """Liquidity density relative to a ``reference_range_pct``-wide position."""
        if half_width_pct is None or half_width_pct <= 0:
            return Decimal("0")
        factor = self.reference_range_pct / half_width_pct
        return min(factor, self.max_concentration_multiplier)

    @staticmethod
    def half_width_pct(price: Decimal, lower_price: Decimal, upper_price: Decimal) -> Decimal:
        """Half-width of a realized range as a fraction of the current price."""
        if price <= 0:
            return Decimal("0")
        return (upper_price - lower_price) / (Decimal("2") * price)

    def expected_fees(self,
                      price: Decimal,
                      lower_price: Decimal,
                      upper_price: Decimal,
                      forecast_hours: Optional[Decimal] = None) -> Decimal:
        """Expected fees in quote currency over the forecast horizon."""
        if price is None or price <= 0 or upper_price <= lower_price:
            return Decimal("0")

        hours = self.forecast_hours if forecast_hours is None else forecast_hours
        utilization = self.utilization(price, lower_price, upper_price)
        concentration = self.concentration(self.half_width_pct(price, lower_price, upper_price))

        expected_volume = self.expected_hourly_volume * utilization * concentration
        return expected_volume * self.pool_fee_rate * hours

    def expected_fees_for_range(self,
                                price: Decimal,
                                price_range: PriceRange,
                                forecast_hours: Optional[Decimal] = None) -> Decimal:
        return self.expected_fees(price, price_range.lower_price, price_range.upper_price, forecast_hours)


class RebalanceCostModel:
    """
    Flat, configurable estimate of what one rebalance costs in quote currency.

    A rebalance is close + (optional) swap + open, so the components are:

    - ``tx_cost``:       chain fees and rent for the close/open transactions
    - ``swap_cost``:     the swap fee paid re-balancing the token mix for the new range
    - ``slippage_cost``: price impact of that swap

    They are flat quote amounts rather than percentages because on Solana the chain
    cost genuinely is flat, and because a flat floor is the conservative choice for a
    small position: it never under-charges a rebalance the way a percentage would.
    """

    def __init__(self,
                 tx_cost: Decimal = Decimal("0.02"),
                 swap_cost: Decimal = Decimal("0.50"),
                 slippage_cost: Decimal = Decimal("0.10")):
        for name, value in (("tx_cost", tx_cost), ("swap_cost", swap_cost), ("slippage_cost", slippage_cost)):
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        self.tx_cost = tx_cost
        self.swap_cost = swap_cost
        self.slippage_cost = slippage_cost

    def estimate(self) -> Decimal:
        return self.tx_cost + self.swap_cost + self.slippage_cost

    def breakdown(self) -> Dict[str, Decimal]:
        return {
            "tx_cost": self.tx_cost,
            "swap_cost": self.swap_cost,
            "slippage_cost": self.slippage_cost,
            "total": self.estimate(),
        }


class EconomicRebalanceFilter:
    """
    The gate that stops the strategy from trading itself to death.

    Reaching the edge of the range is *necessary* for a rebalance but never
    sufficient: the position is only moved when the extra fees the re-centred range
    is expected to earn over the forecast horizon exceed the cost of moving it by
    ``min_profit_multiple``.
    """

    def __init__(self,
                 fee_model: ExpectedFeeModel,
                 cost_model: RebalanceCostModel,
                 min_profit_multiple: Decimal = Decimal("2.0")):
        if min_profit_multiple < 0:
            raise ValueError("min_profit_multiple cannot be negative")
        self.fee_model = fee_model
        self.cost_model = cost_model
        self.min_profit_multiple = min_profit_multiple

    def evaluate(self,
                 position: PositionSnapshot,
                 candidate_range: PriceRange) -> RebalanceEvaluation:
        cost = self.cost_model.estimate()
        required = cost * self.min_profit_multiple

        current_fees = self.fee_model.expected_fees(
            position.current_price, position.lower_price, position.upper_price
        )
        new_fees = self.fee_model.expected_fees_for_range(position.current_price, candidate_range)

        # The benefit of rebalancing is the *incremental* fee expectation, floored at
        # zero: a candidate range that is worse than the one already deployed is never
        # a reason to pay for a rebalance.
        benefit = new_fees - current_fees
        if benefit < 0:
            benefit = Decimal("0")

        should = benefit > required
        if should:
            reason = (
                f"expected benefit {benefit:.4f} > required {required:.4f} "
                f"(cost {cost:.4f} x {self.min_profit_multiple})"
            )
        else:
            reason = (
                f"expected benefit {benefit:.4f} <= required {required:.4f} "
                f"(cost {cost:.4f} x {self.min_profit_multiple}) - holding"
            )

        return RebalanceEvaluation(
            expected_benefit=benefit,
            estimated_cost=cost,
            required_benefit=required,
            should_rebalance=should,
            reason=reason,
            current_range_fees=current_fees,
            new_range_fees=new_fees,
        )


class FeeCollectionPolicy:
    """
    Decides whether uncollected fees justify a collect-fees transaction.

    Collect only when ``fees_available > collection_cost * fee_profit_multiple``.
    """

    def __init__(self,
                 collection_cost: Decimal = Decimal("0.02"),
                 fee_profit_multiple: Decimal = Decimal("2.0")):
        if collection_cost < 0:
            raise ValueError("collection_cost cannot be negative")
        if fee_profit_multiple < 0:
            raise ValueError("fee_profit_multiple cannot be negative")
        self.collection_cost = collection_cost
        self.fee_profit_multiple = fee_profit_multiple

    def evaluate(self, fees_available: Decimal) -> FeeCollectionEvaluation:
        if fees_available is None or fees_available < 0:
            fees_available = Decimal("0")
        required = self.collection_cost * self.fee_profit_multiple
        should = fees_available > required
        reason = (
            f"fees {fees_available:.6f} > required {required:.6f} - collecting"
            if should else
            f"fees {fees_available:.6f} <= required {required:.6f} - accruing"
        )
        return FeeCollectionEvaluation(
            fees_available=fees_available,
            estimated_cost=self.collection_cost,
            required_fees=required,
            should_collect=should,
            reason=reason,
        )
