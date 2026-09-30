"""
Offline simulator for the Adaptive Orca LP strategy.

This is not an Orca emulator. It replays a price series through the *same* strategy
components the live controller uses - the volatility estimator, the regime detector,
the dynamic range calculator, the edge/hysteresis rules and the economic filter - and
accounts for fees and rebalance costs with the production models. Its purpose is to
answer one question:

    Does an adaptive range beat a fixed range on the same price path?

What it deliberately does not model: divergence (impermanent) loss, real swap routing,
Whirlpool tick spacing, pool composition, or the fee share of other LPs. Reported P&L
is fee income net of rebalance costs, nothing more.
"""
import math
import random
from dataclasses import dataclass, field
from decimal import Decimal
from typing import List, Optional, Sequence, Tuple

from .economic_model import EconomicRebalanceFilter, ExpectedFeeModel, RebalanceCostModel
from .models import MarketRegime, PositionSnapshot, PriceRange
from .range_calculator import DynamicRangeCalculator, RegimeDetector
from .volatility import VolatilityEstimator


@dataclass
class SimulationResult:
    """Everything the fixed-vs-adaptive comparison reports."""
    label: str
    total_steps: int = 0
    steps_in_range: int = 0
    steps_out_of_range: int = 0
    steps_without_position: int = 0
    total_fees: Decimal = Decimal("0")
    rebalance_count: int = 0
    rebalance_costs: Decimal = Decimal("0")
    regime_counts: dict = field(default_factory=dict)

    @property
    def net_return(self) -> Decimal:
        return self.total_fees - self.rebalance_costs

    @property
    def deployed_steps(self) -> int:
        return self.steps_in_range + self.steps_out_of_range

    @property
    def time_in_range_pct(self) -> Decimal:
        if self.total_steps == 0:
            return Decimal("0")
        return Decimal(self.steps_in_range) / Decimal(self.total_steps) * Decimal("100")

    @property
    def time_out_of_range_pct(self) -> Decimal:
        if self.total_steps == 0:
            return Decimal("0")
        return Decimal(self.steps_out_of_range) / Decimal(self.total_steps) * Decimal("100")


def generate_price_series(steps: int,
                          start_price: Decimal = Decimal("150"),
                          base_volatility: float = 0.0012,
                          drift: float = 0.0,
                          seed: int = 42,
                          volatility_schedule: Optional[Sequence[Tuple[int, float]]] = None) -> List[Decimal]:
    """
    Geometric Brownian motion with scheduled volatility shifts.

    ``volatility_schedule`` is a sequence of ``(start_step, multiplier)`` applied to
    ``base_volatility`` from that step onwards. The default ramps from calm through
    normal to stressed and back, which is the setting an adaptive range is supposed to
    handle better than a fixed one. It deliberately ramps rather than jumps, and a
    gradual ramp does not reach the EXTREME regime: the rolling window absorbs it
    slowly enough that the baseline keeps pace. Pass a schedule with a short, sharp
    burst (see ``--shock`` in scripts/backtest_adaptive_orca_lp.py) to exercise the
    extreme-volatility withdrawal.

    Deterministic for a given seed, so simulation runs are reproducible.
    """
    if steps < 2:
        raise ValueError("steps must be at least 2")
    if start_price <= 0:
        raise ValueError("start_price must be positive")

    if volatility_schedule is None:
        volatility_schedule = (
            (0, 0.6),
            (int(steps * 0.20), 1.0),
            (int(steps * 0.45), 2.2),
            (int(steps * 0.60), 4.5),
            (int(steps * 0.72), 1.2),
        )
    schedule = sorted(volatility_schedule, key=lambda item: item[0])

    rng = random.Random(seed)
    prices = [start_price]
    price = float(start_price)
    for step in range(1, steps):
        multiplier = 1.0
        for start_step, value in schedule:
            if step >= start_step:
                multiplier = value
        sigma = base_volatility * multiplier
        price *= math.exp(drift - 0.5 * sigma ** 2 + sigma * rng.gauss(0.0, 1.0))
        prices.append(Decimal(str(price)))
    return prices


class RangePolicy:
    """Interface shared by the fixed and adaptive policies under test."""

    name = "policy"

    def observe(self, price: Decimal) -> None:
        """Feed a new price sample (before the decision for that step)."""

    def target_range(self, price: Decimal) -> Optional[PriceRange]:
        """The range this policy would deploy at ``price``, or None if it should stay flat."""
        raise NotImplementedError

    def should_rebalance(self, position: PositionSnapshot, step: int) -> Tuple[bool, Optional[PriceRange]]:
        """Whether to move the position now, and where to."""
        raise NotImplementedError

    def should_exit(self) -> bool:
        """Whether to withdraw entirely (extreme volatility protection)."""
        return False

    @property
    def regime(self) -> MarketRegime:
        return MarketRegime.NORMAL


class FixedRangePolicy(RangePolicy):
    """
    The baseline: a constant half-width range, re-centred whenever the price leaves it.

    This is what a conventional CLMM LP bot does - it has no view on volatility, so it
    only reacts once the position has already stopped earning.
    """

    name = "fixed"

    def __init__(self, half_width_pct: Decimal = Decimal("0.01"), cooldown_steps: int = 0):
        if half_width_pct <= 0:
            raise ValueError("half_width_pct must be positive")
        self.half_width_pct = half_width_pct
        self.cooldown_steps = cooldown_steps
        self._last_rebalance_step: Optional[int] = None

    def target_range(self, price: Decimal) -> Optional[PriceRange]:
        return PriceRange(
            lower_price=price * (Decimal("1") - self.half_width_pct),
            upper_price=price * (Decimal("1") + self.half_width_pct),
            width_pct=self.half_width_pct,
        )

    def should_rebalance(self, position: PositionSnapshot, step: int) -> Tuple[bool, Optional[PriceRange]]:
        if position.in_range:
            return False, None
        if self._last_rebalance_step is not None and (step - self._last_rebalance_step) < self.cooldown_steps:
            return False, None
        self._last_rebalance_step = step
        return True, self.target_range(position.current_price)


class AdaptiveRangePolicy(RangePolicy):
    """
    The strategy under test, wired from the same components as the live controller.

    Volatility -> regime -> dynamic width, with edge confirmation, a rebalance
    cooldown, an economic filter and an extreme-volatility exit.
    """

    name = "adaptive"

    def __init__(self,
                 volatility_estimator: Optional[VolatilityEstimator] = None,
                 regime_detector: Optional[RegimeDetector] = None,
                 range_calculator: Optional[DynamicRangeCalculator] = None,
                 rebalance_filter: Optional[EconomicRebalanceFilter] = None,
                 edge_threshold: Decimal = Decimal("0.10"),
                 edge_confirmation_ticks: int = 5,
                 cooldown_steps: int = 0,
                 extreme_reopen_cooldown_steps: int = 0):
        self.volatility_estimator = volatility_estimator or VolatilityEstimator()
        self.regime_detector = regime_detector or RegimeDetector()
        self.range_calculator = range_calculator or DynamicRangeCalculator()
        self.rebalance_filter = rebalance_filter or EconomicRebalanceFilter(
            ExpectedFeeModel(), RebalanceCostModel()
        )
        self.edge_threshold = edge_threshold
        self.edge_confirmation_ticks = edge_confirmation_ticks
        self.cooldown_steps = cooldown_steps
        self.extreme_reopen_cooldown_steps = extreme_reopen_cooldown_steps

        self._regime = MarketRegime.NORMAL
        self._edge_counter = 0
        self._last_rebalance_step: Optional[int] = None
        self._extreme_exit_step: Optional[int] = None
        self._step = 0

    @property
    def regime(self) -> MarketRegime:
        return self._regime

    def observe(self, price: Decimal) -> None:
        self.volatility_estimator.add_price(price)
        self._regime = self.regime_detector.detect(self.volatility_estimator.volatility_ratio)

    def set_step(self, step: int) -> None:
        self._step = step

    def should_exit(self) -> bool:
        if self._regime is MarketRegime.EXTREME:
            self._extreme_exit_step = self._step
            self._edge_counter = 0
            return True
        return False

    def target_range(self, price: Decimal) -> Optional[PriceRange]:
        if not self.volatility_estimator.is_ready or self._regime is MarketRegime.EXTREME:
            return None
        if self._extreme_exit_step is not None:
            if (self._step - self._extreme_exit_step) < self.extreme_reopen_cooldown_steps:
                return None
            self._extreme_exit_step = None
        return self.range_calculator.calculate(
            price=price, volatility=self.volatility_estimator.volatility, regime=self._regime
        )

    def should_rebalance(self, position: PositionSnapshot, step: int) -> Tuple[bool, Optional[PriceRange]]:
        near_edge = (
            position.lower_ratio < self.edge_threshold or position.upper_ratio < self.edge_threshold
        )
        if near_edge:
            self._edge_counter += 1
        else:
            self._edge_counter = 0
            return False, None

        if self._edge_counter < self.edge_confirmation_ticks:
            return False, None
        if self._last_rebalance_step is not None and (step - self._last_rebalance_step) < self.cooldown_steps:
            return False, None

        candidate = self.target_range(position.current_price)
        if candidate is None:
            return False, None

        evaluation = self.rebalance_filter.evaluate(position, candidate)
        if not evaluation.should_rebalance:
            return False, None

        self._edge_counter = 0
        self._last_rebalance_step = step
        return True, candidate


class LPSimulator:
    """Replays a price series through a :class:`RangePolicy` and books fees and costs."""

    def __init__(self,
                 fee_model: Optional[ExpectedFeeModel] = None,
                 cost_model: Optional[RebalanceCostModel] = None,
                 step_hours: Decimal = Decimal("1") / Decimal("60"),
                 warmup_steps: int = 0):
        self.fee_model = fee_model or ExpectedFeeModel()
        self.cost_model = cost_model or RebalanceCostModel()
        self.step_hours = step_hours
        self.warmup_steps = warmup_steps

    def run(self, prices: Sequence[Decimal], policy: RangePolicy, label: Optional[str] = None) -> SimulationResult:
        result = SimulationResult(label=label or policy.name)
        current_range: Optional[PriceRange] = None

        for step, price in enumerate(prices):
            policy.observe(price)
            if hasattr(policy, "set_step"):
                policy.set_step(step)

            if step < self.warmup_steps:
                continue
            result.total_steps += 1
            regime = policy.regime.value
            result.regime_counts[regime] = result.regime_counts.get(regime, 0) + 1

            # 1. Extreme-volatility protection withdraws before anything else.
            if policy.should_exit():
                if current_range is not None:
                    current_range = None
                    result.rebalance_count += 1
                    result.rebalance_costs += self.cost_model.estimate()
                result.steps_without_position += 1
                continue

            # 2. Open when flat.
            if current_range is None:
                current_range = policy.target_range(price)
                if current_range is None:
                    result.steps_without_position += 1
                    continue
                result.rebalance_costs += self.cost_model.estimate()
                result.rebalance_count += 1

            position = PositionSnapshot(
                current_price=price,
                lower_price=current_range.lower_price,
                upper_price=current_range.upper_price,
            )

            # 3. Accrue this step's fees for the range that was live during it.
            result.total_fees += self.fee_model.expected_fees(
                price, current_range.lower_price, current_range.upper_price, forecast_hours=self.step_hours
            )
            if position.in_range:
                result.steps_in_range += 1
            else:
                result.steps_out_of_range += 1

            # 4. Decide whether to move.
            should_rebalance, new_range = policy.should_rebalance(position, step)
            if should_rebalance and new_range is not None:
                current_range = new_range
                result.rebalance_count += 1
                result.rebalance_costs += self.cost_model.estimate()

        return result


def compare(prices: Sequence[Decimal],
            fixed_policy: Optional[RangePolicy] = None,
            adaptive_policy: Optional[RangePolicy] = None,
            simulator: Optional[LPSimulator] = None) -> List[SimulationResult]:
    """Run both strategies over the same price series and return their results."""
    simulator = simulator or LPSimulator()
    fixed = fixed_policy or FixedRangePolicy()
    adaptive = adaptive_policy or AdaptiveRangePolicy()
    return [
        simulator.run(prices, fixed, label="Fixed range"),
        simulator.run(prices, adaptive, label="Adaptive range"),
    ]


def format_comparison(results: Sequence[SimulationResult], quote_token: str = "USDC") -> str:
    """Render a comparison table for the terminal."""
    headers = ["Strategy", "Fees", "Rebalances", "Costs", "Net", "In range", "Out of range"]
    rows = [[
        r.label,
        f"{float(r.total_fees):.2f}",
        str(r.rebalance_count),
        f"{float(r.rebalance_costs):.2f}",
        f"{float(r.net_return):.2f}",
        f"{float(r.time_in_range_pct):.1f}%",
        f"{float(r.time_out_of_range_pct):.1f}%",
    ] for r in results]

    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]

    def render(cells):
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

    lines = [
        f"Fees, costs and net are in {quote_token}.",
        render(headers),
        "  ".join("-" * w for w in widths),
    ]
    lines.extend(render(row) for row in rows)
    return "\n".join(lines)
