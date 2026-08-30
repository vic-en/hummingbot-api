"""
Adaptive Orca LP - a volatility-aware concentrated-liquidity controller.

The controller owns the *strategy*: how wide the range should be, when the price is
close enough to a boundary to matter, and whether moving the position pays for
itself. Every on-chain action is delegated to Hummingbot's existing
:class:`~hummingbot.strategy_v2.executors.lp_executor.lp_executor.LPExecutor` and the
Gateway connector, which is also what converts price bounds into tick-spacing-aligned
Whirlpool ticks.

Config and controller live in the same module because
``ControllerConfigBase.get_controller_class()`` resolves the controller by inspecting
``config.__module__``.
"""
import logging
import math
from decimal import ROUND_DOWN, Decimal
from typing import Any, Dict, List, Optional

from hummingbot.core.data_type.common import MarketDict, TradeType
from hummingbot.core.gateway.gateway_http_client import GatewayHttpClient
from hummingbot.core.utils.async_utils import safe_ensure_future
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.logger import HummingbotLogger
from hummingbot.strategy_v2.controllers import ControllerBase, ControllerConfigBase
from hummingbot.strategy_v2.executors.data_types import ConnectorPair
from hummingbot.strategy_v2.executors.gateway_utils import parse_provider
from hummingbot.strategy_v2.executors.lp_executor.data_types import LPExecutorConfig
from hummingbot.strategy_v2.executors.order_executor.data_types import ExecutionStrategy, OrderExecutorConfig
from hummingbot.strategy_v2.models.base import RunnableStatus
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, ExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.models.executors import CloseType
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo
from pydantic import Field, field_validator, model_validator

from .economic_model import EconomicRebalanceFilter, ExpectedFeeModel, FeeCollectionPolicy, RebalanceCostModel
from .models import (
    FeeCollectionEvaluation,
    MarketRegime,
    PositionPlan,
    PositionSnapshot,
    PriceRange,
    RebalanceEvaluation,
    StrategyAction,
    StrategyState,
)
from .range_calculator import DynamicRangeCalculator, RegimeDetector
from .volatility import VolatilityEstimator

# LP executor states in which an on-chain transaction is already in flight. The
# controller must never issue a competing instruction while one of these is active.
TRANSIENT_EXECUTOR_STATES = ("OPENING", "CLOSING", "SWAPPING")
# LP executor states in which a position is actually deployed.
DEPLOYED_EXECUTOR_STATES = ("IN_RANGE", "OUT_OF_RANGE")
# How long to wait for a created executor to show up in ``executors_info`` before
# assuming the create was lost. Guards against opening a second position while the
# first request is still in flight.
OPEN_REQUEST_TIMEOUT = 60.0
# A price parked just past the edge threshold would otherwise re-log an identical
# rejected evaluation on every sample. Approvals are always logged.
EVALUATION_LOG_INTERVAL = 60.0


class AdaptiveOrcaLPConfig(ControllerConfigBase):
    """
    Configuration for the Adaptive Orca LP controller.

    Provider architecture matches the rest of the V2 LP stack:
    - ``connector_name`` is the *network* (e.g. "solana-mainnet-beta")
    - ``lp_provider`` is "dex/trading_type" (e.g. "orca/clmm")
    """
    controller_type: str = "generic"
    controller_name: str = "adaptive_orca_lp"
    candles_config: List[CandlesConfig] = []

    # ------------------------------------------------------------------ market
    connector_name: str = Field(
        default="solana-mainnet-beta",
        description="Gateway network connector, e.g. 'solana-mainnet-beta'",
    )
    lp_provider: str = Field(
        default="orca/clmm",
        description="LP provider as 'dex/trading_type', e.g. 'orca/clmm'",
    )
    trading_pair: str = Field(default="SOL-USDC", description="Pool trading pair, e.g. 'SOL-USDC'")
    pool_address: str = Field(
        default="",
        description=(
            "Whirlpool address. Leave empty to resolve it from the trading pair via "
            "Gateway's pool lookup on the first control cycle."
        ),
    )

    # ``total_amount_quote`` is inherited from ControllerConfigBase and is the
    # "capital_quote" of the strategy spec: the notional deployed into the position.
    total_amount_quote: Decimal = Field(
        default=Decimal("1000"),
        json_schema_extra={
            "prompt": "Total capital in quote asset to deploy as liquidity (e.g. 1000): ",
            "prompt_on_new": True,
            "is_updatable": True,
        },
    )

    # ------------------------------------------------------------- volatility
    volatility_window: int = Field(
        default=120,
        json_schema_extra={"is_updatable": False},
        description="Number of log returns held by the rolling volatility estimator",
    )
    volatility_min_samples: int = Field(
        default=30,
        json_schema_extra={"is_updatable": False},
        description="Minimum log returns before volatility is considered usable",
    )
    volatility_sample_interval: Decimal = Field(
        default=Decimal("5"),
        json_schema_extra={"is_updatable": True},
        description=(
            "Seconds between volatility samples. Decoupled from the control interval so the "
            "window length is a wall-clock horizon rather than a function of tick rate."
        ),
    )
    baseline_alpha: Decimal = Field(
        default=Decimal("0.002"),
        json_schema_extra={"is_updatable": False},
        description=(
            "EWMA weight applied to the newest volatility reading for the baseline. Must be well "
            "below 1/volatility_window or the baseline tracks the current reading and the regime "
            "detector never fires; 0.002 gives a ~346-reading half-life against a 120-sample window."
        ),
    )

    # ------------------------------------------------------------------ range
    z_score: Decimal = Field(default=Decimal("2.0"), json_schema_extra={"is_updatable": True})
    min_range_pct: Decimal = Field(
        default=Decimal("0.0025"),
        json_schema_extra={"is_updatable": True},
        description="Floor for the range half-width as a fraction of price",
    )
    max_range_pct: Decimal = Field(
        default=Decimal("0.05"),
        json_schema_extra={"is_updatable": True},
        description="Ceiling for the range half-width as a fraction of price",
    )
    calm_multiplier: Decimal = Field(default=Decimal("0.70"), json_schema_extra={"is_updatable": True})
    normal_multiplier: Decimal = Field(default=Decimal("1.00"), json_schema_extra={"is_updatable": True})
    high_vol_multiplier: Decimal = Field(default=Decimal("1.75"), json_schema_extra={"is_updatable": True})
    extreme_multiplier: Decimal = Field(default=Decimal("3.00"), json_schema_extra={"is_updatable": True})

    # ----------------------------------------------------------------- regimes
    calm_threshold: Decimal = Field(default=Decimal("0.75"), json_schema_extra={"is_updatable": True})
    high_vol_threshold: Decimal = Field(default=Decimal("1.50"), json_schema_extra={"is_updatable": True})
    extreme_vol_threshold: Decimal = Field(default=Decimal("2.50"), json_schema_extra={"is_updatable": True})

    # --------------------------------------------------------------- boundary
    edge_threshold: Decimal = Field(
        default=Decimal("0.10"),
        json_schema_extra={"is_updatable": True},
        description="Position is 'near edge' when either boundary ratio falls below this",
    )
    edge_confirmation_ticks: int = Field(
        default=5,
        json_schema_extra={"is_updatable": True},
        description="Consecutive control cycles the edge condition must hold before a rebalance is considered",
    )
    rebalance_cooldown: Decimal = Field(
        default=Decimal("900"),
        json_schema_extra={"is_updatable": True},
        description="Minimum seconds between rebalances",
    )

    # -------------------------------------------------------------- economics
    estimated_tx_cost: Decimal = Field(default=Decimal("0.02"), json_schema_extra={"is_updatable": True})
    estimated_swap_cost: Decimal = Field(default=Decimal("0.50"), json_schema_extra={"is_updatable": True})
    estimated_slippage: Decimal = Field(default=Decimal("0.10"), json_schema_extra={"is_updatable": True})
    min_profit_multiple: Decimal = Field(
        default=Decimal("2.0"),
        json_schema_extra={"is_updatable": True},
        description="Expected benefit must exceed estimated cost by this multiple to rebalance",
    )

    pool_fee_rate: Decimal = Field(default=Decimal("0.003"), json_schema_extra={"is_updatable": True})
    expected_hourly_volume: Decimal = Field(
        default=Decimal("100000"),
        json_schema_extra={"is_updatable": True},
        description=(
            "Quote-currency volume expected to route through this position's range per hour, "
            "already net of the position's share of the pool. Calibrate per pool."
        ),
    )
    forecast_hours: Decimal = Field(default=Decimal("1"), json_schema_extra={"is_updatable": True})
    fee_reference_range_pct: Decimal = Field(
        default=Decimal("0.01"),
        json_schema_extra={"is_updatable": True},
        description="Half-width at which expected_hourly_volume is quoted; sets the concentration baseline",
    )
    max_concentration_multiplier: Decimal = Field(
        default=Decimal("10"),
        json_schema_extra={"is_updatable": True},
        description="Cap on the concentration factor so a degenerate range cannot imply unbounded fees",
    )

    # --------------------------------------------------------- fee collection
    enable_fee_collection: bool = Field(
        default=True,
        json_schema_extra={"is_updatable": True},
        description=(
            "Harvest fees from the live position between rebalances. Fees are always collected "
            "when the position closes, regardless of this setting."
        ),
    )
    estimated_fee_collection_cost: Decimal = Field(default=Decimal("0.02"), json_schema_extra={"is_updatable": True})
    fee_profit_multiple: Decimal = Field(default=Decimal("2.0"), json_schema_extra={"is_updatable": True})
    fee_collection_cooldown: Decimal = Field(
        default=Decimal("300"),
        json_schema_extra={"is_updatable": True},
        description="Minimum seconds between collect-fees transactions",
    )

    # ------------------------------------------------------------- protection
    extreme_vol_reopen_cooldown: Decimal = Field(
        default=Decimal("1800"),
        json_schema_extra={"is_updatable": True},
        description="Seconds to stay flat after exiting on extreme volatility",
    )
    safety_exit_pct: Decimal = Field(
        default=Decimal("0.02"),
        json_schema_extra={"is_updatable": True},
        description=(
            "Executor-level fail-safe: the LP executor auto-closes if price moves this far beyond "
            "the position bounds, covering the case where the controller stops ticking. "
            "Set to 0 to disable."
        ),
    )
    max_consecutive_failures: int = Field(
        default=5,
        json_schema_extra={"is_updatable": True},
        description="Consecutive executor failures after which the controller stops retrying",
    )
    error_backoff_seconds: Decimal = Field(
        default=Decimal("300"),
        json_schema_extra={"is_updatable": True},
        description="Base backoff after an executor failure; multiplied by the consecutive failure count",
    )
    autoswap: bool = Field(
        default=True,
        json_schema_extra={"is_updatable": True},
        description=(
            "Swap the wallet into the token mix a new range needs. A re-centring CLMM strategy "
            "converts toward one token on every move, so without this it stalls on 'insufficient "
            "balance' exactly when a rebalance is due. Disable only if you rebalance the wallet yourself."
        ),
    )
    swap_buffer_pct: Decimal = Field(
        default=Decimal("1"),
        json_schema_extra={"is_updatable": True},
        description="Extra percent swapped beyond the deficit to absorb slippage (1 = 1%)",
    )
    max_consecutive_swap_failures: int = Field(
        default=3,
        json_schema_extra={"is_updatable": True},
        description="Failed autoswaps before autoswap gives up and the controller just holds",
    )
    slippage_pct: Decimal = Field(
        default=Decimal("1"),
        json_schema_extra={"is_updatable": True},
        description="Slippage percent passed to Gateway when quoting position amounts",
    )
    min_position_notional_quote: Decimal = Field(
        default=Decimal("5"),
        json_schema_extra={"is_updatable": True},
        description="Refuse to open a position smaller than this notional",
    )
    status_log_interval: Decimal = Field(
        default=Decimal("30"),
        json_schema_extra={"is_updatable": True},
        description="Seconds between periodic status log lines (0 disables)",
    )

    @field_validator("volatility_window", "volatility_min_samples", "edge_confirmation_ticks",
                     "max_consecutive_failures", mode="before")
    @classmethod
    def validate_positive_int(cls, v, info):
        value = int(v)
        if value < 1:
            raise ValueError(f"{info.field_name} must be at least 1")
        return value

    @model_validator(mode="after")
    def validate_strategy_parameters(self):
        if self.min_range_pct <= 0:
            raise ValueError("min_range_pct must be positive")
        if self.max_range_pct <= self.min_range_pct:
            raise ValueError("max_range_pct must be greater than min_range_pct")
        if self.max_range_pct >= 1:
            raise ValueError("max_range_pct must be below 1")
        if not (self.calm_threshold < self.high_vol_threshold < self.extreme_vol_threshold):
            raise ValueError("regime thresholds must satisfy calm < high_vol < extreme")
        if not (Decimal("0") < self.edge_threshold < Decimal("0.5")):
            raise ValueError("edge_threshold must be in (0, 0.5)")
        if not (Decimal("0") < self.baseline_alpha <= Decimal("1")):
            raise ValueError("baseline_alpha must be in (0, 1]")
        if self.volatility_min_samples > self.volatility_window:
            raise ValueError("volatility_min_samples cannot exceed volatility_window")
        if self.volatility_window < 2:
            raise ValueError("volatility_window must be at least 2")
        if self.volatility_sample_interval <= 0:
            raise ValueError("volatility_sample_interval must be positive")
        if self.z_score <= 0:
            raise ValueError("z_score must be positive")
        if self.safety_exit_pct < 0:
            raise ValueError("safety_exit_pct cannot be negative")
        return self

    def update_markets(self, markets: MarketDict) -> MarketDict:
        return markets.add_or_update(self.connector_name, self.trading_pair)


class AdaptiveOrcaLP(ControllerBase):
    """
    Volatility-adaptive single-position CLMM liquidity manager.

    Cycle:
        pool price -> volatility -> regime -> dynamic range -> economic filter -> LP executor

    The controller holds at most one position at a time. Liquidity is deployed and
    withdrawn exclusively through ``LPExecutor``; the only direct Gateway calls are
    read-only pool/position queries, the quote used to size a position, and the
    optional collect-fees transaction (for which no executor path exists).
    """

    _logger: Optional[HummingbotLogger] = None

    @classmethod
    def logger(cls) -> HummingbotLogger:
        if cls._logger is None:
            cls._logger = logging.getLogger(__name__)
        return cls._logger

    def __init__(self, config: AdaptiveOrcaLPConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config: AdaptiveOrcaLPConfig = config

        self.lp_dex_name, self.lp_trading_type = parse_provider(config.lp_provider, default_trading_type="clmm")
        parts = config.trading_pair.split("-")
        self._base_token: str = parts[0] if len(parts) == 2 else ""
        self._quote_token: str = parts[1] if len(parts) == 2 else ""

        # The estimator owns accumulated history, so it is built once and never
        # rebuilt; its window and alpha are deliberately not hot-updatable.
        self.volatility_estimator = VolatilityEstimator(
            window=config.volatility_window,
            baseline_alpha=config.baseline_alpha,
            min_samples=config.volatility_min_samples,
        )
        self._build_strategy_components()

        # ------------------------------------------------------- market state
        self._pool_address: str = config.pool_address
        self._pool_price: Optional[Decimal] = None
        self._pool_fee_pct: Optional[Decimal] = None
        self._last_sample_timestamp: float = 0.0
        self._regime: MarketRegime = MarketRegime.NORMAL
        self._previous_regime: Optional[MarketRegime] = None

        # ---------------------------------------------------- position state
        self._current_executor_id: Optional[str] = None
        self._rebalance_requested_for: Optional[str] = None
        self._position: Optional[PositionSnapshot] = None
        self._position_opened_at: Optional[float] = None
        self._edge_counter: int = 0
        self._last_rebalance_timestamp: float = 0.0
        self._rebalance_count: int = 0
        self._position_plan: Optional[PositionPlan] = None
        self._open_requested_at: Optional[float] = None

        # ---------------------------------------------------------- autoswap
        self._swap_executor_id: Optional[str] = None
        self._swap_requested_at: Optional[float] = None
        self._consecutive_swap_failures: int = 0

        # --------------------------------------------------- recovery/adoption
        self._recovery_done: bool = False
        self._adopted_position_address: Optional[str] = None
        self._adopted_close_in_flight: bool = False
        self._extra_position_addresses: List[str] = []

        # ------------------------------------------------------ fee collection
        self._fee_collection_in_flight: bool = False
        self._last_fee_collection_timestamp: float = 0.0
        self._collected_fees_quote: Decimal = Decimal("0")

        # ------------------------------------------------------------- safety
        self._extreme_exit_timestamp: Optional[float] = None
        self._consecutive_failures: int = 0
        self._error_until: float = 0.0
        self._halted: bool = False

        # ---------------------------------------------------------- reporting
        self._state: StrategyState = StrategyState.NO_POSITION
        self._last_action: StrategyAction = StrategyAction.HOLD
        self._last_reason: str = "starting up"
        self._last_status_log: float = 0.0
        self._last_evaluation_log: float = 0.0
        self._insufficient_data_logged: bool = False

        self.market_data_provider.initialize_rate_sources([
            ConnectorPair(connector_name=config.connector_name, trading_pair=config.trading_pair)
        ])

        self.logger().info(
            f"Adaptive Orca LP starting: pair={config.trading_pair} provider={config.lp_provider} "
            f"network={config.connector_name} pool={config.pool_address or '(resolve from pair)'} "
            f"capital={config.total_amount_quote} {self._quote_token} "
            f"vol_window={config.volatility_window} z={config.z_score} "
            f"range=[{config.min_range_pct}, {config.max_range_pct}]"
        )
        self._log_economics_calibration()

    def _log_economics_calibration(self) -> None:
        """
        Surface whether the fee model and the cost model live on the same scale.

        ``expected_hourly_volume`` is a per-pool, per-position quantity with no
        universally correct default. If it is left far too high the economic filter
        approves every rebalance and Phase-5 protection silently stops existing, so
        the ratio is stated once at startup rather than left to be discovered live.
        """
        max_hourly_fee = self.config.expected_hourly_volume * self.config.pool_fee_rate
        rebalance_cost = self.cost_model.estimate()
        self.logger().info(
            f"Economics calibration: max fees at full utilization = "
            f"{self._fmt(max_hourly_fee)} {self._quote_token}/h "
            f"(volume {self.config.expected_hourly_volume} x fee rate {self.config.pool_fee_rate}); "
            f"rebalance cost = {self._fmt(rebalance_cost)} {self._quote_token}; "
            f"required benefit = {self._fmt(rebalance_cost * self.config.min_profit_multiple)}"
        )
        if rebalance_cost > 0 and max_hourly_fee > rebalance_cost * Decimal("100"):
            self.logger().warning(
                f"expected_hourly_volume={self.config.expected_hourly_volume} implies hourly fees "
                f"~{self._fmt(max_hourly_fee)} {self._quote_token} against a "
                f"{self._fmt(rebalance_cost)} {self._quote_token} rebalance cost. At that ratio the "
                f"economic filter will approve essentially every rebalance. Calibrate "
                f"expected_hourly_volume to the volume actually routed through THIS position's range "
                f"(pool volume x your liquidity share), not the pool's total volume."
            )

    def _build_strategy_components(self) -> None:
        """
        (Re)build every stateless strategy component from the current config.

        Called from ``__init__`` and again from ``update_config`` so that the
        parameters marked updatable - regime thresholds, range multipliers, costs,
        profit multiples - actually take effect at runtime instead of being frozen
        into objects built once at startup.
        """
        config = self.config
        self.regime_detector = RegimeDetector(
            calm_threshold=config.calm_threshold,
            high_vol_threshold=config.high_vol_threshold,
            extreme_threshold=config.extreme_vol_threshold,
        )
        self.range_calculator = DynamicRangeCalculator(
            z_score=config.z_score,
            min_range_pct=config.min_range_pct,
            max_range_pct=config.max_range_pct,
            calm_multiplier=config.calm_multiplier,
            normal_multiplier=config.normal_multiplier,
            high_vol_multiplier=config.high_vol_multiplier,
            extreme_multiplier=config.extreme_multiplier,
        )
        self.fee_model = ExpectedFeeModel(
            pool_fee_rate=config.pool_fee_rate,
            expected_hourly_volume=config.expected_hourly_volume,
            forecast_hours=config.forecast_hours,
            reference_range_pct=config.fee_reference_range_pct,
            max_concentration_multiplier=config.max_concentration_multiplier,
        )
        self.cost_model = RebalanceCostModel(
            tx_cost=config.estimated_tx_cost,
            swap_cost=config.estimated_swap_cost,
            slippage_cost=config.estimated_slippage,
        )
        self.rebalance_filter = EconomicRebalanceFilter(
            fee_model=self.fee_model,
            cost_model=self.cost_model,
            min_profit_multiple=config.min_profit_multiple,
        )
        self.fee_collection_policy = FeeCollectionPolicy(
            collection_cost=config.estimated_fee_collection_cost,
            fee_profit_multiple=config.fee_profit_multiple,
        )

    def update_config(self, new_config: ControllerConfigBase):
        super().update_config(new_config)
        self._build_strategy_components()

    # =====================================================================
    # Async data pass
    # =====================================================================

    async def update_processed_data(self):
        now = self.market_data_provider.time()

        await self._ensure_pool_address()
        price_ok = await self._update_pool_state()
        sampled = self._sample_volatility(now) if price_ok else False
        self._detect_regime()

        await self._recover_existing_positions()

        self._position = await self._build_position_snapshot(now)
        if sampled:
            self._update_edge_tracking(self._position)

        candidate_range = self._candidate_range()
        rebalance_eval = self._evaluate_rebalance(candidate_range)
        fee_eval = self._evaluate_fee_collection()

        await self._maintain_position_plan(candidate_range, now)

        self.processed_data = self._build_processed_data(
            now=now, candidate_range=candidate_range, rebalance_eval=rebalance_eval, fee_eval=fee_eval
        )

    async def _ensure_pool_address(self) -> None:
        """Resolve the Whirlpool address from the trading pair when it was left blank."""
        if self._pool_address:
            return
        connector = self._connector()
        if connector is None:
            return
        try:
            pool_address = await connector.get_pool_address(
                self.config.trading_pair, dex_name=self.lp_dex_name, trading_type=self.lp_trading_type
            )
        except Exception as e:                                       # pragma: no cover - network path
            self.logger().warning(f"Could not resolve pool address for {self.config.trading_pair}: {e}")
            return
        if pool_address:
            self._pool_address = pool_address
            self.logger().info(f"Resolved {self.config.trading_pair} {self.config.lp_provider} pool: {pool_address}")

    async def _update_pool_state(self) -> bool:
        """Refresh the pool price. Returns False when the price is missing or unusable."""
        connector = self._connector()
        if connector is None or not self._pool_address:
            return False
        try:
            pool_info = await connector.get_pool_info_by_address(
                self._pool_address, dex_name=self.lp_dex_name, trading_type=self.lp_trading_type
            )
        except Exception as e:                                       # pragma: no cover - network path
            self.logger().warning(f"Could not fetch pool info for {self._pool_address}: {e}")
            return False

        if pool_info is None:
            self.logger().warning(f"Gateway returned no pool info for {self._pool_address}")
            return False

        price = self._to_decimal(getattr(pool_info, "price", None))
        if price is None or not price.is_finite() or price <= 0:
            self.logger().warning(f"Ignoring invalid pool price {getattr(pool_info, 'price', None)!r}")
            return False

        self._pool_price = price
        fee_pct = self._to_decimal(getattr(pool_info, "fee_pct", None))
        if fee_pct is not None and fee_pct >= 0:
            self._pool_fee_pct = fee_pct
        return True

    def _sample_volatility(self, now: float) -> bool:
        """Feed the estimator at most once per ``volatility_sample_interval``."""
        if self._pool_price is None:
            return False
        interval = float(self.config.volatility_sample_interval)
        if self._last_sample_timestamp and (now - self._last_sample_timestamp) < interval:
            return False

        self._last_sample_timestamp = now
        self.volatility_estimator.add_price(self._pool_price)

        if not self.volatility_estimator.is_ready:
            if not self._insufficient_data_logged:
                self.logger().info(
                    f"Insufficient data for volatility: {self.volatility_estimator.sample_count}/"
                    f"{self.config.volatility_min_samples} samples "
                    f"(~{self.config.volatility_min_samples * interval:.0f}s of history needed)"
                )
                self._insufficient_data_logged = True
            return True

        if self._insufficient_data_logged:
            self.logger().info(
                f"Volatility estimator ready: {self.volatility_estimator.sample_count} samples, "
                f"volatility={self.volatility_estimator.volatility}"
            )
            self._insufficient_data_logged = False
        return True

    def _detect_regime(self) -> MarketRegime:
        regime = self.regime_detector.detect(self.volatility_estimator.volatility_ratio)
        if self.volatility_estimator.is_ready and regime is not self._previous_regime:
            if self._previous_regime is not None:
                self.logger().info(
                    f"REGIME CHANGE: {self._previous_regime.value} -> {regime.value} "
                    f"(volatility={self._fmt(self.volatility_estimator.volatility)}, "
                    f"baseline={self._fmt(self.volatility_estimator.baseline_volatility)}, "
                    f"ratio={self._fmt(self.volatility_estimator.volatility_ratio, 2)})"
                )
            self._previous_regime = regime
        self._regime = regime
        return regime

    def _candidate_range(self) -> Optional[PriceRange]:
        """The range the strategy would deploy right now, or None when it cannot."""
        if self._pool_price is None or not self.volatility_estimator.is_ready:
            return None
        return self.range_calculator.calculate(
            price=self._pool_price,
            volatility=self.volatility_estimator.volatility,
            regime=self._regime,
        )

    # =====================================================================
    # Restart / recovery
    # =====================================================================

    async def _recover_existing_positions(self) -> None:
        """
        Adopt an on-chain position left behind by a previous run.

        Runs once, before any liquidity may be deployed. ``determine_executor_actions``
        refuses to open anything until this has succeeded, so a restart can never
        stack a second position on top of one that is already open. A failed query
        leaves the flag unset so the next cycle retries rather than assuming "none".
        """
        if self._recovery_done:
            return
        connector = self._connector()
        if connector is None or not self._pool_address:
            return

        executor_positions = {
            info.custom_info.get("position_address")
            for info in self.executors_info
            if getattr(info.config, "type", None) == "lp_executor" and info.custom_info.get("position_address")
        }

        try:
            positions = await connector.get_user_positions(
                dex_name=self.lp_dex_name,
                trading_type=self.lp_trading_type,
                pool_address=self._pool_address,
            )
        except Exception as e:                                       # pragma: no cover - network path
            self.logger().warning(f"Position recovery query failed, will retry: {e}")
            return

        self._recovery_done = True
        candidates = sorted(
            [p for p in (positions or []) if getattr(p, "address", None) and p.address not in executor_positions],
            key=lambda p: p.address,
        )

        if not candidates:
            self.logger().info(
                f"Recovery: no unmanaged {self.config.trading_pair} positions on pool {self._pool_address}"
            )
            return

        adopted = candidates[0]
        self._adopted_position_address = adopted.address
        self._extra_position_addresses = [p.address for p in candidates[1:]]
        self.logger().info(
            f"Recovery: adopting existing position {adopted.address} "
            f"[{adopted.lower_price} - {adopted.upper_price}] "
            f"base={adopted.base_token_amount} quote={adopted.quote_token_amount}"
        )
        if self._extra_position_addresses:
            self.logger().warning(
                f"Recovery: {len(candidates)} positions found on pool {self._pool_address}. "
                f"This controller manages exactly one; monitoring {adopted.address} and IGNORING "
                f"{self._extra_position_addresses}. No new position will be opened until only one "
                f"remains - close the extras manually with 'lp close-position'."
            )

    # =====================================================================
    # Position monitoring
    # =====================================================================

    async def _build_position_snapshot(self, now: float) -> Optional[PositionSnapshot]:
        executor = self._active_lp_executor()
        if executor is not None:
            return self._snapshot_from_executor(executor)
        if self._adopted_position_address:
            return await self._snapshot_from_chain(now)
        self._position_opened_at = None
        return None

    def _snapshot_from_executor(self, executor: ExecutorInfo) -> Optional[PositionSnapshot]:
        custom = executor.custom_info or {}
        position_address = custom.get("position_address")
        if not position_address:
            # Still opening - no bounds to monitor yet.
            return None

        price = self._to_decimal(custom.get("current_price")) or self._pool_price
        lower = self._to_decimal(custom.get("lower_price"))
        upper = self._to_decimal(custom.get("upper_price"))
        if price is None or lower is None or upper is None:
            self.logger().warning(f"Executor {executor.id} reported incomplete position data: {custom}")
            return None

        snapshot = PositionSnapshot(
            position_address=position_address,
            current_price=price,
            lower_price=lower,
            upper_price=upper,
            base_amount=self._to_decimal(custom.get("base_amount")) or Decimal("0"),
            quote_amount=self._to_decimal(custom.get("quote_amount")) or Decimal("0"),
            base_fee=self._to_decimal(custom.get("base_fee")) or Decimal("0"),
            quote_fee=self._to_decimal(custom.get("quote_fee")) or Decimal("0"),
            created_at=executor.timestamp,
            adopted=False,
        )
        if not snapshot.is_valid:
            self.logger().warning(
                f"Executor {executor.id} reported an invalid range "
                f"[{snapshot.lower_price}, {snapshot.upper_price}] at price {snapshot.current_price}"
            )
            return None
        self._position_opened_at = executor.timestamp
        return snapshot

    async def _snapshot_from_chain(self, now: float) -> Optional[PositionSnapshot]:
        """Read an adopted position straight from Gateway - no executor owns it."""
        connector = self._connector()
        if connector is None:
            return None
        try:
            info = await connector.get_position_info(
                trading_pair=self.config.trading_pair,
                dex_name=self.lp_dex_name,
                trading_type=self.lp_trading_type,
                position_address=self._adopted_position_address,
            )
        except Exception as e:                                       # pragma: no cover - network path
            self.logger().warning(f"Could not read adopted position {self._adopted_position_address}: {e}")
            return None

        if info is None:
            self.logger().info(
                f"Adopted position {self._adopted_position_address} is gone on-chain - releasing it"
            )
            self._adopted_position_address = None
            self._position_opened_at = None
            self._edge_counter = 0
            return None

        try:
            snapshot = PositionSnapshot(
                position_address=info.address,
                current_price=Decimal(str(info.price)),
                lower_price=Decimal(str(info.lower_price)),
                upper_price=Decimal(str(info.upper_price)),
                base_amount=Decimal(str(info.base_token_amount)),
                quote_amount=Decimal(str(info.quote_token_amount)),
                base_fee=Decimal(str(info.base_fee_amount)),
                quote_fee=Decimal(str(info.quote_fee_amount)),
                lower_tick=getattr(info, "lower_bin_id", None),
                upper_tick=getattr(info, "upper_bin_id", None),
                created_at=self._position_opened_at,
                adopted=True,
            )
        except Exception as e:
            self.logger().warning(f"Unexpected adopted position payload for {self._adopted_position_address}: {e}")
            return None

        if not snapshot.is_valid:
            self.logger().warning(
                f"Adopted position {snapshot.position_address} has an invalid range "
                f"[{snapshot.lower_price}, {snapshot.upper_price}] at price {snapshot.current_price}"
            )
            return None
        if self._position_opened_at is None:
            self._position_opened_at = now
        return snapshot

    def _update_edge_tracking(self, position: Optional[PositionSnapshot]) -> None:
        """Hysteresis: the edge condition must persist before it can trigger anything."""
        if position is None or not position.is_valid:
            self._edge_counter = 0
            return

        threshold = self.config.edge_threshold
        near_lower = position.lower_ratio < threshold
        near_upper = position.upper_ratio < threshold

        if near_lower or near_upper:
            self._edge_counter += 1
            edge = "LOWER" if near_lower else "UPPER"
            ratio = position.lower_ratio if near_lower else position.upper_ratio
            self.logger().info(
                f"PRICE APPROACHING {edge} EDGE at {self._fmt(position.current_price)} "
                f"(distance ratio {self._fmt(ratio, 4)} < {threshold}) - "
                f"confirmation {self._edge_counter}/{self.config.edge_confirmation_ticks}"
            )
        elif self._edge_counter:
            self.logger().info(
                f"Price moved back inside the safe zone (lower={self._fmt(position.lower_ratio, 4)}, "
                f"upper={self._fmt(position.upper_ratio, 4)}) - edge confirmation reset "
                f"from {self._edge_counter}"
            )
            self._edge_counter = 0

    def _evaluate_rebalance(self, candidate_range: Optional[PriceRange]) -> Optional[RebalanceEvaluation]:
        if self._position is None or candidate_range is None:
            return None
        return self.rebalance_filter.evaluate(self._position, candidate_range)

    def _evaluate_fee_collection(self) -> Optional[FeeCollectionEvaluation]:
        if self._position is None:
            return None
        return self.fee_collection_policy.evaluate(self._position.uncollected_fees_quote)

    # =====================================================================
    # Position sizing
    # =====================================================================

    async def _maintain_position_plan(self, candidate_range: Optional[PriceRange], now: float) -> None:
        """
        Keep a costed open-intent ready for the synchronous decision pass.

        Sizing needs a Gateway round trip, which cannot happen inside
        ``determine_executor_actions``; the plan is built here and invalidated as soon
        as the price drifts away from the one it was quoted at.
        """
        if not self._can_prepare_plan():
            self._position_plan = None
            return
        if self._position_plan is not None and self._plan_is_fresh(now):
            return
        if candidate_range is None or self._pool_price is None:
            self._position_plan = None
            return

        amounts = await self._quote_position_amounts(candidate_range)
        if amounts is None:
            self._position_plan = None
            return

        base_amount, quote_amount = amounts
        self._position_plan = PositionPlan(
            price_range=candidate_range,
            base_amount=base_amount,
            quote_amount=quote_amount,
            reference_price=self._pool_price,
            regime=self._regime,
            timestamp=now,
        )

    def _can_prepare_plan(self) -> bool:
        return (
            not self._halted
            and self._recovery_done
            and not self._extra_position_addresses
            and self._adopted_position_address is None
            and self._position is None
            and self._active_lp_executor() is None
            and self._regime is not MarketRegime.EXTREME
            and self.volatility_estimator.is_ready
        )

    def _plan_is_fresh(self, now: float) -> bool:
        plan = self._position_plan
        if plan is None or self._pool_price is None:
            return False
        if now - plan.timestamp > float(self.config.volatility_sample_interval) * 6:
            return False
        if plan.regime is not self._regime:
            return False
        drift = abs(self._pool_price - plan.reference_price) / plan.reference_price
        return drift <= Decimal("0.002")

    async def _quote_position_amounts(self, price_range: PriceRange) -> Optional[tuple]:
        """
        Ask Gateway for the token amounts a position over ``price_range`` requires.

        Falls back to the geometric split used elsewhere in the V2 LP stack when the
        quote endpoint is unavailable, so a transient Gateway hiccup degrades sizing
        accuracy instead of stalling the strategy.
        """
        capital = self.config.total_amount_quote
        price = self._pool_price
        if price is None or capital is None or capital < self.config.min_position_notional_quote:
            return None

        connector = self._connector()
        base_amount: Optional[Decimal] = None
        quote_amount: Optional[Decimal] = None

        if connector is not None:
            try:
                response = await GatewayHttpClient.get_instance().clmm_quote_position(
                    network=connector.network,
                    pool_address=self._pool_address,
                    lower_price=float(price_range.lower_price),
                    upper_price=float(price_range.upper_price),
                    dex=self.lp_dex_name,
                    trading_type=self.lp_trading_type,
                    quote_token_amount=float(capital / Decimal("2")),
                    slippage_pct=float(self.config.slippage_pct),
                )
                base_amount = self._to_decimal((response or {}).get("baseTokenAmount"))
                quote_amount = self._to_decimal((response or {}).get("quoteTokenAmount"))
            except Exception as e:                                   # pragma: no cover - network path
                self.logger().warning(f"quote-position failed, falling back to geometric split: {e}")

        if base_amount is None or quote_amount is None:
            base_amount, quote_amount = self._fallback_amounts(price_range, capital, price)

        if base_amount < 0 or quote_amount < 0:
            self.logger().warning(f"Rejecting negative position amounts base={base_amount} quote={quote_amount}")
            return None

        # Never deploy more than the configured capital, whatever the quote returns.
        notional = base_amount * price + quote_amount
        if notional <= 0:
            self.logger().warning("Rejecting zero-notional position plan")
            return None
        if notional > capital:
            scale = capital / notional
            base_amount *= scale
            quote_amount *= scale
            notional = capital

        base_amount = self._quantize(base_amount)
        quote_amount = self._quantize(quote_amount)
        if base_amount * price + quote_amount < self.config.min_position_notional_quote:
            self.logger().warning(
                f"Position notional {self._fmt(notional)} below minimum "
                f"{self.config.min_position_notional_quote} - not opening"
            )
            return None
        return base_amount, quote_amount

    def _fallback_amounts(self, price_range: PriceRange, capital: Decimal, price: Decimal) -> tuple:
        """Split capital by where the price sits inside the range (same shape as lp_rebalancer)."""
        width = price_range.upper_price - price_range.lower_price
        if width <= 0 or price <= price_range.lower_price:
            return Decimal("0"), capital
        if price >= price_range.upper_price:
            return capital / price, Decimal("0")
        quote_share = (price - price_range.lower_price) / width
        quote_amount = capital * quote_share
        base_amount = (capital - quote_amount) / price
        return base_amount, quote_amount

    # =====================================================================
    # Synchronous decision pass
    # =====================================================================

    def determine_executor_actions(self) -> List[ExecutorAction]:
        now = self.market_data_provider.time()
        actions = self._decide(now)
        # Logged here rather than in the data pass so the line reports the decision
        # this cycle actually reached.
        self._maybe_log_status(now)
        return actions

    def _decide(self, now: float) -> List[ExecutorAction]:
        self._reconcile_executors(now)

        if self._halted:
            return self._hold(StrategyState.ERROR, "halted after repeated executor failures - restart required")
        if self._pool_price is None:
            return self._hold(self._state, "waiting for a valid pool price")
        if now < self._error_until:
            return self._hold(
                StrategyState.ERROR, f"backing off after failure for another {self._error_until - now:.0f}s"
            )
        if not self._recovery_done:
            return self._hold(self._state, "waiting for on-chain position recovery")

        # Extreme volatility overrides every other consideration.
        if self._regime is MarketRegime.EXTREME:
            return self._handle_extreme_volatility(now)

        if self._reconcile_swap(now):
            return self._hold(StrategyState.NO_POSITION, "autoswap in flight - waiting for it to settle")

        executor = self._active_lp_executor()
        if executor is not None:
            return self._manage_active_position(executor, now)
        if self._adopted_position_address is not None:
            return self._manage_adopted_position(now)
        return self._consider_open(now)

    def _reconcile_executors(self, now: float) -> None:
        """Track the live executor and settle the books for one that has terminated."""
        # Settle the tracked executor before adopting a newly active one. The reverse
        # order would silently retarget the tracking id and drop the old executor's
        # fee accounting, which matters the moment the open guard is ever relaxed.
        if self._current_executor_id:
            info = self._find_executor(self._current_executor_id)
            if info is None:
                self._current_executor_id = None
                self._rebalance_requested_for = None
            elif info.status is RunnableStatus.TERMINATED:
                self._settle_terminated_executor(info, now)

        active = self._active_lp_executor()
        if active is not None:
            if self._current_executor_id != active.id:
                self._current_executor_id = active.id
                self.logger().info(f"Tracking LP executor {active.id}")
            self._open_requested_at = None

    def _settle_terminated_executor(self, info: ExecutorInfo, now: float) -> None:
        """Book a terminated executor's outcome: fees harvested, or a failure to back off from."""
        custom = info.custom_info or {}
        if info.close_type is CloseType.FAILED:
            self._consecutive_failures += 1
            self._error_until = now + float(self.config.error_backoff_seconds) * self._consecutive_failures
            self.logger().error(
                f"LP executor {info.id} FAILED (attempt {self._consecutive_failures}/"
                f"{self.config.max_consecutive_failures}). Position state is NOT assumed to be active; "
                f"backing off until {self._error_until:.0f}. Executor report: {custom}"
            )
            if self._consecutive_failures >= self.config.max_consecutive_failures:
                self._halted = True
                self.logger().error(
                    f"Reached {self._consecutive_failures} consecutive executor failures on "
                    f"{self.config.trading_pair}. Halting - inspect the wallet and pool, then restart the controller."
                )
        else:
            if self._consecutive_failures:
                self.logger().info(
                    f"Recovered after {self._consecutive_failures} failure(s); executor {info.id} "
                    f"closed with {info.close_type.name if info.close_type else 'UNKNOWN'}"
                )
            self._consecutive_failures = 0
            self._error_until = 0.0
            price = self._to_decimal(custom.get("current_price")) or self._pool_price or Decimal("0")
            harvested = (
                (self._to_decimal(custom.get("base_fee")) or Decimal("0")) * price
                + (self._to_decimal(custom.get("quote_fee")) or Decimal("0"))
            )
            if harvested > 0:
                self._collected_fees_quote += harvested
                self.logger().info(
                    f"Position {custom.get('position_address')} closed; fees harvested on close: "
                    f"{self._fmt(harvested)} {self._quote_token} "
                    f"(cumulative {self._fmt(self._collected_fees_quote)})"
                )

        self.logger().info(
            f"LP executor {info.id} terminated "
            f"(close_type={info.close_type.name if info.close_type else 'UNKNOWN'}) - controller state is "
            f"{'ERROR' if info.close_type is CloseType.FAILED else 'NO_POSITION'} until a new position opens"
        )
        self._current_executor_id = None
        self._rebalance_requested_for = None
        self._position_opened_at = None
        self._edge_counter = 0
        if self._active_lp_executor() is None:
            self._position = None

    def _handle_extreme_volatility(self, now: float) -> List[ExecutorAction]:
        self._position_plan = None
        # Stamped on every extreme cycle, not just the one that closes a position:
        # the reopen cooldown has to run from the last time the market looked like
        # this, otherwise the strategy would redeploy the instant the ratio dips
        # back under the threshold - in the middle of the same turbulence.
        self._extreme_exit_timestamp = now
        executor = self._active_lp_executor()

        if executor is not None:
            if self._rebalance_requested_for == executor.id:
                return self._hold(StrategyState.REBALANCING, "extreme volatility - waiting for the position to close")
            executor_state = (executor.custom_info or {}).get("state")
            if executor_state in TRANSIENT_EXECUTOR_STATES:
                return self._hold(
                    StrategyState.REBALANCING, f"extreme volatility - executor busy ({executor_state})"
                )
            self.logger().warning(
                f"EXTREME VOLATILITY DETECTED (ratio={self._fmt(self.volatility_estimator.volatility_ratio, 2)} "
                f">= {self.config.extreme_vol_threshold}) - withdrawing liquidity from "
                f"{(executor.custom_info or {}).get('position_address')}. Fees are harvested by the close itself."
            )
            self._rebalance_requested_for = executor.id
            self._state = StrategyState.REBALANCING
            self._last_action = StrategyAction.CLOSE
            self._last_reason = "extreme volatility exit"
            return [StopExecutorAction(
                controller_id=self.config.id, executor_id=executor.id, keep_position=True
            )]

        if self._adopted_position_address is not None:
            if self._adopted_close_in_flight:
                return self._hold(StrategyState.REBALANCING, "extreme volatility - adopted position close in flight")
            self.logger().warning(
                f"EXTREME VOLATILITY DETECTED - closing adopted position {self._adopted_position_address}"
            )
            self._start_adopted_close("extreme volatility")
            return self._hold(
                StrategyState.REBALANCING, "extreme volatility exit", action=StrategyAction.CLOSE
            )

        return self._hold(
            StrategyState.PAUSED,
            f"extreme volatility (ratio={self._fmt(self.volatility_estimator.volatility_ratio, 2)}) - staying flat",
        )

    def _manage_active_position(self, executor: ExecutorInfo, now: float) -> List[ExecutorAction]:
        if self._rebalance_requested_for == executor.id:
            return self._hold(StrategyState.REBALANCING, f"waiting for executor {executor.id} to close")

        executor_state = (executor.custom_info or {}).get("state")
        if executor_state in TRANSIENT_EXECUTOR_STATES:
            state = StrategyState.NO_POSITION if executor_state == "OPENING" else StrategyState.REBALANCING
            return self._hold(state, f"executor transaction in flight ({executor_state})")
        if executor_state not in DEPLOYED_EXECUTOR_STATES or self._position is None:
            return self._hold(StrategyState.NO_POSITION, f"position not deployed yet (executor state {executor_state})")

        return self._monitor_deployed_position(now, executor=executor)

    def _manage_adopted_position(self, now: float) -> List[ExecutorAction]:
        if self._adopted_close_in_flight:
            return self._hold(StrategyState.REBALANCING, "adopted position close in flight")
        if self._position is None:
            return self._hold(StrategyState.NO_POSITION, "waiting for adopted position data")
        return self._monitor_deployed_position(now, executor=None)

    def _monitor_deployed_position(self, now: float, executor: Optional[ExecutorInfo]) -> List[ExecutorAction]:
        """
        Shared monitoring path for both executor-owned and adopted positions.

        Only the *execution* of a close differs between them; the boundary,
        hysteresis, cooldown and economic decisions are identical.

        The rebalance decision is taken before fee collection is even considered.
        Closing a position harvests its fees anyway, so collecting immediately before
        a rebalance would pay for a transaction that the close makes redundant.
        """
        if self._fee_collection_in_flight:
            return self._hold(StrategyState.COLLECTING, "collect-fees transaction in flight",
                              action=StrategyAction.COLLECT_FEES)

        rebalance_actions = self._consider_rebalance(now, executor)
        if rebalance_actions is not None:
            return rebalance_actions

        self._maybe_collect_fees(now)
        if self._fee_collection_in_flight:
            return self._hold(StrategyState.COLLECTING, "collect-fees transaction submitted",
                              action=StrategyAction.COLLECT_FEES)
        return self._hold(StrategyState.ACTIVE, self._last_reason)

    def _consider_rebalance(self, now: float,
                            executor: Optional[ExecutorInfo]) -> Optional[List[ExecutorAction]]:
        """
        Run the edge/cooldown/economic gates.

        Returns the actions to take, or None when no rebalance is warranted - which
        leaves the caller free to consider fee collection instead. A ``_hold`` result
        here means "rebalancing is settled, and the answer was no".
        """
        if self._edge_counter < self.config.edge_confirmation_ticks:
            if self._edge_counter:
                reason = f"edge confirmation {self._edge_counter}/{self.config.edge_confirmation_ticks}"
            else:
                reason = "position comfortably in range"
            self._last_reason = reason
            return None

        if self._last_rebalance_timestamp:
            elapsed = now - self._last_rebalance_timestamp
            if elapsed < float(self.config.rebalance_cooldown):
                self._last_reason = (
                    f"edge confirmed but rebalance cooldown has "
                    f"{float(self.config.rebalance_cooldown) - elapsed:.0f}s left"
                )
                return None

        evaluation: Optional[RebalanceEvaluation] = self.processed_data.get("rebalance_evaluation")
        if evaluation is None:
            self._last_reason = "edge confirmed but no range candidate available"
            return None

        approved = evaluation.should_rebalance
        if approved or (now - self._last_evaluation_log) >= EVALUATION_LOG_INTERVAL:
            self._last_evaluation_log = now
            self._log_rebalance_evaluation(evaluation)

        if not approved:
            self._last_reason = f"rebalance rejected - {evaluation.reason}"
            return None

        # Approved. The cooldown clock starts now so a failing rebalance cannot spin.
        self._last_rebalance_timestamp = now
        self._rebalance_count += 1
        self._edge_counter = 0
        self._position_plan = None

        if executor is not None:
            self.logger().info(
                f"REBALANCE APPROVED (#{self._rebalance_count}) - stopping executor {executor.id} "
                f"to re-centre around {self._fmt(self._pool_price)}"
            )
            self._rebalance_requested_for = executor.id
            self._state = StrategyState.REBALANCING
            self._last_action = StrategyAction.REBALANCE
            self._last_reason = evaluation.reason
            return [StopExecutorAction(controller_id=self.config.id, executor_id=executor.id, keep_position=True)]

        self.logger().info(
            f"REBALANCE APPROVED (#{self._rebalance_count}) - closing adopted position "
            f"{self._adopted_position_address} to re-centre around {self._fmt(self._pool_price)}"
        )
        self._start_adopted_close("adaptive rebalance")
        return self._hold(StrategyState.REBALANCING, evaluation.reason, action=StrategyAction.REBALANCE)

    def _log_rebalance_evaluation(self, evaluation: RebalanceEvaluation) -> None:
        self.logger().info(
            "REBALANCE EVALUATION\n"
            f"  current range fees : {self._fmt(evaluation.current_range_fees)} {self._quote_token}\n"
            f"  new range fees     : {self._fmt(evaluation.new_range_fees)} {self._quote_token}\n"
            f"  expected benefit   : {self._fmt(evaluation.expected_benefit)} {self._quote_token}\n"
            f"  estimated cost     : {self._fmt(evaluation.estimated_cost)} {self._quote_token}\n"
            f"  required benefit   : {self._fmt(evaluation.required_benefit)} {self._quote_token}\n"
            f"  decision           : {'REBALANCE' if evaluation.should_rebalance else 'HOLD'}"
        )

    def _consider_open(self, now: float) -> List[ExecutorAction]:
        if self._extra_position_addresses:
            return self._hold(
                StrategyState.PAUSED,
                f"{len(self._extra_position_addresses) + 1} positions exist on this pool - "
                "refusing to open another until the extras are closed manually",
            )
        if self._current_executor_id:
            return self._hold(StrategyState.REBALANCING, "waiting for the previous executor to terminate")
        if self._open_requested_at is not None:
            if now - self._open_requested_at < OPEN_REQUEST_TIMEOUT:
                return self._hold(StrategyState.NO_POSITION, "open request submitted, waiting for the executor")
            self.logger().warning(
                f"No LP executor appeared {OPEN_REQUEST_TIMEOUT:.0f}s after the open request - retrying"
            )
            self._open_requested_at = None

        if self._extreme_exit_timestamp is not None:
            elapsed = now - self._extreme_exit_timestamp
            cooldown = float(self.config.extreme_vol_reopen_cooldown)
            if elapsed < cooldown:
                return self._hold(
                    StrategyState.PAUSED,
                    f"extreme-volatility reopen cooldown: {cooldown - elapsed:.0f}s remaining",
                )
            self.logger().info("Extreme-volatility reopen cooldown elapsed - liquidity may be deployed again")
            self._extreme_exit_timestamp = None

        if not self.volatility_estimator.is_ready:
            return self._hold(
                StrategyState.NO_POSITION,
                f"insufficient volatility history "
                f"({self.volatility_estimator.sample_count}/{self.config.volatility_min_samples} samples)",
            )

        plan = self._position_plan
        if plan is None:
            return self._hold(StrategyState.NO_POSITION, "waiting for a sized position plan")
        if not self._plan_is_fresh(now):
            self._position_plan = None
            return self._hold(StrategyState.NO_POSITION, "position plan went stale - re-quoting")
        shortfall = self._balance_shortfall(plan)
        if shortfall is not None:
            return self._handle_shortfall(plan, shortfall, now)

        executor_config = self._build_lp_executor_config(plan, now)
        if executor_config is None:
            self._position_plan = None
            return self._hold(StrategyState.NO_POSITION, "planned position failed validation")

        self.logger().info(
            f"OPENING LP POSITION\n"
            f"  regime      : {plan.regime.value}\n"
            f"  volatility  : {self._fmt(self.volatility_estimator.volatility, 6)} "
            f"(baseline {self._fmt(self.volatility_estimator.baseline_volatility, 6)}, "
            f"ratio {self._fmt(self.volatility_estimator.volatility_ratio, 2)})\n"
            f"  price       : {self._fmt(plan.reference_price)}\n"
            f"  half-width  : {self._fmt(plan.price_range.width_pct * 100, 4)}%\n"
            f"  range       : {self._fmt(plan.price_range.lower_price)} - "
            f"{self._fmt(plan.price_range.upper_price)}\n"
            f"  amounts     : {self._fmt(plan.base_amount, 6)} {self._base_token} + "
            f"{self._fmt(plan.quote_amount, 6)} {self._quote_token} "
            f"(~{self._fmt(plan.notional_quote)} {self._quote_token})\n"
            f"  ticks       : aligned by Gateway/Orca on submit; realized bounds are read back from the position"
        )

        self._position_plan = None
        self._open_requested_at = now
        self._edge_counter = 0
        self._state = StrategyState.NO_POSITION
        self._last_action = StrategyAction.OPEN
        self._last_reason = f"opening {plan.regime.value} range"
        return [CreateExecutorAction(controller_id=self.config.id, executor_config=executor_config)]

    def _build_lp_executor_config(self, plan: PositionPlan, now: float) -> Optional[LPExecutorConfig]:
        price_range = plan.price_range
        if not self._pool_address:
            self.logger().warning("No pool address available - cannot open a position")
            return None
        if price_range.lower_price <= 0 or price_range.lower_price >= price_range.upper_price:
            self.logger().error(
                f"Refusing to submit an invalid range [{price_range.lower_price}, {price_range.upper_price}]"
            )
            return None
        if plan.base_amount < 0 or plan.quote_amount < 0 or plan.notional_quote <= 0:
            self.logger().error(
                f"Refusing to submit invalid amounts base={plan.base_amount} quote={plan.quote_amount}"
            )
            return None

        safety = self.config.safety_exit_pct
        upper_limit = price_range.upper_price * (Decimal("1") + safety) if safety > 0 else None
        lower_limit = price_range.lower_price * (Decimal("1") - safety) if safety > 0 else None

        return LPExecutorConfig(
            timestamp=now,
            controller_id=self.config.id,
            connector_name=self.config.connector_name,
            lp_provider=self.config.lp_provider,
            trading_pair=self.config.trading_pair,
            pool_address=self._pool_address,
            lower_price=price_range.lower_price,
            upper_price=price_range.upper_price,
            base_amount=plan.base_amount,
            quote_amount=plan.quote_amount,
            side=TradeType.RANGE,
            upper_limit_price=upper_limit,
            lower_limit_price=lower_limit,
            keep_position=True,
        )

    def _balance_shortfall(self, plan: PositionPlan) -> Optional[Dict[str, Decimal]]:
        """
        How much of each token the planned position is short.

        Returns None when the wallet can fund it, otherwise a dict with the deficit in
        each token (zero or negative where there is no deficit). Reporting the deficit
        rather than a bare bool is what lets autoswap size the trade.
        """
        connector = self._connector()
        if connector is None:
            return None
        try:
            base_balance = Decimal(str(self.market_data_provider.get_balance(
                self.config.connector_name, self._base_token)))
            quote_balance = Decimal(str(self.market_data_provider.get_balance(
                self.config.connector_name, self._quote_token)))
        except Exception as e:
            self.logger().debug(f"Balance check unavailable ({e}); deferring validation to Gateway")
            return None

        native = (getattr(connector, "native_currency", None) or "").upper()
        buffer_fn = getattr(connector, "get_native_currency_buffer", None)
        buffer = Decimal(str(buffer_fn())) if callable(buffer_fn) else Decimal("0.005")

        required_base = plan.base_amount + (buffer if native and self._base_token.upper() == native else Decimal("0"))
        required_quote = plan.quote_amount + (buffer if native and self._quote_token.upper() == native else Decimal("0"))

        base_deficit = required_base - base_balance
        quote_deficit = required_quote - quote_balance
        if base_deficit <= 0 and quote_deficit <= 0:
            return None

        # Name only the leg that is actually short: printing both reads as if the
        # funded side had failed too.
        short_legs = []
        if base_deficit > 0:
            short_legs.append(
                f"{self._fmt(base_deficit, 6)} {self._base_token} "
                f"(need {self._fmt(required_base, 6)}, have {self._fmt(base_balance, 6)})"
            )
        if quote_deficit > 0:
            short_legs.append(
                f"{self._fmt(quote_deficit, 6)} {self._quote_token} "
                f"(need {self._fmt(required_quote, 6)}, have {self._fmt(quote_balance, 6)})"
            )
        self.logger().info(f"Position underfunded - short {' and '.join(short_legs)}")

        return {
            "base_deficit": base_deficit,
            "quote_deficit": quote_deficit,
            "base_balance": base_balance,
            "quote_balance": quote_balance,
            "required_base": required_base,
            "required_quote": required_quote,
        }

    def _handle_shortfall(self, plan: PositionPlan, shortfall: Dict[str, Decimal],
                          now: float) -> List[ExecutorAction]:
        """Swap into the missing token when autoswap is on, otherwise hold."""
        if not self.config.autoswap:
            return self._hold(
                StrategyState.NO_POSITION,
                "insufficient token balances for the planned position (autoswap disabled)",
            )
        if self._consecutive_swap_failures >= self.config.max_consecutive_swap_failures:
            return self._hold(
                StrategyState.NO_POSITION,
                f"autoswap gave up after {self._consecutive_swap_failures} failures - "
                f"rebalance the wallet manually",
            )

        swap_config = self._build_swap_config(shortfall, now)
        if swap_config is None:
            return self._hold(StrategyState.NO_POSITION, self._last_reason)

        self._swap_requested_at = now
        self._position_plan = None      # re-quote once the wallet has changed
        self._last_action = StrategyAction.HOLD
        self._state = StrategyState.NO_POSITION
        self._last_reason = (
            f"swapping to fund the position: {swap_config.side.name} "
            f"{self._fmt(swap_config.amount, 6)} {self._base_token}"
        )
        self.logger().info(f"AUTOSWAP: {self._last_reason}")
        return [CreateExecutorAction(controller_id=self.config.id, executor_config=swap_config)]

    def _build_swap_config(self, shortfall: Dict[str, Decimal], now: float) -> Optional[OrderExecutorConfig]:
        """
        Size a market swap that covers the deficit out of the surplus token.

        Amount is always expressed in base units, matching OrderExecutor's contract.
        """
        price = self._pool_price
        if price is None or price <= 0:
            self._last_reason = "waiting for a pool price before swapping"
            return None

        base_deficit = shortfall["base_deficit"]
        quote_deficit = shortfall["quote_deficit"]
        base_balance = shortfall["base_balance"]
        quote_balance = shortfall["quote_balance"]
        buffer_multiplier = Decimal("1") + (self.config.swap_buffer_pct / Decimal("100"))

        if base_deficit > 0 and quote_deficit > 0:
            total = base_deficit * price + quote_deficit
            self._last_reason = (
                f"both tokens short by ~{self._fmt(total)} {self._quote_token} in total - "
                f"the wallet needs more capital, not a swap"
            )
            self.logger().warning(f"AUTOSWAP: {self._last_reason}")
            return None

        if quote_deficit > 0:
            # Short quote, so sell base for it. Only the balance in excess of what the
            # position itself needs is sellable - the rest is the deposit plus the
            # chain's native-currency reserve for rent and fees (0.1 SOL on Solana),
            # and selling into it would just recreate the shortfall on the other leg.
            surplus_base = base_balance - shortfall["required_base"]
            swap_amount = self._quantize((quote_deficit / price) * buffer_multiplier)
            if swap_amount <= 0:
                self._last_reason = "quote deficit too small to swap"
                return None
            if surplus_base < swap_amount:
                self._last_reason = (
                    f"cannot cover the {self._fmt(quote_deficit)} {self._quote_token} deficit - "
                    f"selling it needs {self._fmt(swap_amount, 6)} {self._base_token} but only "
                    f"{self._fmt(surplus_base, 6)} is spare once the position and its native "
                    f"reserve are set aside"
                )
                self.logger().warning(f"AUTOSWAP: {self._last_reason}")
                return None
            side = TradeType.SELL
        else:
            # Short base, so buy it with quote. Symmetrically, only quote beyond what the
            # position needs may be spent.
            surplus_quote = quote_balance - shortfall["required_quote"]
            swap_amount = self._quantize(base_deficit * buffer_multiplier)
            if swap_amount <= 0:
                self._last_reason = "base deficit too small to swap"
                return None
            cost_quote = swap_amount * price * buffer_multiplier
            if surplus_quote < cost_quote:
                self._last_reason = (
                    f"cannot cover the {self._fmt(base_deficit, 6)} {self._base_token} deficit - "
                    f"buying it costs ~{self._fmt(cost_quote)} {self._quote_token} but only "
                    f"{self._fmt(surplus_quote)} is spare once the position is set aside"
                )
                self.logger().warning(f"AUTOSWAP: {self._last_reason}")
                return None
            side = TradeType.BUY

        return OrderExecutorConfig(
            timestamp=now,
            controller_id=self.config.id,
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            side=side,
            amount=swap_amount,
            execution_strategy=ExecutionStrategy.MARKET,
        )

    def _active_swap_executor(self) -> Optional[ExecutorInfo]:
        for info in self.executors_info:
            if info.is_active and getattr(info.config, "type", None) == "order_executor":
                return info
        return None

    def _reconcile_swap(self, now: float) -> bool:
        """
        Track the autoswap order executor. Returns True while one is still in flight.

        The swap has to settle before the position is sized again: quoting against a
        wallet that is mid-swap would just reproduce the same shortfall.
        """
        active = self._active_swap_executor()
        if active is not None:
            self._swap_executor_id = active.id
            return True

        if self._swap_executor_id:
            info = self._find_executor(self._swap_executor_id)
            self._swap_executor_id = None
            self._swap_requested_at = None
            if info is not None and info.close_type is CloseType.FAILED:
                self._consecutive_swap_failures += 1
                self.logger().error(
                    f"AUTOSWAP failed (attempt {self._consecutive_swap_failures}/"
                    f"{self.config.max_consecutive_swap_failures})"
                )
            else:
                self._consecutive_swap_failures = 0
                self.logger().info("AUTOSWAP completed; re-sizing the position against the new balances")
                self._trigger_balance_update()
            self._position_plan = None
            return False

        # A create that never surfaced as an executor - allow a retry rather than hang.
        if self._swap_requested_at is not None:
            if now - self._swap_requested_at < OPEN_REQUEST_TIMEOUT:
                return True
            self.logger().warning("No swap executor appeared after the request - retrying")
            self._swap_requested_at = None
        return False

    def _trigger_balance_update(self) -> None:
        """Ask the connector to refresh balances so the next cycle sizes against reality."""
        connector = self._connector()
        update = getattr(connector, "update_balances", None) if connector else None
        if callable(update):
            try:
                safe_ensure_future(update())
            except Exception as e:
                self.logger().debug(f"Could not trigger a balance update: {e}")

    # =====================================================================
    # Gateway side-effects that have no executor path
    # =====================================================================

    def _maybe_collect_fees(self, now: float) -> None:
        """
        Harvest fees mid-position when they are worth more than the transaction.

        The LP executor already collects fees whenever it closes a position, so this
        only covers the in-between case. It touches fees only - never liquidity - so
        it cannot conflict with the executor's own lifecycle. One consequence worth
        knowing: the executor derives unrealized P&L from the position's *uncollected*
        fees, so its reported P&L drops by whatever is harvested here. The controller
        tracks the harvested total itself and reports it in the status panel.
        """
        if not self.config.enable_fee_collection or self._fee_collection_in_flight:
            return
        if self._position is None or not self._position.position_address:
            return
        if self._last_fee_collection_timestamp:
            elapsed = now - self._last_fee_collection_timestamp
            if elapsed < float(self.config.fee_collection_cooldown):
                return

        evaluation: Optional[FeeCollectionEvaluation] = self.processed_data.get("fee_evaluation")
        if evaluation is None or not evaluation.should_collect:
            return

        self.logger().info(
            "FEE COLLECTION\n"
            f"  fees available : {self._fmt(evaluation.fees_available, 6)} {self._quote_token}\n"
            f"  estimated cost : {self._fmt(evaluation.estimated_cost, 6)} {self._quote_token}\n"
            f"  required       : {self._fmt(evaluation.required_fees, 6)} {self._quote_token}\n"
            f"  decision       : COLLECT"
        )
        self._fee_collection_in_flight = True
        self._last_fee_collection_timestamp = now
        safe_ensure_future(
            self._collect_fees_task(self._position.position_address, evaluation.fees_available)
        )

    async def _collect_fees_task(self, position_address: str, fees_expected: Decimal) -> None:
        try:
            connector = self._connector()
            if connector is None:
                return
            result = await GatewayHttpClient.get_instance().clmm_collect_fees(
                network=connector.network,
                wallet_address=connector.address,
                position_address=position_address,
                dex=self.lp_dex_name,
                trading_type=self.lp_trading_type,
            )
            signature = (result or {}).get("signature")
            if signature:
                self._collected_fees_quote += fees_expected
                self.logger().info(
                    f"Collected ~{self._fmt(fees_expected, 6)} {self._quote_token} of fees from "
                    f"{position_address} (tx {signature}); cumulative "
                    f"{self._fmt(self._collected_fees_quote, 6)} {self._quote_token}"
                )
            else:
                self.logger().warning(f"Fee collection returned no signature for {position_address}: {result}")
        except Exception as e:
            self.logger().error(f"Fee collection failed for {position_address}: {e}")
        finally:
            self._fee_collection_in_flight = False

    def _start_adopted_close(self, reason: str) -> None:
        if self._adopted_close_in_flight or not self._adopted_position_address:
            return
        self._adopted_close_in_flight = True
        safe_ensure_future(self._close_adopted_position(self._adopted_position_address, reason))

    async def _close_adopted_position(self, position_address: str, reason: str) -> None:
        """
        Close a position adopted at startup.

        An adopted position predates this controller run, so there is no executor to
        stop; the close goes through the same connector call ``LPExecutor`` itself
        makes. Once it is gone the normal executor-managed lifecycle takes over, which
        is what makes restart recovery complete rather than read-only.
        """
        connector = self._connector()
        try:
            if connector is None:
                self.logger().error("No connector available to close the adopted position")
                return
            self.logger().info(f"Closing adopted position {position_address} ({reason})")
            order_id = connector.create_market_order_id(TradeType.RANGE, self.config.trading_pair)
            signature = await connector._clmm_close_position(
                trade_type=TradeType.RANGE,
                order_id=order_id,
                trading_pair=self.config.trading_pair,
                position_address=position_address,
                dex_name=self.lp_dex_name,
                trading_type=self.lp_trading_type,
            )
            metadata = getattr(connector, "_lp_orders_metadata", {}).pop(order_id, {}) or {}
            price = self._pool_price or Decimal("0")
            harvested = (
                Decimal(str(metadata.get("base_fee", 0))) * price + Decimal(str(metadata.get("quote_fee", 0)))
            )
            self._collected_fees_quote += harvested
            self.logger().info(
                f"Adopted position {position_address} closed (tx {signature}); fees harvested "
                f"{self._fmt(harvested, 6)} {self._quote_token}"
            )
            self._adopted_position_address = None
            self._position = None
            self._position_opened_at = None
            self._edge_counter = 0
            self._consecutive_failures = 0
        except Exception as e:
            self._consecutive_failures += 1
            self._error_until = (
                self.market_data_provider.time()
                + float(self.config.error_backoff_seconds) * self._consecutive_failures
            )
            self.logger().error(
                f"Failed to close adopted position {position_address}: {e}. "
                f"The position is still open on-chain; retrying after backoff."
            )
        finally:
            self._adopted_close_in_flight = False

    # =====================================================================
    # Helpers
    # =====================================================================

    def _connector(self):
        try:
            return self.market_data_provider.get_connector(self.config.connector_name)
        except Exception as e:
            self.logger().debug(f"Connector {self.config.connector_name} not available: {e}")
            return None

    def _active_lp_executor(self) -> Optional[ExecutorInfo]:
        active = [
            info for info in self.executors_info
            if info.is_active and getattr(info.config, "type", None) == "lp_executor"
        ]
        if len(active) > 1:
            self.logger().warning(
                f"{len(active)} active LP executors found; this controller manages one. "
                f"Using {active[0].id} and leaving the rest untouched."
            )
        return active[0] if active else None

    def _find_executor(self, executor_id: str) -> Optional[ExecutorInfo]:
        for info in self.executors_info:
            if info.id == executor_id:
                return info
        return None

    def _hold(self, state: StrategyState, reason: str,
              action: StrategyAction = StrategyAction.HOLD) -> List[ExecutorAction]:
        self._state = state
        self._last_action = action
        self._last_reason = reason
        return []

    @staticmethod
    def _to_decimal(value: Any) -> Optional[Decimal]:
        """Convert Gateway/executor payload values to Decimal, rejecting junk."""
        if value is None:
            return None
        try:
            if isinstance(value, float) and not math.isfinite(value):
                return None
            result = Decimal(str(value))
        except Exception:
            return None
        return result if result.is_finite() else None

    @staticmethod
    def _quantize(amount: Decimal, places: str = "0.000000001") -> Decimal:
        """Round token amounts down so rounding can never overspend a balance."""
        return amount.quantize(Decimal(places), rounding=ROUND_DOWN)

    @staticmethod
    def _fmt(value: Optional[Decimal], places: int = 4) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.{places}f}"

    def _seconds_since(self, timestamp: Optional[float], now: float) -> Optional[float]:
        if not timestamp:
            return None
        return now - timestamp

    # =====================================================================
    # Metrics, status and logging
    # =====================================================================

    def _build_processed_data(self,
                              now: float,
                              candidate_range: Optional[PriceRange],
                              rebalance_eval: Optional[RebalanceEvaluation],
                              fee_eval: Optional[FeeCollectionEvaluation]) -> Dict[str, Any]:
        return {
            "timestamp": now,
            "pool_address": self._pool_address,
            "current_price": self._pool_price,
            "pool_fee_pct": self._pool_fee_pct,
            "volatility": self.volatility_estimator.volatility,
            "baseline_volatility": self.volatility_estimator.baseline_volatility,
            "volatility_ratio": self.volatility_estimator.volatility_ratio,
            "volatility_samples": self.volatility_estimator.sample_count,
            "volatility_ready": self.volatility_estimator.is_ready,
            "market_regime": self._regime,
            "regime_multiplier": self.range_calculator.regime_multiplier(self._regime),
            "candidate_range": candidate_range,
            "position": self._position,
            "rebalance_evaluation": rebalance_eval,
            "fee_evaluation": fee_eval,
            "edge_counter": self._edge_counter,
        }

    def _maybe_log_status(self, now: float) -> None:
        interval = float(self.config.status_log_interval)
        if interval <= 0 or (now - self._last_status_log) < interval:
            return
        self._last_status_log = now

        position = self._position
        parts = [
            f"[{self.config.trading_pair}] state={self._state.value}",
            f"price={self._fmt(self._pool_price)}",
            f"vol={self._fmt(self.volatility_estimator.volatility, 6)}",
            f"baseline={self._fmt(self.volatility_estimator.baseline_volatility, 6)}",
            f"ratio={self._fmt(self.volatility_estimator.volatility_ratio, 2)}",
            f"regime={self._regime.value}",
        ]
        if position is not None:
            parts.append(
                f"range=[{self._fmt(position.lower_price)}, {self._fmt(position.upper_price)}]"
            )
            parts.append(
                f"edges=(lower {self._fmt(position.lower_ratio, 3)}, upper {self._fmt(position.upper_ratio, 3)})"
            )
            parts.append(f"fees={self._fmt(position.uncollected_fees_quote, 6)}")
        parts.append(f"action={self._last_action.value}")
        parts.append(f"reason={self._last_reason}")
        self.logger().info(" | ".join(parts))

    def get_custom_info(self) -> dict:
        """Compact, JSON-friendly snapshot published alongside the performance report."""
        now = self.market_data_provider.time()
        position = self._position
        rebalance_eval: Optional[RebalanceEvaluation] = self.processed_data.get("rebalance_evaluation")
        fee_eval: Optional[FeeCollectionEvaluation] = self.processed_data.get("fee_evaluation")

        info: Dict[str, Any] = {
            "trading_pair": self.config.trading_pair,
            "pool_address": self._pool_address,
            "lp_provider": self.config.lp_provider,
            "current_price": self._as_float(self._pool_price),
            "current_volatility": self._as_float(self.volatility_estimator.volatility),
            "baseline_volatility": self._as_float(self.volatility_estimator.baseline_volatility),
            "volatility_ratio": self._as_float(self.volatility_estimator.volatility_ratio),
            "volatility_samples": self.volatility_estimator.sample_count,
            "market_regime": self._regime.value,
            "position_active": position is not None,
            "position_address": position.position_address if position else None,
            "position_adopted": position.adopted if position else False,
            "position_lower_price": self._as_float(position.lower_price) if position else None,
            "position_upper_price": self._as_float(position.upper_price) if position else None,
            "position_lower_tick": position.lower_tick if position else None,
            "position_upper_tick": position.upper_tick if position else None,
            "distance_to_lower": self._as_float(position.distance_from_lower) if position else None,
            "distance_to_upper": self._as_float(position.distance_from_upper) if position else None,
            "lower_ratio": self._as_float(position.lower_ratio) if position else None,
            "upper_ratio": self._as_float(position.upper_ratio) if position else None,
            "liquidity": self._as_float(position.liquidity_quote) if position else None,
            "uncollected_fees": self._as_float(position.uncollected_fees_quote) if position else None,
            "collected_fees": self._as_float(self._collected_fees_quote),
            "edge_counter": self._edge_counter,
            "edge_confirmation_ticks": self.config.edge_confirmation_ticks,
            "position_age_seconds": self._seconds_since(self._position_opened_at, now),
            "last_rebalance_timestamp": self._last_rebalance_timestamp or None,
            "seconds_since_last_rebalance": self._seconds_since(self._last_rebalance_timestamp, now),
            "rebalance_count": self._rebalance_count,
            "estimated_rebalance_cost": self._as_float(self.cost_model.estimate()),
            "estimated_rebalance_benefit": self._as_float(rebalance_eval.expected_benefit) if rebalance_eval else None,
            "required_rebalance_benefit": self._as_float(rebalance_eval.required_benefit) if rebalance_eval else None,
            "fees_required_to_collect": self._as_float(fee_eval.required_fees) if fee_eval else None,
            "last_action": self._last_action.value,
            "last_reason": self._last_reason,
            "strategy_state": self._state.value,
            "consecutive_failures": self._consecutive_failures,
            "halted": self._halted,
        }
        return info

    @staticmethod
    def _as_float(value: Optional[Decimal]) -> Optional[float]:
        return float(value) if value is not None else None

    def to_format_status(self) -> List[str]:
        # Content is collected first and the box is sized to fit it, so a long pool
        # address or reason string widens the panel instead of breaking its border.
        rows: List[Optional[str]] = []

        def line(text: str = ""):
            rows.append(text)

        def rule():
            rows.append(None)

        now = self.market_data_provider.time()
        position = self._position
        rebalance_eval: Optional[RebalanceEvaluation] = self.processed_data.get("rebalance_evaluation")
        fee_eval: Optional[FeeCollectionEvaluation] = self.processed_data.get("fee_evaluation")

        rule()
        line(f"ADAPTIVE ORCA LP   {self.config.trading_pair}   {self.config.lp_provider} @ {self.config.connector_name}")
        line(f"Pool: {self._pool_address or '(unresolved)'}")
        rule()

        line(f"Price            : {self._fmt(self._pool_price, 6)}")
        line(
            f"Volatility       : {self._fmt(self.volatility_estimator.volatility, 6)}   "
            f"Baseline: {self._fmt(self.volatility_estimator.baseline_volatility, 6)}   "
            f"Ratio: {self._fmt(self.volatility_estimator.volatility_ratio, 2)}   "
            f"Samples: {self.volatility_estimator.sample_count}/{self.config.volatility_window}"
        )
        line(
            f"Regime           : {self._regime.value}   "
            f"(range multiplier {self.range_calculator.regime_multiplier(self._regime)})"
        )
        line(f"State            : {self._state.value}   Last action: {self._last_action.value}")
        line(f"Reason           : {self._last_reason}")
        rule()

        if position is not None:
            adopted = "  (adopted on restart)" if position.adopted else ""
            line(f"Position         : {position.position_address}{adopted}")
            half_width_pct = self.fee_model.half_width_pct(
                position.current_price, position.lower_price, position.upper_price
            ) * Decimal("100")
            line(
                f"LP range         : {self._fmt(position.lower_price, 6)} - {self._fmt(position.upper_price, 6)}"
                f"   (half-width {self._fmt(half_width_pct, 4)}%)"
            )
            if position.lower_tick is not None and position.upper_tick is not None:
                line(f"Ticks            : {position.lower_tick} .. {position.upper_tick}")
            line(
                f"Distance         : lower {self._fmt(position.lower_ratio * 100, 2)}%   "
                f"upper {self._fmt(position.upper_ratio * 100, 2)}%   "
                f"(edge threshold {self._fmt(self.config.edge_threshold * 100, 2)}%)"
            )
            line(
                f"Liquidity        : {self._fmt(position.base_amount, 6)} {self._base_token} + "
                f"{self._fmt(position.quote_amount, 6)} {self._quote_token} = "
                f"{self._fmt(position.liquidity_quote, 4)} {self._quote_token}"
            )
            line(
                f"Fees             : uncollected {self._fmt(position.uncollected_fees_quote, 6)} "
                f"{self._quote_token}   collected {self._fmt(self._collected_fees_quote, 6)} {self._quote_token}"
            )
            if fee_eval is not None:
                line(
                    f"Fee collection   : requires > {self._fmt(fee_eval.required_fees, 6)} {self._quote_token}"
                    f"   -> {'COLLECT' if fee_eval.should_collect else 'ACCRUE'}"
                )
            line(
                f"Edge confirm     : {self._edge_counter}/{self.config.edge_confirmation_ticks}"
            )
            age = self._seconds_since(self._position_opened_at, now)
            since_rebalance = self._seconds_since(self._last_rebalance_timestamp, now)
            line(
                f"Timing           : age {age:.0f}s" if age is not None else "Timing           : age n/a"
            )
            line(
                f"                   since last rebalance "
                f"{f'{since_rebalance:.0f}s' if since_rebalance is not None else 'n/a'}   "
                f"rebalances {self._rebalance_count}"
            )
            for viz_line in self._range_visualization(position).split("\n"):
                line(viz_line)
        else:
            line("Position         : none")
            candidate: Optional[PriceRange] = self.processed_data.get("candidate_range")
            if candidate is not None:
                line(
                    f"Next range       : {self._fmt(candidate.lower_price, 6)} - "
                    f"{self._fmt(candidate.upper_price, 6)}   "
                    f"(half-width {self._fmt(candidate.width_pct * 100, 4)}%)"
                )
            if self._extreme_exit_timestamp is not None:
                remaining = float(self.config.extreme_vol_reopen_cooldown) - (now - self._extreme_exit_timestamp)
                line(f"Reopen cooldown  : {max(0.0, remaining):.0f}s remaining")

        rule()
        cost = self.cost_model.estimate()
        if rebalance_eval is not None:
            line(
                f"Rebalance economics: benefit {self._fmt(rebalance_eval.expected_benefit)} "
                f"{self._quote_token}  |  cost {self._fmt(rebalance_eval.estimated_cost)}  |  "
                f"required {self._fmt(rebalance_eval.required_benefit)}  ->  "
                f"{'REBALANCE' if rebalance_eval.should_rebalance else 'HOLD'}"
            )
        else:
            line(
                f"Rebalance economics: estimated cost {self._fmt(cost)} {self._quote_token} "
                f"(tx {self.config.estimated_tx_cost} + swap {self.config.estimated_swap_cost} + "
                f"slippage {self.config.estimated_slippage}), min profit multiple "
                f"{self.config.min_profit_multiple}"
            )
        if self._consecutive_failures:
            line(
                f"Failures         : {self._consecutive_failures}/{self.config.max_consecutive_failures}"
                f"{'  HALTED' if self._halted else ''}"
            )
        rule()

        box_width = max([100] + [len(text) + 3 for text in rows if text is not None])
        border = "+" + "-" * box_width + "+"
        return [
            border if text is None else f"| {text}".ljust(box_width + 1) + "|"
            for text in rows
        ]

    def _range_visualization(self, position: PositionSnapshot, bar_width: int = 60) -> str:
        """Simple ASCII range bar: '[' and ']' are the bounds, '*' the current price."""
        span = position.range_width
        if span <= 0:
            return f"[{self._fmt(position.lower_price, 6)}] (zero width)"

        bar = ["-"] * bar_width
        bar[0] = "["
        bar[-1] = "]"
        index = int((position.current_price - position.lower_price) / span * (bar_width - 1))
        if index < 0:
            rendered = "* " + "".join(bar)
        elif index >= bar_width:
            rendered = "".join(bar) + " *"
        else:
            bar[index] = "*"
            rendered = "".join(bar)
        lower_label = self._fmt(position.lower_price, 6)
        upper_label = self._fmt(position.upper_price, 6)
        pad = max(1, bar_width - len(lower_label) - len(upper_label))
        return f"{rendered}\n{lower_label}{' ' * pad}{upper_label}"
