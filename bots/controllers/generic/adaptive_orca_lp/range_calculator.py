"""
Regime detection and dynamic LP range sizing.

Pure functions of (volatility ratio, volatility, price). No Gateway access, no
executor access - so the whole range-selection policy is unit testable and can be
replayed offline by the simulator.
"""
from decimal import Decimal
from typing import Optional

from .models import MarketRegime, PriceRange


class RegimeDetector:
    """
    Classifies the market into a volatility regime from the volatility ratio.

        CALM      ratio <  calm_threshold
        NORMAL    calm_threshold  <= ratio < high_vol_threshold
        HIGH_VOL  high_vol_threshold <= ratio < extreme_threshold
        EXTREME   ratio >= extreme_threshold
    """

    def __init__(self,
                 calm_threshold: Decimal = Decimal("0.75"),
                 high_vol_threshold: Decimal = Decimal("1.50"),
                 extreme_threshold: Decimal = Decimal("2.50")):
        if not (calm_threshold < high_vol_threshold < extreme_threshold):
            raise ValueError(
                "regime thresholds must be strictly increasing: "
                f"calm={calm_threshold}, high_vol={high_vol_threshold}, extreme={extreme_threshold}"
            )
        self.calm_threshold = calm_threshold
        self.high_vol_threshold = high_vol_threshold
        self.extreme_threshold = extreme_threshold

    def detect(self, volatility_ratio: Optional[Decimal]) -> MarketRegime:
        """
        An unknown ratio (not enough history yet) is reported as NORMAL. Callers must
        still gate on the estimator being ready before deploying capital - NORMAL
        here means "no evidence of stress", not "safe to trade".
        """
        if volatility_ratio is None:
            return MarketRegime.NORMAL
        if volatility_ratio < self.calm_threshold:
            return MarketRegime.CALM
        if volatility_ratio < self.high_vol_threshold:
            return MarketRegime.NORMAL
        if volatility_ratio < self.extreme_threshold:
            return MarketRegime.HIGH_VOL
        return MarketRegime.EXTREME


class DynamicRangeCalculator:
    """
    Turns a volatility estimate into concrete LP price bounds.

        width = z_score * volatility * regime_multiplier
        width = clamp(width, min_range_pct, max_range_pct)

        lower_price = price * (1 - width)
        upper_price = price * (1 + width)

    ``width`` is a half-width expressed as a fraction of the current price, so the
    full range spans ``2 * width`` around the price.

    These are *prices*, not ticks. Whirlpool tick indexes and tick-spacing alignment
    are produced by Gateway/the Orca SDK when the position is opened; the realized
    (aligned) bounds are read back from the position afterwards. This module never
    approximates ``log_1.0001(price)`` itself.
    """

    def __init__(self,
                 z_score: Decimal = Decimal("2.0"),
                 min_range_pct: Decimal = Decimal("0.0025"),
                 max_range_pct: Decimal = Decimal("0.05"),
                 calm_multiplier: Decimal = Decimal("0.70"),
                 normal_multiplier: Decimal = Decimal("1.00"),
                 high_vol_multiplier: Decimal = Decimal("1.75"),
                 extreme_multiplier: Decimal = Decimal("3.00")):
        if min_range_pct <= 0:
            raise ValueError("min_range_pct must be positive")
        if max_range_pct <= min_range_pct:
            raise ValueError("max_range_pct must be greater than min_range_pct")
        if max_range_pct >= 1:
            raise ValueError("max_range_pct must be below 1 (a wider range would imply a non-positive lower bound)")
        if z_score <= 0:
            raise ValueError("z_score must be positive")

        self.z_score = z_score
        self.min_range_pct = min_range_pct
        self.max_range_pct = max_range_pct
        self._multipliers = {
            MarketRegime.CALM: calm_multiplier,
            MarketRegime.NORMAL: normal_multiplier,
            MarketRegime.HIGH_VOL: high_vol_multiplier,
            MarketRegime.EXTREME: extreme_multiplier,
        }

    def regime_multiplier(self, regime: MarketRegime) -> Decimal:
        return self._multipliers[regime]

    def calculate_width(self, volatility: Optional[Decimal], regime: MarketRegime) -> Decimal:
        """Raw width clamped into [min_range_pct, max_range_pct]."""
        if volatility is None or volatility < 0:
            volatility = Decimal("0")
        width = self.z_score * volatility * self.regime_multiplier(regime)
        if width < self.min_range_pct:
            return self.min_range_pct
        if width > self.max_range_pct:
            return self.max_range_pct
        return width

    def calculate(self,
                  price: Decimal,
                  volatility: Optional[Decimal],
                  regime: MarketRegime) -> Optional[PriceRange]:
        """
        Build the range around ``price``.

        Returns None for an unusable price rather than raising: a bad Gateway read is
        an expected runtime condition, and the controller's answer to it is to hold,
        not to crash the control loop.
        """
        if price is None or not price.is_finite() or price <= 0:
            return None

        width = self.calculate_width(volatility, regime)
        lower_price = price * (Decimal("1") - width)
        upper_price = price * (Decimal("1") + width)

        # Defence in depth: the clamps above already guarantee this, but a range is
        # never submitted on-chain without the invariant being checked explicitly.
        if lower_price <= 0 or lower_price >= upper_price:
            return None

        return PriceRange(lower_price=lower_price, upper_price=upper_price, width_pct=width)
