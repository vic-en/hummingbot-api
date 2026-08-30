"""
Rolling volatility estimation for the Adaptive Orca LP controller.

Deliberately dependency-free and deterministic so it can be unit tested in
isolation and reused by the offline simulator.
"""
import math
from collections import deque
from decimal import Decimal
from typing import Deque, Optional

from .models import VolatilitySnapshot

# Number of digits kept when converting float statistics back to Decimal. Volatility
# is a ratio, not a traded amount, so this only needs to be finer than any threshold
# the strategy compares it against.
_VOLATILITY_PRECISION = Decimal("0.000000000001")


class VolatilityEstimator:
    """
    Rolling sample standard deviation of logarithmic returns, plus an exponentially
    weighted baseline used to classify the current volatility regime.

        return[t]  = ln(price[t] / price[t-1])
        volatility = stdev(returns)                       # sample stdev, n-1
        baseline   = alpha * volatility + (1 - alpha) * baseline

    The estimator keeps ``window`` returns, i.e. it needs ``window + 1`` prices to be
    fully populated. It starts reporting once ``min_samples`` returns are available.

    ``baseline_alpha`` has to be small enough that the baseline is genuinely slower
    than the rolling window it summarises. At alpha=0.02 the baseline half-life is
    ~35 readings against a 120-sample window, so the baseline tracks the current
    reading almost as fast as the window produces it and the ratio never leaves ~1.0
    - measured over an 8x volatility swing, the ratio peaked at 1.33 and the HIGH_VOL
    and EXTREME regimes never triggered at all. The default here (half-life ~346
    readings) keeps the baseline a long-run reference. See strategy.md.

    Statistics are computed in float (``math.log``) and converted back to Decimal:
    volatility is a dimensionless ratio, never an amount that settles on-chain, so
    float is the right domain for the log/sqrt and Decimal for everything the
    strategy compares or reports.
    """

    def __init__(self,
                 window: int = 120,
                 baseline_alpha: Decimal = Decimal("0.002"),
                 min_samples: int = 30):
        if window < 2:
            raise ValueError("volatility window must be at least 2")
        if not (Decimal("0") < baseline_alpha <= Decimal("1")):
            raise ValueError("baseline_alpha must be in (0, 1]")
        if min_samples < 2:
            raise ValueError("min_samples must be at least 2")

        self._window = window
        self._alpha = baseline_alpha
        self._min_samples = min(min_samples, window)
        self._returns: Deque[float] = deque(maxlen=window)
        self._last_price: Optional[Decimal] = None
        self._baseline: Optional[Decimal] = None
        self._current: Optional[Decimal] = None

    # ------------------------------------------------------------------ properties

    @property
    def window(self) -> int:
        return self._window

    @property
    def sample_count(self) -> int:
        """Number of log returns currently held (not the number of prices seen)."""
        return len(self._returns)

    @property
    def is_ready(self) -> bool:
        return self.sample_count >= self._min_samples

    @property
    def volatility(self) -> Optional[Decimal]:
        return self._current

    @property
    def baseline_volatility(self) -> Optional[Decimal]:
        return self._baseline

    @property
    def volatility_ratio(self) -> Optional[Decimal]:
        """
        ``volatility / baseline_volatility``.

        A zero baseline (a perfectly flat price series) has no meaningful ratio, so
        it reports 1.0 - "exactly as volatile as it has always been" - which lands in
        the NORMAL regime instead of dividing by zero or spuriously reading EXTREME.
        """
        if self._current is None or self._baseline is None:
            return None
        if self._baseline <= 0:
            return Decimal("1")
        return self._current / self._baseline

    # --------------------------------------------------------------------- updates

    def add_price(self, price: Decimal) -> bool:
        """
        Feed one price sample.

        Returns True when the sample produced a new log return. Non-positive or
        non-finite prices are rejected outright: ``ln`` is undefined there and a bad
        Gateway read must never corrupt the rolling window.
        """
        if price is None or not price.is_finite() or price <= 0:
            return False

        previous = self._last_price
        self._last_price = price
        if previous is None:
            return False

        self._returns.append(math.log(float(price) / float(previous)))
        self._recompute()
        return True

    def _recompute(self) -> None:
        if len(self._returns) < self._min_samples:
            return

        n = len(self._returns)
        mean = sum(self._returns) / n
        variance = sum((r - mean) ** 2 for r in self._returns) / (n - 1)
        volatility = Decimal(str(math.sqrt(variance))).quantize(_VOLATILITY_PRECISION)

        self._current = volatility
        if self._baseline is None:
            # Seed the baseline with the first reading so the ratio starts at 1.0
            # rather than spiking through EXTREME on the very first estimate.
            self._baseline = volatility
        else:
            self._baseline = (
                self._alpha * volatility + (Decimal("1") - self._alpha) * self._baseline
            ).quantize(_VOLATILITY_PRECISION)

    def snapshot(self) -> VolatilitySnapshot:
        return VolatilitySnapshot(
            samples=self.sample_count,
            volatility=self._current,
            baseline_volatility=self._baseline,
            volatility_ratio=self.volatility_ratio,
        )

    def reset(self) -> None:
        """Drop all history. Used when the price feed is known to be discontinuous."""
        self._returns.clear()
        self._last_price = None
        self._baseline = None
        self._current = None
