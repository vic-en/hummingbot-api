# Adaptive Orca LP

A Hummingbot V2 controller that manages a single Orca Whirlpool (CLMM) position and
sizes its range from live volatility instead of a fixed percentage.

- Controller: `controllers/generic/adaptive_orca_lp/adaptive_orca_lp.py`
- Execution: the existing `LPExecutor` (`hummingbot/strategy_v2/executors/lp_executor/`)
- Chain access: the existing Gateway connector (`hummingbot/connector/gateway/gateway.py`)

---

## Problem

A concentrated-liquidity position earns fees only while the price is inside its range,
and it earns them in proportion to how tightly that capital is packed. Those two facts
pull in opposite directions, and the right trade-off moves with the market:

- A range sized for a calm market is too tight when volatility triples. The price
  leaves it, the position stops earning, and it ends up fully converted into the
  losing side of the move.
- A range sized for a turbulent market is wastefully wide when things settle down.
  The capital is spread thin, so it captures a small share of the flow that does
  cross it.

A fixed-range LP bot has no view on which of those it is in. It reacts *after* the
position has already stopped earning, and it re-centres on every excursion whether or
not the move pays for the transactions it costs.

## Solution

Estimate volatility continuously, classify it against its own long-run baseline, and
size the range from that estimate. Then guard every rebalance behind hysteresis, a
cooldown and an explicit profitability test, so adaptation does not become churn.

---

## Diagrams

Six flowcharts live in [`diagrams/`](diagrams/), each as `.excalidraw` (editable),
`.svg` and `.png`:

| File | Shows |
|------|-------|
| `01-architecture` | The pipeline from pool price to LP executor |
| `02-control-cycle` | The decision tree run on every control tick |
| `03-rebalance-gates` | The four gates between "near the edge" and a transaction |
| `04-regimes` | Regime thresholds, multipliers and the width formula |
| `05-state-machine` | Controller states and transitions |
| `06-funding-and-recovery` | Autoswap funding and restart recovery |

Import an `.excalidraw` file at excalidraw.com to edit it. All three formats are emitted
from one layout spec, so they cannot drift apart.

## Architecture

```
Orca Whirlpool (Gateway)
        |  pool price
        v
VolatilityEstimator          rolling stdev of log returns + EWMA baseline
        |  volatility ratio
        v
RegimeDetector               CALM / NORMAL / HIGH_VOL / EXTREME
        |  regime
        v
DynamicRangeCalculator       width = z * volatility * regime_multiplier, clamped
        |  candidate range (prices)
        v
Position monitor             boundary ratios, edge hysteresis, cooldown
        |  confirmed edge
        v
EconomicRebalanceFilter      expected benefit vs estimated cost x min_profit_multiple
        |  approved
        v
LPExecutor  ->  Gateway  ->  Orca Whirlpools SDK  ->  Solana
```

Module map:

| File | Responsibility |
|------|----------------|
| `adaptive_orca_lp.py` | `AdaptiveOrcaLPConfig` + `AdaptiveOrcaLP` controller (strategy orchestration, executor actions, status) |
| `volatility.py` | `VolatilityEstimator` - rolling log-return stdev and EWMA baseline |
| `range_calculator.py` | `RegimeDetector`, `DynamicRangeCalculator` |
| `economic_model.py` | `ExpectedFeeModel`, `RebalanceCostModel`, `EconomicRebalanceFilter`, `FeeCollectionPolicy` |
| `models.py` | Enums and data types (`MarketRegime`, `StrategyState`, `PositionSnapshot`, ...) |
| `simulation.py` | Offline fixed-vs-adaptive simulator |

The config class lives in `adaptive_orca_lp.py` alongside the controller because
`ControllerConfigBase.get_controller_class()` resolves the controller by inspecting
`config.__module__`. Splitting them would break controller loading. Everything that
does *not* have that constraint is a separate, dependency-free module.

### Why intra-package imports are relative

Modules inside this package import their siblings relatively (`from .models import ...`),
not absolutely (`from controllers.generic.adaptive_orca_lp.models import ...`), because the
package is imported under two different roots:

- plain Hummingbot imports it as `controllers.generic.adaptive_orca_lp`
- Hummingbot API imports it as `bots.controllers.generic.adaptive_orca_lp`
  (`utils/file_system.py`, `load_controller_config_class`)

With absolute intra-package imports the API path silently resolves `controllers` to
whatever other top-level `controllers` package happens to be on `sys.path` — a different
checkout entirely — so the import fails and the API returns `None` for the config class.
That breaks config editing, validation and backtesting through the API and Condor while
the bot container itself still runs fine, which makes it an easy failure to miss.

### Why the folder and module share a name

`adaptive_orca_lp/adaptive_orca_lp.py`, not `adaptive_orca_lp/adaptive_lp.py`. The API's
package-style controller discovery (`routers/controllers.py`, `list_controllers`) only
recognises a folder as a controller when it contains a `.py` file of the same name. Plain
Hummingbot resolves the package either way, so a mismatch is invisible locally and makes
the controller vanish from the API listing and from Condor's picker.

### Why there is no tick arithmetic here

Whirlpool tick indexes and tick-spacing alignment are produced by Gateway and the Orca
SDK behind it. `LPExecutorConfig` takes `lower_price` / `upper_price`, and the
realized, tick-aligned bounds are read back afterwards from
`Gateway.get_position_info()` (`lowerPrice` / `upperPrice`, plus `lowerBinId` /
`upperBinId`, surfaced in the status panel as ticks). This controller therefore never
approximates `log_1.0001(price)` itself - a client-side approximation would round to
the wrong tick and silently produce a range different from the one the strategy chose.

---

## Regimes

The regime comes from the *ratio* of current volatility to its own baseline, not from
an absolute volatility level, so it means "unusual for this pair" rather than
"unusual for crypto".

| Regime | Ratio | Range multiplier | Behaviour |
|--------|-------|------------------|-----------|
| `CALM` | `< 0.75` | `0.70` | Tighten. Denser liquidity, more fees per unit of capital. |
| `NORMAL` | `0.75 - 1.50` | `1.00` | Baseline width. |
| `HIGH_VOL` | `1.50 - 2.50` | `1.75` | Widen. Fewer rebalances, more time in range. |
| `EXTREME` | `>= 2.50` | `3.00` | Withdraw. See [Safety](#safety). |

All four thresholds and all four multipliers are configurable.

---

## Dynamic Range

```
width = z_score * volatility * regime_multiplier
width = clamp(width, min_range_pct, max_range_pct)

lower_price = price * (1 - width)
upper_price = price * (1 + width)
```

`width` is a **half**-width as a fraction of price, so the full range spans `2 * width`.
With the defaults (`z=2.0`, clamps `[0.0025, 0.05]`) the range can never be tighter
than +/-0.25% or wider than +/-5%, whatever the volatility estimate says.

### Volatility

```
return[t] = ln(price[t] / price[t-1])
volatility = stdev(returns)                        # sample stdev, n-1 denominator
baseline   = alpha * volatility + (1 - alpha) * baseline
ratio      = volatility / baseline
```

The baseline is seeded with the first volatility reading, so the ratio starts at 1.0
rather than spiking through EXTREME on the first estimate. A perfectly flat series
(zero volatility, zero baseline) reports a ratio of 1.0 instead of dividing by zero.

Samples are taken every `volatility_sample_interval` seconds rather than once per
control tick. Gateway's pool-info response is TTL-cached for 5 seconds, so sampling on
every tick would feed the same cached price into the estimator repeatedly and make the
window length a function of tick rate instead of wall-clock time. At the defaults, 120
samples at 5s is a 10-minute horizon.

> **Deviation from the original specification.** The spec called for
> `baseline_alpha = 0.02`. Measured against an 8x volatility swing in the simulator,
> that value keeps the volatility ratio between 0.42 and 1.33 - the `HIGH_VOL` and
> `EXTREME` regimes never trigger at all, which makes regime detection and the
> extreme-volatility protection dead code. The cause is that an EWMA at alpha=0.02 has
> a half-life of ~35 readings, so the "long-run baseline" tracks the 120-sample rolling
> window almost as fast as the window produces it. The default here is **0.002**
> (half-life ~346 readings), which exercises all four regimes on the same series. The
> parameter is still configurable; the rule is that `baseline_alpha` must be well below
> `1 / volatility_window`.

---

## Rebalance Protection

Reaching the edge of the range is necessary but never sufficient. Four independent
gates stand between "the price is close to a boundary" and "submit a transaction":

1. **Edge threshold.** For a position with `range_width = upper - lower`:

   ```
   lower_ratio = (price - lower) / range_width
   upper_ratio = (upper - price) / range_width
   near_edge   = lower_ratio < edge_threshold or upper_ratio < edge_threshold
   ```

   At the default `edge_threshold = 0.10`, the outer 10% of the range on either side
   counts as "near edge".

2. **Confirmation / hysteresis.** `edge_confirmation_ticks` (default 5) consecutive
   volatility samples must be near the edge. Any sample back in the middle resets the
   counter to zero. A single wick cannot trigger a rebalance.

3. **Cooldown.** `rebalance_cooldown` (default 900s) must have elapsed since the last
   rebalance. The clock starts when the rebalance is *approved*, so a rebalance that
   fails on-chain still consumes the cooldown and cannot spin.

4. **Economic filter.** See below. This is the gate that actually decides.

---

## Economics

### Expected fees

```
utilization   = clamp(2 * distance_to_nearest_boundary / range_width, 0, 1)
concentration = clamp(fee_reference_range_pct / half_width_pct, 0, max_concentration_multiplier)
expected_fees = expected_hourly_volume * pool_fee_rate * forecast_hours
                * utilization * concentration
```

`utilization` is 1.0 with the price centred and 0.0 at (or beyond) a boundary.

> **Deviation from the original specification.** The spec's form was
> `utilization = 1 - distance_to_nearest_boundary / range_width`, which *peaks* when the
> price sits on a boundary - exactly when a CLMM position has stopped earning. Used as a
> fee weight it makes an about-to-exit range look maximally productive, so the benefit
> of re-centring is always negative and the filter never approves anything. The
> complementary form above is used instead. (The spec's quantity is a reasonable measure
> of *capital conversion*; it is not a measure of fee capture.)

`concentration` is an addition, not in the original spec. Without it, range width would
not affect expected fees at all, a wider range would dominate on every metric, and the
fixed-vs-adaptive comparison would be meaningless. It expresses the core CLMM property
that the same capital over a narrower range is denser liquidity taking a larger share
of the flow that crosses it, normalised against the width at which
`expected_hourly_volume` was quoted and capped so a degenerate range cannot imply
unbounded fees.

> **Calibrate `expected_hourly_volume`.** It is the quote volume expected to route
> through *this position's range* per hour - pool volume multiplied by your liquidity
> share - not the pool's total volume. The spec's default of 100000 is a pool-scale
> number: with a 0.003 fee rate it implies up to 300 quote/hour of fees against a 0.62
> rebalance cost, at which ratio the economic filter approves essentially every
> rebalance. The controller logs this ratio at startup and warns when it exceeds 100x.
> This model is deliberately crude and isolated in `ExpectedFeeModel` so it can be
> replaced by a measured estimator without touching the controller.

### Rebalance cost and the decision

```
estimated_cost   = estimated_tx_cost + estimated_swap_cost + estimated_slippage
required_benefit = estimated_cost * min_profit_multiple

expected_benefit = max(0, fees(new re-centred range) - fees(current range))
rebalance        <=>  expected_benefit > required_benefit
```

The benefit is the *incremental* fee expectation, floored at zero: a candidate range
worse than the one already deployed is never a reason to pay for a rebalance. Costs
are flat quote amounts rather than percentages - on Solana the chain cost genuinely is
flat, and a flat floor never under-charges a small position the way a percentage would.

Every evaluation is logged in full:

```
REBALANCE EVALUATION
  current range fees : 0.0000 USDC
  new range fees     : 6.7228 USDC
  expected benefit   : 6.7228 USDC
  estimated cost     : 0.6200 USDC
  required benefit   : 1.2400 USDC
  decision           : REBALANCE
```

### Fee collection

`LPExecutor` already harvests fees whenever it closes a position, so mid-position
collection only covers the in-between case:

```
collect  <=>  uncollected_fees > estimated_fee_collection_cost * fee_profit_multiple
```

There is no executor path for a standalone collect, so this one call goes through
`GatewayHttpClient.clmm_collect_fees()` - the same client the `lp collect-fees` CLI
command uses. It touches fees only, never liquidity, so it cannot conflict with the
executor's lifecycle. One consequence: the executor derives unrealized P&L from the
position's *uncollected* fees, so its reported P&L drops by whatever is harvested. The
controller tracks the harvested total itself and reports it separately. Set
`enable_fee_collection: false` to leave all fees to be swept at close.

---

## Safety

### Extreme volatility

When the ratio reaches `extreme_vol_threshold` the controller stops adding liquidity
and withdraws what it has, via `StopExecutorAction(keep_position=True)` - the close
harvests the accrued fees as part of the same transaction, so no separate collect is
needed or attempted. It then stays flat for `extreme_vol_reopen_cooldown` (default
1800s). The cooldown clock is re-stamped on *every* extreme cycle, not just the one
that closes the position, so the strategy cannot redeploy the instant the ratio dips
back under the threshold in the middle of the same turbulence.

### Everything else

| Risk | Mitigation |
|------|-----------|
| Missing / invalid pool price | Non-finite, zero and negative prices are rejected before they reach the estimator; the cycle holds |
| Insufficient volatility history | No capital is deployed until `volatility_min_samples` returns exist |
| Invalid range, `lower >= upper` | Clamped in `DynamicRangeCalculator`, then re-checked in `_build_lp_executor_config` before submission |
| Duplicate positions | One executor invariant, an on-chain scan before the first open, and a `OPEN_REQUEST_TIMEOUT` guard while a create is in flight |
| Duplicate rebalance requests | `_rebalance_requested_for` records the executor already asked to stop |
| Acting mid-transaction | `OPENING` / `CLOSING` / `SWAPPING` executor states and an in-flight fee collection all block new instructions |
| Transaction frequency | Edge confirmation + rebalance cooldown + fee-collection cooldown |
| Gateway / executor failure | `CloseType.FAILED` increments a failure counter, sets an escalating backoff (`error_backoff_seconds x failures`), and halts after `max_consecutive_failures` |
| Restart with a live position | See [Recovery](#recovery) |
| Unexpected position payloads | Every Gateway field goes through a `Decimal` conversion that rejects junk; a snapshot that fails validation is simply not monitored |
| Insufficient balances | Checked against wallet balances, with the chain's native-currency buffer, before the open |
| Precision | `Decimal` throughout; token amounts quantized with `ROUND_DOWN` so rounding can never overspend |
| Controller stops ticking | `safety_exit_pct` sets executor-level limit prices that auto-close the position if price runs far past the bounds |

A failed rebalance never reports `ACTIVE`. If the close succeeds and the reopen fails,
the executor terminates `FAILED`, the controller clears its position state, reports
`ERROR`, and retries after the backoff.

---

## Recovery

On the first control cycle, before any capital can be deployed, the controller queries
`Gateway.get_user_positions()` for the configured pool.

- **No unmanaged position** - normal start.
- **One unmanaged position** - it is *adopted*. The controller monitors it through
  `get_position_info()` and runs the identical boundary/hysteresis/economic logic
  against it. When a rebalance is approved (or extreme volatility hits), it closes the
  adopted position through `Gateway._clmm_close_position()` - the same connector call
  `LPExecutor` makes - and the next cycle opens a fresh, executor-managed position.
  Recovery is therefore complete, not read-only. Note that an adopted position's P&L is
  not attributed to any executor, since none owned it.
- **More than one** - a warning naming every address, the lowest-sorted one is
  monitored, and **no new position is opened** until the extras are closed manually
  (`lp close-position`).

If the recovery query itself fails, the flag stays unset and the next cycle retries.
The controller will not open anything while recovery is outstanding, so a Gateway
outage at startup can never cause a duplicate position.

---

## State machine

Controller states are a view over the executor lifecycle plus the controller's own
bookkeeping, not a competing state machine.

| State | Meaning |
|-------|---------|
| `NO_POSITION` | No liquidity deployed; free to open once every gate is clear |
| `ACTIVE` | Position deployed and monitored |
| `REBALANCING` | A close is requested or in flight; nothing is deployed yet |
| `COLLECTING` | A collect-fees transaction is in flight |
| `PAUSED` | Extreme volatility, its reopen cooldown, or multiple positions needing manual cleanup |
| `ERROR` | Executor/Gateway failure; backing off, or halted |

Actions: `HOLD`, `OPEN`, `REBALANCE`, `COLLECT_FEES`, `CLOSE`.

---

## Configuration

Copy `adaptive_orca_lp_sol_usdc.example.yml` into `conf/controllers/`.

### Market

| Parameter | Default | Description |
|-----------|---------|-------------|
| `connector_name` | `solana-mainnet-beta` | Gateway network connector |
| `lp_provider` | `orca/clmm` | LP provider as `dex/trading_type` |
| `trading_pair` | `SOL-USDC` | Pool trading pair |
| `pool_address` | `''` | Whirlpool address. Blank resolves it from `trading_pair` via Gateway |
| `total_amount_quote` | `1000` | Capital deployed, in quote currency (the spec's `capital_quote`) |

### Volatility

| Parameter | Default | Description |
|-----------|---------|-------------|
| `volatility_window` | `120` | Log returns held in the rolling window |
| `volatility_min_samples` | `30` | Returns required before any capital is deployed |
| `volatility_sample_interval` | `5` | Seconds between samples |
| `baseline_alpha` | `0.002` | EWMA weight for the baseline. Must be well below `1 / volatility_window` |

### Dynamic range

| Parameter | Default | Description |
|-----------|---------|-------------|
| `z_score` | `2.0` | Standard deviations covered by the half-width |
| `min_range_pct` | `0.0025` | Half-width floor |
| `max_range_pct` | `0.05` | Half-width ceiling |
| `calm_multiplier` | `0.70` | Range multiplier in CALM |
| `normal_multiplier` | `1.00` | Range multiplier in NORMAL |
| `high_vol_multiplier` | `1.75` | Range multiplier in HIGH_VOL |
| `extreme_multiplier` | `3.00` | Range multiplier in EXTREME (used only if a range is ever built there) |

### Regimes

| Parameter | Default | Description |
|-----------|---------|-------------|
| `calm_threshold` | `0.75` | Below this ratio the market is CALM |
| `high_vol_threshold` | `1.50` | At or above this ratio the market is HIGH_VOL |
| `extreme_vol_threshold` | `2.50` | At or above this ratio the market is EXTREME |

### Boundary monitoring

| Parameter | Default | Description |
|-----------|---------|-------------|
| `edge_threshold` | `0.10` | Boundary ratio below which the position counts as near-edge |
| `edge_confirmation_ticks` | `5` | Consecutive near-edge samples required |
| `rebalance_cooldown` | `900` | Seconds between rebalances |

### Economics

| Parameter | Default | Description |
|-----------|---------|-------------|
| `estimated_tx_cost` | `0.02` | Chain fees and rent per rebalance, in quote |
| `estimated_swap_cost` | `0.50` | Swap fee to rebalance the token mix |
| `estimated_slippage` | `0.10` | Price impact of that swap |
| `min_profit_multiple` | `2.0` | Benefit must exceed cost by this multiple |
| `pool_fee_rate` | `0.003` | Pool swap fee rate |
| `expected_hourly_volume` | `100000` | Quote volume through *this range* per hour. **Calibrate.** |
| `forecast_hours` | `1` | Forecast horizon |
| `fee_reference_range_pct` | `0.01` | Half-width at which `expected_hourly_volume` is quoted |
| `max_concentration_multiplier` | `10` | Cap on the concentration factor |

### Fee collection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_fee_collection` | `true` | Harvest fees between rebalances |
| `estimated_fee_collection_cost` | `0.02` | Cost of one collect-fees transaction |
| `fee_profit_multiple` | `2.0` | Fees must exceed the cost by this multiple |
| `fee_collection_cooldown` | `300` | Seconds between collections |

### Autoswap

A re-centring CLMM position converts toward one token every time the price moves, so the
wallet mix drifts away from the ratio the next range needs. Without a swap the strategy
stalls on `insufficient balance` precisely when a rebalance is due. When the planned
position is short one token and holds a surplus of the other, the controller submits a
market `OrderExecutor` swap, waits for it to settle, then re-quotes and opens. If *both*
tokens are short it does not swap — that is a capital shortfall, not a mix problem.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `autoswap` | `true` | Swap into the token mix the new range needs |
| `swap_buffer_pct` | `1` | Extra percent swapped beyond the deficit, to absorb slippage |
| `max_consecutive_swap_failures` | `3` | Failed swaps before autoswap gives up and holds |

### Protection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `extreme_vol_reopen_cooldown` | `1800` | Seconds flat after extreme volatility |
| `safety_exit_pct` | `0.02` | Executor-level auto-close distance beyond the bounds; `0` disables |
| `max_consecutive_failures` | `5` | Failures before the controller halts |
| `error_backoff_seconds` | `300` | Base backoff, multiplied by the failure count |
| `slippage_pct` | `1` | Slippage passed to Gateway when quoting position amounts |
| `min_position_notional_quote` | `5` | Refuse to open a smaller position |
| `status_log_interval` | `30` | Seconds between periodic status lines; `0` disables |

---

## Running

Requires a running Gateway with a funded Solana wallet. Nothing below uses real funds
except the live run itself.

### 1. Install the two config files

```bash
# The controller config
cp controllers/generic/adaptive_orca_lp/adaptive_orca_lp_sol_usdc.example.yml \
   conf/controllers/adaptive_orca_lp_sol_usdc.yml

# The script config that points v2_with_controllers.py at it
cp controllers/generic/adaptive_orca_lp/conf_v2_with_controllers_adaptive_orca_lp.example.yml \
   conf/scripts/conf_v2_with_controllers_adaptive_orca_lp.yml
```

Both directories are gitignored, which is why the templates ship inside the controller
package. To build them through the client instead of copying:

```
>>> create --controller-config adaptive_orca_lp     # writes to conf/controllers/
>>> create --v2-config v2_with_controllers          # writes to conf/scripts/
```

### 2. Connect Gateway

```bash
./start                    # requires the hummingbot conda env to be active
```

```
>>> gateway ping solana                 # confirm Gateway is reachable
>>> gateway connect solana              # add the wallet
>>> gateway balance solana              # confirm SOL and USDC are funded
>>> gateway pool orca/clmm SOL-USDC     # see the Whirlpool the controller will resolve
```

Check both mint addresses on that pool before funding anything. Leaving
`pool_address: ''` lets the controller resolve it through the same lookup and log the
address it chose.

### 3. Run

```
>>> start --v2 conf_v2_with_controllers_adaptive_orca_lp.yml
>>> status                 # renders the panel below
>>> status --live          # auto-refreshing, best for a demo
```

The controller needs `volatility_min_samples x volatility_sample_interval` seconds of
price history (10 minutes at the defaults) before it deploys anything. Until then the
status reason reads `insufficient volatility history (n/30 samples)`. For a shorter
demo warm-up, lower `volatility_min_samples` and `volatility_window`.

Useful while it runs:

```
>>> gateway lp orca/clmm position-info SOL-USDC   # the position on-chain
>>> lphistory -v                                  # opens, closes and fees recorded
>>> stop                                          # closes the position via the executor
```

Non-interactively, e.g. in Docker:

```bash
./start -p <password> --v2 conf_v2_with_controllers_adaptive_orca_lp.yml
```

### Status panel

```
+----------------------------------------------------------------------------+
| ADAPTIVE ORCA LP   SOL-USDC   orca/clmm @ solana-mainnet-beta               |
| Pool: <whirlpool address>                                                   |
+----------------------------------------------------------------------------+
| Price            : 150.230000                                              |
| Volatility       : 0.001820   Baseline: 0.001210   Ratio: 1.50  Samples: 120|
| Regime           : HIGH_VOL   (range multiplier 1.75)                       |
| State            : ACTIVE   Last action: HOLD                               |
| Reason           : edge confirmation 3/5                                    |
+----------------------------------------------------------------------------+
| Position         : <position address>                                       |
| LP range         : 147.910000 - 152.550000   (half-width 1.5443%)           |
| Ticks            : -12352 .. -12032                                         |
| Distance         : lower 50.00%   upper 4.90%   (edge threshold 10.00%)     |
| Liquidity        : 3.320000 SOL + 500.000000 USDC = 998.7200 USDC           |
| Fees             : uncollected 43.210000 USDC   collected 12.340000 USDC    |
| Edge confirm     : 3/5                                                      |
| [------------------------*----------------------------------------------]  |
+----------------------------------------------------------------------------+
| Rebalance economics: benefit 7.4200 USDC | cost 0.6200 | required 1.2400 -> REBALANCE |
+----------------------------------------------------------------------------+
```

The same fields are published through `get_custom_info()` for MQTT/API consumers.

---

## Testing

### Unit tests

Fully mocked - no Gateway, no network, no wallet.

```bash
conda run -n hummingbot python -m pytest test/controllers/generic/adaptive_orca_lp/ -q
```

| File | Covers |
|------|--------|
| `test_volatility.py` | Constant prices, known returns, insufficient history, baseline EWMA, window roll-off, invalid prices |
| `test_range_calculator.py` | All four regimes and their boundaries, min/max clamps, dynamic width, invalid inputs |
| `test_economic_model.py` | Utilization, concentration, expected fees, cost model, filter accept/reject, fee-collection threshold |
| `test_adaptive_orca_lp.py` | Open flow, sizing, boundary detection, edge confirmation and reset, cooldown, economic veto, extreme volatility, fee collection, restart recovery, duplicate prevention, failure recovery, metrics |
| `test_simulation.py` | Series determinism, both policies, simulator accounting, fixed-vs-adaptive comparison |

### Simulation and comparison

```bash
conda run -n hummingbot python scripts/backtest_adaptive_orca_lp.py
conda run -n hummingbot python scripts/backtest_adaptive_orca_lp.py --steps 4320 --seed 7
conda run -n hummingbot python scripts/backtest_adaptive_orca_lp.py --fixed-width 0.005 --json
conda run -n hummingbot python scripts/backtest_adaptive_orca_lp.py --shock
```

`--shock` swaps in a schedule with an abrupt volatility burst instead of a gentle ramp.
That is what it takes to push the ratio past 2.50, so it is the scenario that actually
exercises the EXTREME withdrawal and the reopen cooldown.

The simulator replays one seeded price series - calm, normal, stressed, back to
normal - through both strategies using the *same* volatility estimator, regime detector,
range calculator, economic filter and fee/cost models the live controller uses, and
reports fees, rebalance count, costs, time in and out of range, and net return.

It is not an Orca emulator. Divergence (impermanent) loss, swap routing, Whirlpool tick
spacing, pool composition and other LPs' fee share are not modelled; reported P&L is
fee income net of rebalance costs. Absolute fee levels scale linearly with
`--expected-hourly-volume`; the comparison does not, but the economic filter's bite
does, so keep volume and costs on the same scale.

### Integration testing

There is no integration-test harness for LP executors in this repository, and adding
one would require a funded devnet wallet, so none is included. The equivalent manual
check is to point the config at a devnet Whirlpool with `total_amount_quote` set to a
few dollars and watch the status panel through one full open/edge/rebalance cycle.
