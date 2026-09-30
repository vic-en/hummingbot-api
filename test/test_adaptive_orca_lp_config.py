"""A config the validators accept must not crash the controller's __init__.

`AdaptiveOrcaLP.__init__` builds its strategy components -- the volatility estimator, the
range calculator, the fee and cost models -- from the config. Each of those constructors
asserts its own preconditions and raises on a value it cannot use.

That raise lands inside `StrategyV2Base.add_controller`, which logs it and carries on:

    ERROR - Error adding controller: forecast_hours must be positive

and the bot then comes up "healthy" with no controller at all -- status "stopped",
controller "N/A". Nothing says the strategy is empty. It is the same failure this repo
already hit through `lp_rebalancer` (see test_controllers_instantiate) and the reason the
contract belongs on the config, where it surfaces as a 422 before a bot is deployed.

Twelve fields reached a component constructor without the config asserting anything.
"""
from decimal import Decimal

import pytest

pytest.importorskip("hummingbot")

from utils.file_system import fs_util  # noqa: E402

# field -> a value its component constructor refuses
REFUSED_VALUES = {
    "volatility_min_samples": 1,
    "forecast_hours": Decimal("0"),
    "fee_reference_range_pct": Decimal("0"),
    "max_concentration_multiplier": Decimal("0"),
    "pool_fee_rate": Decimal("-0.001"),
    "expected_hourly_volume": Decimal("-1"),
    "min_profit_multiple": Decimal("-1"),
    "estimated_tx_cost": Decimal("-1"),
    "estimated_swap_cost": Decimal("-1"),
    "estimated_slippage": Decimal("-1"),
    "estimated_fee_collection_cost": Decimal("-1"),
    "fee_profit_multiple": Decimal("-1"),
}


@pytest.fixture(scope="module")
def config_class():
    return fs_util.load_controller_config_class("generic", "adaptive_orca_lp")


def _config(config_class, **overrides):
    return config_class(id="test", controller_name="adaptive_orca_lp", **overrides)


@pytest.mark.parametrize("field,value", sorted(REFUSED_VALUES.items()))
def test_a_value_a_component_refuses_is_refused_at_config_load(config_class, field, value):
    """The contract is asserted before a bot exists, not inside __init__ where it is swallowed."""
    with pytest.raises(ValueError):
        _config(config_class, **{field: value})


def test_an_untyped_lp_provider_is_refused_rather_than_guessed(config_class):
    """parse_provider defaults an untyped provider to "router" -- the wrong branch for an LP
    controller, and a trading type Gateway 400s on mid-operation. Same contract as
    lp_rebalancer, enforced here so the call site never reaches the default."""
    with pytest.raises(ValueError, match="expected 'name/type'"):
        _config(config_class, lp_provider="orca")


def test_the_trading_type_comes_from_the_provider(config_class):
    from unittest.mock import MagicMock

    config = _config(config_class, lp_provider="orca/clmm")
    controller = config.get_controller_class()(config, MagicMock(), MagicMock())

    assert (controller.lp_dex_name, controller.lp_trading_type) == ("orca", "clmm")


def test_a_refused_hot_update_keeps_the_running_config(config_class):
    """`ControllerConfigBase.update_config` assigns each updatable field in turn under
    validate_assignment, and pydantic writes the offending value before the model validator
    rejects it. Left alone, the controller keeps managing a live position against a config
    holding a value its own validator refuses, and every later update fails on that stale
    field instead of on whatever the operator just edited."""
    from unittest.mock import MagicMock

    config = _config(config_class)
    controller = config.get_controller_class()(config, MagicMock(), MagicMock())

    incoming = _config(config_class)
    # Bypass validation to build the config a bad YAML edit would produce.
    incoming.__dict__["max_range_pct"] = Decimal("0.0001")

    controller.update_config(incoming)

    assert controller.config.max_range_pct > controller.config.min_range_pct
    # and the config is not poisoned: a later, valid update still lands
    controller.update_config(_config(config_class, edge_threshold=Decimal("0.2")))
    assert controller.config.edge_threshold == Decimal("0.2")
