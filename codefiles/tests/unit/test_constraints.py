from gas_power.optimization import DispatchLimits


def test_official_dispatch_bounds() -> None:
    limits = DispatchLimits()
    assert limits.validate_holder(30_000)
    assert limits.validate_holder(180_000)
    assert not limits.validate_holder(29_999)
    assert limits.validate_generation(200, 240)
    assert not limits.validate_generation(201, 240)

