"""Official hard bounds used by optimization validation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DispatchLimits:
    holder_min_m3: float = 30_000.0
    holder_max_m3: float = 180_000.0
    generator_50mw_group_capacity_mw: float = 200.0
    generator_120mw_group_capacity_mw: float = 240.0

    def validate_holder(self, value_m3: float) -> bool:
        return self.holder_min_m3 <= value_m3 <= self.holder_max_m3

    def validate_generation(self, group_50_mw: float, group_120_mw: float) -> bool:
        return (
            0.0 <= group_50_mw <= self.generator_50mw_group_capacity_mw
            and 0.0 <= group_120_mw <= self.generator_120mw_group_capacity_mw
        )

