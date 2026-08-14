"""Deterministic model-spec selection for chronological pseudo-test evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    weight_name: str
    weight_d5: float
    weight_d4: float
    weight_d6: float
    threshold: float
    gate_floor: float
    beta_h15: float
    beta_h120: float

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


WEIGHTS = (
    ("d5", 1.0, 0.0, 0.0),
    ("d4", 0.0, 1.0, 0.0),
    ("d6", 0.0, 0.0, 1.0),
    ("d5_d6", 0.5, 0.0, 0.5),
    ("equal", 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
)
THRESHOLDS = (0.0, 0.03, 0.08, 0.12)
GATE_FLOORS = (0.25, 0.50, 0.75)
BETA_ENDPOINTS = (
    (0.25, 0.25),
    (0.50, 0.50),
    (0.75, 0.75),
    (1.00, 1.00),
    (0.50, 0.25),
    (0.75, 0.50),
    (1.00, 0.75),
    (0.55, 0.45),
    (1.00, 0.80),
)


def build_candidate_family() -> list[CandidateSpec]:
    candidates = [
        CandidateSpec("persistence", "persistence", 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    ]
    for weight_name, weight_d5, weight_d4, weight_d6 in WEIGHTS:
        gates = [(0.0, 1.0)] + [
            (threshold, floor) for threshold in THRESHOLDS[1:] for floor in GATE_FLOORS
        ]
        for threshold, gate_floor in gates:
            for beta_h15, beta_h120 in BETA_ENDPOINTS:
                candidate_id = (
                    f"{weight_name}__t{threshold:.2f}__f{gate_floor:.2f}"
                    f"__b{beta_h15:.2f}-{beta_h120:.2f}"
                )
                candidates.append(
                    CandidateSpec(
                        candidate_id=candidate_id,
                        weight_name=weight_name,
                        weight_d5=weight_d5,
                        weight_d4=weight_d4,
                        weight_d6=weight_d6,
                        threshold=threshold,
                        gate_floor=gate_floor,
                        beta_h15=beta_h15,
                        beta_h120=beta_h120,
                    )
                )
    return sorted(candidates, key=lambda candidate: candidate.candidate_id)


def predict_candidate(
    current: np.ndarray,
    corrections: np.ndarray,
    horizon_step: np.ndarray,
    candidate: CandidateSpec,
) -> np.ndarray:
    current_array = np.asarray(current, dtype=np.float64)
    correction_array = np.asarray(corrections, dtype=np.float64)
    horizon_array = np.asarray(horizon_step, dtype=np.float64)
    if candidate.candidate_id == "persistence":
        return current_array.copy()
    weights = np.asarray(
        [candidate.weight_d5, candidate.weight_d4, candidate.weight_d6], dtype=np.float64
    )
    blended = correction_array @ weights
    if candidate.threshold == 0.0:
        gate = np.ones_like(blended)
    else:
        relative_magnitude = np.abs(blended) / np.maximum(np.abs(current_array), 1e-6)
        magnitude_gate = np.minimum(1.0, relative_magnitude / candidate.threshold)
        gate = candidate.gate_floor + (1.0 - candidate.gate_floor) * magnitude_gate
    horizon_fraction = (horizon_array - 1.0) / 7.0
    beta = candidate.beta_h15 + (candidate.beta_h120 - candidate.beta_h15) * horizon_fraction
    return np.maximum(current_array + beta * gate * blended, 0.0)


def mape(actual: np.ndarray, prediction: np.ndarray) -> float:
    actual_array = np.asarray(actual, dtype=np.float64)
    prediction_array = np.asarray(prediction, dtype=np.float64)
    return float(np.mean(np.abs(prediction_array - actual_array) / np.maximum(np.abs(actual_array), 1e-6)))


def select_candidate(
    actual: np.ndarray,
    current: np.ndarray,
    corrections: np.ndarray,
    horizon_step: np.ndarray,
    candidates: Iterable[CandidateSpec],
) -> tuple[CandidateSpec, float]:
    ranked: list[tuple[float, str, CandidateSpec]] = []
    for candidate in candidates:
        prediction = predict_candidate(current, corrections, horizon_step, candidate)
        ranked.append((mape(actual, prediction), candidate.candidate_id, candidate))
    best_mape, _, best_candidate = min(ranked, key=lambda row: (row[0], row[1]))
    return best_candidate, best_mape
