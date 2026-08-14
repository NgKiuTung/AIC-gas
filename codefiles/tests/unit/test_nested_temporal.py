from __future__ import annotations

import numpy as np
from gas_power.evaluation.nested_temporal import (
    CandidateSpec,
    build_candidate_family,
    predict_candidate,
    select_candidate,
)


def test_candidate_family_is_deterministic_and_contains_persistence() -> None:
    first = build_candidate_family()
    second = build_candidate_family()
    assert first == second
    assert first[0].candidate_id <= first[-1].candidate_id
    assert any(candidate.candidate_id == "persistence" for candidate in first)
    assert len(first) == 451


def test_prediction_uses_only_row_local_inputs() -> None:
    candidate = CandidateSpec("d6", "d6", 0.0, 0.0, 1.0, 0.08, 0.75, 0.55, 0.45)
    current = np.array([100.0, 100.0])
    corrections = np.array([[1.0, 20.0, 10.0], [999.0, 999.0, 999.0]])
    horizon = np.array([1.0, 8.0])
    original = predict_candidate(current, corrections, horizon, candidate)[0]
    corrections[1] = -999999.0
    changed_future = predict_candidate(current, corrections, horizon, candidate)[0]
    assert original == changed_future


def test_selection_prefers_the_known_perfect_candidate() -> None:
    current = np.array([10.0, 10.0])
    corrections = np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    horizon = np.array([1.0, 8.0])
    actual = np.array([12.0, 12.0])
    persistence = CandidateSpec("persistence", "persistence", 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    perfect = CandidateSpec("perfect", "d5", 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    selected, score = select_candidate(actual, current, corrections, horizon, [persistence, perfect])
    assert selected.candidate_id == "perfect"
    assert score == 0.0
