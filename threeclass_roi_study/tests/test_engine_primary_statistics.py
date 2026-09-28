from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
import torch
from pandas.testing import assert_frame_equal

from threeclass_roi_study.config import EXPECTED_MODELS, EXPECTED_SEEDS
from threeclass_roi_study.engine import (
    _globally_normalized_weighted_loss,
    _paired_primary_comparisons,
    _selection_monitor,
)
from threeclass_roi_study.protocol import ProtocolGateError


def test_weighted_minibatch_loss_preserves_the_global_weighting_target() -> None:
    losses = torch.tensor([1.0, 0.0, 0.0, 0.0])
    weights = torch.tensor([10.0, 1.0, 1.0, 1.0])
    global_mean = weights.mean()
    minibatch_estimates = [
        _globally_normalized_weighted_loss(
            losses[offset : offset + 2],
            weights[offset : offset + 2],
            global_mean,
        )
        for offset in (0, 2)
    ]
    observed = torch.stack(minibatch_estimates).mean()
    expected = (losses * weights).sum() / weights.sum()
    assert float(observed) == pytest.approx(float(expected))

    # The former batch-local ratio normalization changes the declared target.
    former = torch.stack(
        [
            (losses[offset : offset + 2] * weights[offset : offset + 2]).sum()
            / weights[offset : offset + 2].sum()
            for offset in (0, 2)
        ]
    ).mean()
    assert float(former) != pytest.approx(float(expected))


def _probability(label: int, *, correct: bool = True) -> list[float]:
    prediction = label if correct else (label + 1) % 3
    values = np.full(3, 0.05, dtype=float)
    values[prediction] = 0.90
    return values.tolist()


def _primary_config() -> dict:
    return {
        "study_id": "threeclass-statistics-test",
        "config_sha256": "1" * 64,
        "analysis_status": "exploratory_post_hoc_internal",
        "models": list(EXPECTED_MODELS),
        "split_seeds": list(EXPECTED_SEEDS),
        "classifier": {"primary_strategy": "model_specific"},
        "training": {"seed_offsets": {"bootstrap": 50_000}},
        "statistics": {"bootstrap_draws": 5000},
    }


def _primary_patients() -> pd.DataFrame:
    rows: list[dict] = []
    first_seed = EXPECTED_SEEDS[0]
    for seed, model in itertools.product(EXPECTED_SEEDS, EXPECTED_MODELS):
        for label, repetition in itertools.product(range(3), range(2)):
            # Only one model in one seed abstains for one patient in each class.
            # Reusing the patient IDs across seeds makes accidental pooling easy
            # for this fixture to detect.
            abstain = (
                model == "vit_method2"
                and seed == first_seed
                and repetition == 0
            )
            probability = [np.nan, np.nan, np.nan] if abstain else _probability(label)
            rows.append(
                {
                    "patient_id": f"patient_{label}_{repetition}",
                    "label_3class": label,
                    "evaluable": not abstain,
                    "probability_0": probability[0],
                    "probability_1": probability[1],
                    "probability_2": probability[2],
                    "model": model,
                    "seed": seed,
                    "strategy": "model_specific",
                    "probability_state": "calibrated",
                }
            )
    return pd.DataFrame(rows)


def _holm(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def test_selection_monitor_uses_only_declared_tie_break_and_retains_exact_tie() -> None:
    patients = pd.DataFrame(
        {
            "label_3class": [0, 0, 1, 2],
            "evaluable": [True, True, True, True],
            "probability_0": [0.8, 0.7, 0.1, 0.1],
            "probability_1": [0.1, 0.2, 0.8, 0.1],
            "probability_2": [0.1, 0.1, 0.1, 0.8],
        }
    )
    earlier = _selection_monitor(patients)
    exact_tie = _selection_monitor(patients.copy())

    assert earlier["status"] == "evaluable"
    assert len(earlier["key"]) == 2
    assert earlier["key"] == (
        -earlier["macro_nll"],
        earlier["failure_aware_balanced_accuracy"],
    )
    assert "multiclass_nll" not in earlier
    assert exact_tie["key"] == earlier["key"]
    assert not exact_tie["key"] > earlier["key"]


def test_primary_comparisons_are_per_seed_bca_randomization_and_one_holm_family() -> None:
    cfg = _primary_config()
    patients = _primary_patients()

    result = _paired_primary_comparisons(cfg, patients)
    repeated = _paired_primary_comparisons(cfg, patients.sample(frac=1, random_state=8))
    sort_columns = ["seed", "pair_index_within_seed"]
    assert_frame_equal(
        result.sort_values(sort_columns).reset_index(drop=True),
        repeated.sort_values(sort_columns).reset_index(drop=True),
    )

    assert len(result) == 30
    assert result.groupby("seed").size().to_dict() == {
        seed: 6 for seed in EXPECTED_SEEDS
    }
    assert set(zip(result.left_model, result.right_model)) == set(
        itertools.combinations(EXPECTED_MODELS, 2)
    )
    assert result["n_paired_patients"].eq(6).all()
    assert result["bootstrap_draws"].eq(5000).all()
    assert result["valid_bootstrap_draws"].eq(5000).all()
    assert result["ci_method"].eq("bca").all()
    assert result["bootstrap_stratified_by_class"].all()
    assert result["resampling_unit"].eq("patient_within_seed").all()
    assert result["valid_randomization_draws"].eq(5000).all()
    assert result["p_value_method"].str.contains("sign-flip", regex=False).all()
    assert result["pooled_across_seeds"].eq(False).all()  # noqa: E712

    first = result.loc[
        (result.seed == EXPECTED_SEEDS[0])
        & (result.left_model == "yolo26")
        & (result.right_model == "vit_method2")
    ].iloc[0]
    later = result.loc[
        (result.seed == EXPECTED_SEEDS[1])
        & (result.left_model == "yolo26")
        & (result.right_model == "vit_method2")
    ].iloc[0]
    assert first.difference_left_minus_right == pytest.approx(0.5)
    assert later.difference_left_minus_right == pytest.approx(0.0)

    assert result.family_id.nunique() == 1
    assert result.planned_family_size.eq(30).all()
    assert result.exploratory.all()
    assert result.descriptive.eq(False).all()  # noqa: E712
    assert result.confirmatory.eq(False).all()  # noqa: E712
    expected_holm = _holm(result.p_value_randomization_two_sided.to_numpy(float))
    assert np.allclose(result.p_value_holm, expected_holm)


def test_primary_comparisons_reject_incomplete_same_seed_patient_alignment() -> None:
    patients = _primary_patients()
    broken = patients.drop(
        patients.index[
            (patients.seed == EXPECTED_SEEDS[0])
            & (patients.model == "vit_method2")
            & (patients.patient_id == "patient_0_0")
        ][0]
    )
    with pytest.raises(ProtocolGateError, match="Same-seed patient pairing failed"):
        _paired_primary_comparisons(_primary_config(), broken)
