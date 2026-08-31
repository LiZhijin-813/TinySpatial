import pytest
import torch

from code.train.swe_stability_audit import (
    SWEPerturbationDataset,
    apply_swe_perturbation,
    build_shuffle_indices,
    summarize_stability,
    validate_case_limits,
)


def test_intensity_perturbation_uses_image_range_and_clamps():
    swe = torch.tensor([[[[-1.0, 0.0, 1.0]]]])
    result = apply_swe_perturbation(swe, "intensity_1_1")

    assert result[0, 0, 0].tolist() == pytest.approx([-1.0, 0.1, 1.0])


def test_horizontal_flip_only_reverses_width():
    swe = torch.tensor([[[[1.0, 2.0, 3.0]]]])
    result = apply_swe_perturbation(swe, "horizontal_flip")

    assert result.tolist() == [[[[3.0, 2.0, 1.0]]]]


def test_shuffle_indices_are_deterministic_derangement():
    assert build_shuffle_indices(["a", "b", "c"]) == [1, 2, 0]


def test_shuffle_rejects_single_case():
    with pytest.raises(ValueError, match="至少需要两个病例"):
        build_shuffle_indices(["a"])


def test_case_limits_reject_non_fixed_24_12_combination():
    with pytest.raises(ValueError, match="必须固定为 24 个错误病例和 12 个正确病例"):
        validate_case_limits(23, 12)


def test_shuffle_dataset_uses_next_case_as_swe_donor():
    class FakeDataset:
        samples = [
            {"case_id": "a", "subtype_label": 0},
            {"case_id": "b", "subtype_label": 2},
        ]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            sample = self.samples[index]
            return {
                "case_id": sample["case_id"],
                "subtype_label": sample["subtype_label"],
                "bus_img": torch.tensor([float(index)]),
                "swe_img": torch.tensor([float(index + 10)]),
            }

    result = SWEPerturbationDataset(FakeDataset(), "case_shuffle")[0]

    assert result["case_id"] == "a"
    assert result["bus_img"].item() == 0.0
    assert result["swe_img"].item() == 11.0
    assert result["donor_case_id"] == "b"
    assert result["donor_label"] == 2


def test_summary_counts_damage_correction_and_donor_agreement():
    rows = [
        {
            "condition": "case_shuffle",
            "base_correct": True,
            "perturbed_correct": False,
            "prediction_changed": True,
            "confidence_delta": -0.2,
            "perturbed_predicted_label": 2,
            "donor_label": 2,
        },
        {
            "condition": "case_shuffle",
            "base_correct": False,
            "perturbed_correct": True,
            "prediction_changed": True,
            "confidence_delta": 0.1,
            "perturbed_predicted_label": 1,
            "donor_label": 3,
        },
    ]

    summary = summarize_stability(rows)["case_shuffle"]

    assert summary["prediction_changed_count"] == 2
    assert summary["correct_damaged_count"] == 1
    assert summary["error_corrected_count"] == 1
    assert summary["donor_agreement_rate"] == pytest.approx(0.5)
