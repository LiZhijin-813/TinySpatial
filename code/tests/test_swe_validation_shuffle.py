import pytest

from code.train.swe_validation_shuffle import prepare_validation_cases


def test_prepare_validation_cases_uses_only_malignant_validation():
    splits = {
        "malignant_val": [
            {"case_id": "v1", "subtype_label": 0},
            {"case_id": "v2", "subtype_label": 1},
        ],
        "malignant_test": [{"case_id": "t1", "subtype_label": 2}],
    }

    samples, indices = prepare_validation_cases(splits)

    assert [sample["case_id"] for sample in samples] == ["v1", "v2"]
    assert indices == [1, 0]


def test_prepare_validation_cases_rejects_duplicate_ids():
    splits = {"malignant_val": [
        {"case_id": "v1", "subtype_label": 0},
        {"case_id": "v1", "subtype_label": 1},
    ]}

    with pytest.raises(ValueError, match="病例编号必须唯一"):
        prepare_validation_cases(splits)


def test_prepare_validation_cases_pairs_every_case_with_other_subtype():
    samples = [
        {"case_id": f"v{i}", "subtype_label": label}
        for i, label in enumerate([0, 0, 0, 1, 1, 2, 2, 3])
    ]

    _, indices = prepare_validation_cases({"malignant_val": samples})

    assert sorted(indices) == list(range(len(samples)))
    assert all(
        sample["subtype_label"] != samples[indices[index]]["subtype_label"]
        for index, sample in enumerate(samples)
    )


def test_prepare_validation_cases_rejects_unmatchable_class_balance():
    samples = [
        {"case_id": f"v{i}", "subtype_label": label}
        for i, label in enumerate([0, 0, 0, 0, 1, 2])
    ]

    with pytest.raises(ValueError, match="无法为所有病例找到不同亚型供体"):
        prepare_validation_cases({"malignant_val": samples})
