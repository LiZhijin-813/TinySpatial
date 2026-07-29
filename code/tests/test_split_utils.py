from pathlib import Path

from code.datasets.split_utils import (
    audit_cross_split_groups,
    build_fair_splits,
    build_split_manifest,
    derive_suspected_group_id,
    select_balanced_subset,
    select_samples_by_case_ids,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _case_ids(samples):
    return {sample["case_id"] for sample in samples}


def test_fair_splits_preserve_canonical_malignant_abc():
    flat4 = build_fair_splits(PROJECT_ROOT, "flat4")
    flat5 = build_fair_splits(PROJECT_ROOT, "flat5")
    dual = build_fair_splits(PROJECT_ROOT, "dual_head")

    assert len(flat4["train"]) == 534
    assert len(flat4["malignant_val"]) == 151
    assert len(flat4["malignant_test"]) == 82
    assert len(flat5["train"]) == 534 + 191
    assert len(dual["train"]) == 534 + 191
    assert _case_ids(flat4["malignant_val"]) == _case_ids(flat5["malignant_val"])
    assert _case_ids(flat4["malignant_test"]) == _case_ids(dual["malignant_test"])
    assert sum(s["class_label"] == 4 for s in dual["train"]) == 191
    assert all(s["class_label"] != 4 for s in flat4["train"])


def test_held_out_benign_samples_never_enter_training():
    splits = build_fair_splits(PROJECT_ROOT, "dual_head")
    train_ids = _case_ids(splits["train"])
    held_out_ids = _case_ids(splits["binary_val"]) | _case_ids(splits["binary_test"])
    benign_held_out = {
        s["case_id"]
        for s in splits["binary_val"] + splits["binary_test"]
        if s["class_label"] == 4
    }
    assert len(benign_held_out) == 41 + 42
    assert train_ids.isdisjoint(benign_held_out)
    assert train_ids.isdisjoint(held_out_ids)


def test_balanced_subset_is_fixed_and_complete():
    samples = [
        {
            "case_id": f"{label}-{index}",
            "class_label": label,
            "subtype_label": label,
            "malignancy_label": 1,
            "split": "train",
            "source": "malignant",
        }
        for label in range(4)
        for index in range(20)
    ]
    first = select_balanced_subset(samples, per_class=8, seed=42)
    second = select_balanced_subset(samples, per_class=8, seed=42)
    assert first == second
    assert len(first) == 32
    assert [sum(s["subtype_label"] == c for s in first) for c in range(4)] == [8, 8, 8, 8]


def test_filename_grouping_is_warning_only():
    assert derive_suspected_group_id("1247-L") == "1247"
    assert derive_suspected_group_id("1247-R") == "1247"
    assert derive_suspected_group_id("078") == "078"
    rows = [
        {"case_id": "1247-L", "split": "train"},
        {"case_id": "1247-R", "split": "val"},
        {"case_id": "078", "split": "test"},
    ]
    audit = audit_cross_split_groups(rows)
    assert audit == [{
        "suspected_group_id": "1247",
        "case_ids": ["1247-L", "1247-R"],
        "splits": ["train", "val"],
    }]


def test_real_metadata_audit_counts_are_explicit():
    from code.datasets.split_utils import load_metadata

    data_dir = PROJECT_ROOT / "data"
    malignant = load_metadata(data_dir / "metadata.csv", "malignant")
    five_class = load_metadata(data_dir / "metadata_5class.csv", "five_class")
    assert len(audit_cross_split_groups(malignant)) == 2
    assert len(audit_cross_split_groups(five_class)) == 31


def test_manifest_has_no_duplicate_ids_and_can_restore_order():
    splits = build_fair_splits(PROJECT_ROOT, "flat4")
    manifest = build_split_manifest("flat4", splits, seed=42)
    for case_ids in manifest["splits"].values():
        assert len(case_ids) == len(set(case_ids))
    requested = manifest["splits"]["malignant_val"][:5]
    restored = select_samples_by_case_ids(splits["malignant_val"], requested)
    assert [sample["case_id"] for sample in restored] == requested
