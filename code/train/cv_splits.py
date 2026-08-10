"""提供病例组约束的确定性分层交叉验证划分。"""

import random
from collections import Counter, defaultdict


def _validate_inputs(samples, n_splits, seed):
    """校验交叉验证输入参数和病例标签。"""
    if not isinstance(samples, (list, tuple)) or not samples:
        raise ValueError("样本必须是非空序列")
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("折数必须是不小于 2 的整数")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("随机种子必须是整数")

    case_ids = []
    class_counts = Counter()
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("每个样本必须是字典")
        case_id = sample.get("case_id")
        label = sample.get("subtype_label")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("病例编号必须是非空字符串")
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < 4:
            raise ValueError("亚型标签必须是 0 到 3 的整数")
        case_ids.append(case_id)
        class_counts[label] += 1
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("病例编号不允许重复")
    missing = [label for label in range(4) if class_counts[label] < n_splits]
    if missing:
        raise ValueError(f"类别样本数不足以构造 {n_splits} 折：{missing}")


def _group_id(case_id):
    """按首个连字符前缀推导疑似基础病例组。"""
    return case_id.split("-", maxsplit=1)[0]


def build_group_stratified_folds(samples, n_splits=5, seed=42):
    """将样本分配到保持病例组完整且类别尽量均衡的验证折。"""
    _validate_inputs(samples, n_splits, seed)

    groups = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[_group_id(sample["case_id"])].append((index, dict(sample)))

    total_counts = Counter(sample["subtype_label"] for sample in samples)
    target_counts = [total_counts[label] / n_splits for label in range(4)]
    target_size = len(samples) / n_splits
    rng = random.Random(seed)
    group_entries = list(groups.items())
    rng.shuffle(group_entries)
    group_entries.sort(
        key=lambda item: (
            -len(item[1]),
            -max(Counter(sample["subtype_label"] for _, sample in item[1]).values()),
        )
    )

    fold_samples = [[] for _ in range(n_splits)]
    fold_counts = [[0, 0, 0, 0] for _ in range(n_splits)]
    fold_sizes = [0 for _ in range(n_splits)]
    for _, group_samples in group_entries:
        group_counts = Counter(sample["subtype_label"] for _, sample in group_samples)
        group_size = len(group_samples)
        empty_folds = [index for index, size in enumerate(fold_sizes) if size == 0]
        candidate_indices = empty_folds[:1] if empty_folds else range(n_splits)
        scores = []
        for fold_index in candidate_indices:
            projected = list(fold_counts[fold_index])
            for label, count in group_counts.items():
                projected[label] += count
            class_error = sum(
                (
                    (projected[label] - target_counts[label]) ** 2
                    - (fold_counts[fold_index][label] - target_counts[label]) ** 2
                )
                / max(target_counts[label], 1.0)
                for label in range(4)
            )
            size_error = (
                (
                    (fold_sizes[fold_index] + group_size - target_size) ** 2
                    - (fold_sizes[fold_index] - target_size) ** 2
                )
                / max(target_size, 1.0)
            )
            scores.append((class_error + size_error, fold_index))
        _, selected = min(scores, key=lambda item: (item[0], item[1]))
        fold_samples[selected].extend(group_samples)
        fold_sizes[selected] += group_size
        for label, count in group_counts.items():
            fold_counts[selected][label] += count

    result = []
    for fold in fold_samples:
        result.append([sample for _, sample in sorted(fold, key=lambda item: item[0])])
    return result
