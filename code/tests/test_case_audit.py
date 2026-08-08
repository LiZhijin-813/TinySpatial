"""验证病例级错误审计的可序列化领域产物。"""

import csv
import json
import math
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from code.train.case_audit import (
    build_audit_summary,
    build_case_records,
    validate_output_directory,
    write_audit_outputs,
)
from code.train import audit_stage2 as audit_stage2_module
from code.utils.evaluation import SUBTYPE_NAMES, evaluate_predictions


def _records():
    """构造覆盖正确、亚型错分与端到端良性预测的最小病例集合。"""
    return build_case_records(
        ["病例-甲", "病例-乙", "病例-丙"],
        [0, 1, 2],
        torch.tensor(
            [
                [4.0, 1.0, 0.0, -1.0, -2.0],
                [3.0, 0.0, 2.0, -1.0, -2.0],
                [0.0, 1.0, 2.0, -1.0, 5.0],
            ]
        ),
        [
            [True, True, True, True],
            [True, False, True, True],
            [False, True, True, False],
        ],
    )


def _saved_metrics(records):
    """使用既有评估器构造应当被审计模块复现的保存指标。"""
    return {
        "malignant": evaluate_predictions(
            [record["true_subtype_label"] for record in records],
            [record["conditional_predicted_subtype_label"] for record in records],
            SUBTYPE_NAMES,
        )
    }


def test_build_case_records_preserves_order_and_audits_error_types():
    """病例顺序和三种审计结论必须符合 flat5 语义。"""
    records = _records()

    assert [record["case_id"] for record in records] == ["病例-甲", "病例-乙", "病例-丙"]
    assert [record["error_type"] for record in records] == [
        "正确",
        "亚型错分",
        "恶性病例预测为良性",
    ]
    assert records[0]["true_subtype_name"] == "Luminal A"
    assert records[1]["conditional_predicted_subtype_name"] == "Luminal A"
    assert records[2]["end_to_end_predicted_class_name"] == "良性"


def test_case_record_probabilities_are_finite_and_independently_normalized():
    """五分类和条件四分类概率必须分别是有限且归一的浮点数。"""
    for record in _records():
        for field, expected_size in (("five_class_probabilities", 5), ("conditional_probabilities", 4)):
            probabilities = record[field]
            assert len(probabilities) == expected_size
            assert all(isinstance(value, float) and math.isfinite(value) for value in probabilities)
            assert sum(probabilities) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "case_ids, labels, logits, modalities",
    [
        ([], [], torch.empty((0, 5)), []),
        (["重复", "重复"], [0, 1], torch.zeros((2, 5)), [[True] * 4] * 2),
        (["甲"], [0], torch.zeros((1, 4)), [[True] * 4]),
        (["甲"], [0, 1], torch.zeros((1, 5)), [[True] * 4]),
        (["甲"], [0], torch.tensor([[0.0, 0.0, 0.0, 0.0, float("nan")]]), [[True] * 4]),
        (["甲"], [0], torch.zeros((1, 5)), [[True] * 3]),
    ],
)
def test_build_case_records_rejects_invalid_inputs(case_ids, labels, logits, modalities):
    """记录构造必须拒绝不完整、非有限或不一致的输入。"""
    with pytest.raises(ValueError, match="[\u4e00-\u9fff]"):
        build_case_records(case_ids, labels, logits, modalities)


def test_build_audit_summary_reproduces_saved_malignant_metrics():
    """摘要必须复用评估器并复现保存的恶性区域指标。"""
    records = _records()
    expected = _saved_metrics(records)
    summary = build_audit_summary(records, expected)

    assert summary["malignant"] == expected["malignant"]
    assert summary["error_type_distribution"] == {"正确": 1, "亚型错分": 1, "恶性病例预测为良性": 1}


def test_build_audit_summary_defaults_ablation_modalities_to_empty_list():
    """病例审计汇总默认必须写出空的屏蔽模态列表。"""
    records = _records()
    summary = build_audit_summary(records, _saved_metrics(records))

    assert summary["ablate_modalities"] == []


def test_build_audit_summary_saves_sorted_ablation_modalities():
    """病例审计汇总必须保存排序后的规范化屏蔽模态列表。"""
    records = _records()
    summary = build_audit_summary(
        records,
        _saved_metrics(records),
        ablate_modalities=("text", "swe", "text"),
    )

    assert summary["ablate_modalities"] == ["swe", "text"]


def test_build_audit_summary_accepts_stage2_test_metrics_envelope():
    """审计汇总应支持训练阶段保存的分区嵌套指标格式。"""
    records = _records()
    expected = _saved_metrics(records)
    wrapped_metrics = {
        "malignant": {
            "malignant": expected["malignant"],
            "malignant_end_to_end": {"macro_f1": 0.0},
            "diagnostic": {"logits_var": 0.0},
        },
        "binary": None,
    }

    summary = build_audit_summary(records, wrapped_metrics)

    assert summary["malignant"] == expected["malignant"]


def test_build_audit_summary_rejects_unreproducible_saved_metrics():
    """保存指标与病例记录不一致时必须明确拒绝。"""
    with pytest.raises(ValueError, match="无法复现"):
        build_audit_summary(_records(), {"malignant": {"accuracy": 0.0}})


def test_validate_output_directory_rejects_nonempty_directory_unless_overwritten(tmp_path):
    """非空输出目录只有显式覆盖时才允许继续写入。"""
    output_dir = tmp_path / "审计"
    assert validate_output_directory(output_dir, overwrite=False) == output_dir
    (output_dir / "已有.txt").write_text("内容", encoding="utf-8")

    with pytest.raises(ValueError, match="[\u4e00-\u9fff]"):
        validate_output_directory(output_dir, overwrite=False)
    assert validate_output_directory(output_dir, overwrite=True) == output_dir


def test_write_audit_outputs_rejects_nonempty_directory_unless_overwritten(tmp_path):
    """公开写入接口必须拒绝非空目录，只有显式覆盖时才允许写入。"""
    output_dir = tmp_path / "审计"
    output_dir.mkdir()
    (output_dir / "已有.txt").write_text("内容", encoding="utf-8")
    records = _records()
    summary = build_audit_summary(records, _saved_metrics(records))

    with pytest.raises(ValueError, match="[\u4e00-\u9fff]"):
        write_audit_outputs(output_dir, records, summary)

    write_audit_outputs(output_dir, records, summary, overwrite=True)

    assert (output_dir / "case_predictions.csv").exists()


def test_build_audit_summary_uses_chinese_error_for_missing_saved_metrics():
    """公开异常不得暴露英文保存指标或恶性区域内部字段名。"""
    with pytest.raises(ValueError) as error:
        build_audit_summary(_records(), {})

    assert "保存指标" in str(error.value)
    assert "saved_metrics" not in str(error.value)
    assert "malignant" not in str(error.value)


def _flat5_run(tmp_path, *, metrics=None, manifest_ids=("病例-B", "病例-A")):
    """构造仅含运行产物的最小 flat5 审计目录。"""
    records = _records()
    saved_metrics = metrics or _saved_metrics(records)
    args_path = tmp_path / "args.json"
    manifest_path = tmp_path / "split_manifest.json"
    checkpoint_path = tmp_path / "best_model.pth"
    metrics_path = tmp_path / "metrics_test.json"
    args_path.write_text(json.dumps({
        "task_mode": "flat5",
        "batch_size": 1,
        "img_size": 2,
        "max_text_len": 3,
        "malignant_metadata": "malignant.csv",
        "benign_metadata": "benign.csv",
    }, ensure_ascii=False), encoding="utf-8")
    manifest_path.write_text(json.dumps({
        "task_mode": "flat5",
        "splits": {"malignant_test": list(manifest_ids)},
    }, ensure_ascii=False), encoding="utf-8")
    torch.save({"model_state_dict": {"weight": torch.tensor([1.0])}}, checkpoint_path)
    metrics_path.write_text(json.dumps(saved_metrics, ensure_ascii=False), encoding="utf-8")
    return args_path, manifest_path, checkpoint_path, metrics_path


def _audit_args(run_dir, **overrides):
    """构造不依赖命令行解析的审计参数。"""
    values = {
        "run_dir": Path(run_dir),
        "output_dir": None,
        "device": "cpu",
        "batch_size": None,
        "overwrite": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_audit_parser_exposes_required_run_and_optional_controls():
    """审计入口必须提供受限且可解释的命令行参数。"""
    parser = audit_stage2_module.build_parser()
    args = parser.parse_args(["--run_dir", "运行目录"])

    assert args.run_dir == Path("运行目录")
    assert args.output_dir is None
    assert args.device == "cuda:0"
    assert args.batch_size is None
    assert args.overwrite is False
    assert args.ablate_modalities == []
    with pytest.raises(SystemExit):
        parser.parse_args(["--run_dir", "运行目录", "--batch_size", "0"])


def test_audit_parser_accepts_ablation_modalities():
    """审计入口必须接受零个或多个待屏蔽模态。"""
    args = audit_stage2_module.build_parser().parse_args([
        "--run_dir",
        "杩愯鐩綍",
        "--ablate_modalities",
        "bus",
        "cdfi",
    ])

    assert args.ablate_modalities == ["bus", "cdfi"]


def test_audit_script_help_prefers_project_code_package():
    """直接执行审计脚本时必须优先解析项目 code 包。"""
    result = subprocess.run(
        [sys.executable, str(Path(audit_stage2_module.__file__)), "--help"],
        cwd=audit_stage2_module.PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "flat5 恶性病例错误审计" in result.stdout


@pytest.mark.parametrize(
    "mutate",
    [
        lambda run_dir: (run_dir / "args.json").unlink(),
        lambda run_dir: (run_dir / "split_manifest.json").unlink(),
        lambda run_dir: (run_dir / "best_model.pth").unlink(),
        lambda run_dir: (run_dir / "metrics_test.json").unlink(),
        lambda run_dir: (run_dir / "args.json").write_text(
            json.dumps({"task_mode": "flat4", "batch_size": 1}), encoding="utf-8"
        ),
        lambda run_dir: (run_dir / "split_manifest.json").write_text(
            json.dumps({"task_mode": "flat5", "splits": {"malignant_test": ["重复", "重复"]}}),
            encoding="utf-8",
        ),
    ],
)
def test_load_flat5_audit_run_rejects_incomplete_or_invalid_saved_artifacts(tmp_path, mutate):
    """恢复前必须拒绝缺失、非 flat5 或重复病例的运行产物。"""
    _flat5_run(tmp_path)
    mutate(tmp_path)

    with pytest.raises(ValueError, match="[一-龥]"):
        audit_stage2_module.load_flat5_audit_run(tmp_path)


class _AuditDataset(torch.utils.data.Dataset):
    """不访问真实数据的确定性审计数据集替身。"""

    def __init__(
        self,
        root_dir,
        split,
        img_size,
        max_text_len,
        samples,
        augment,
        ablate_modalities=None,
    ):
        assert root_dir == audit_stage2_module.PROJECT_ROOT
        assert split == "test"
        assert img_size == 2
        assert max_text_len == 3
        assert augment is False
        assert list(ablate_modalities or []) == []
        self.samples = [dict(sample) for sample in samples]
        self.bus_dir = Path(root_dir) / "data" / "images" / "BUS"
        self.swe_dir = Path(root_dir) / "data" / "images" / "SWE"
        self.cdfi_dir = Path(root_dir) / "data" / "images" / "CDFI"
        self.texts_dir = Path(root_dir) / "data" / "texts"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return {
            "case_id": sample["case_id"],
            "bus_img": torch.zeros(1, 2, 2),
            "swe_img": torch.zeros(3, 2, 2),
            "cdfi_img": torch.zeros(3, 2, 2),
            "input_ids": torch.zeros(3, dtype=torch.long),
            "attention_mask": torch.ones(3, dtype=torch.long),
            "subtype_label": sample["subtype_label"],
        }


class _AuditModel:
    """返回固定 flat5 分数并记录严格权重加载的模型替身。"""

    def __init__(self):
        self.loaded = None
        self.evaluated = False

    def load_state_dict(self, state, strict):
        self.loaded = (state, strict)

    def eval(self):
        self.evaluated = True
        return self

    def __call__(self, bus_img, swe_img, cdfi_img, input_ids, attention_mask):
        batch_size = bus_img.shape[0]
        return {"class_logits": torch.tensor(
            [[4.0, 1.0, 0.0, -1.0, -2.0]] * batch_size
        )}


def test_collect_flat5_audit_inputs_rejects_loader_order_different_from_dataset():
    """收集器必须拒绝与恢复样本顺序不同的加载器输出。"""
    dataset = Namespace(
        samples=[{"case_id": "病例-A"}, {"case_id": "病例-B"}],
        bus_dir="不存在/BUS",
        swe_dir="不存在/SWE",
        cdfi_dir="不存在/CDFI",
        texts_dir="不存在/文本",
    )
    loader = [{
        "case_id": ["病例-B", "病例-A"],
        "bus_img": torch.zeros(2, 1, 2, 2),
        "swe_img": torch.zeros(2, 3, 2, 2),
        "cdfi_img": torch.zeros(2, 3, 2, 2),
        "input_ids": torch.zeros(2, 3, dtype=torch.long),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "subtype_label": torch.tensor([0, 1]),
    }]

    with pytest.raises(ValueError, match="[一-龥]"):
        audit_stage2_module.collect_flat5_audit_inputs(
            _AuditModel(), loader, torch.device("cpu"), dataset
        )


def test_run_case_audit_writes_reproduced_artifacts_without_changing_run_files(
    tmp_path,
    monkeypatch,
):
    """完整入口必须以保存配置恢复并只写入新的三个审计产物。"""
    source_paths = _flat5_run(tmp_path, metrics={
        "malignant": evaluate_predictions([0, 0], [0, 0], SUBTYPE_NAMES),
    })
    source_bytes = {path: path.read_bytes() for path in source_paths}
    canonical = {
        "malignant_test": [
            {"case_id": "病例-A", "subtype_label": 0},
            {"case_id": "病例-B", "subtype_label": 0},
        ]
    }
    model = _AuditModel()

    monkeypatch.setattr(audit_stage2_module, "MultiModalBreastDataset", _AuditDataset)
    monkeypatch.setattr(audit_stage2_module, "build_fair_splits", lambda *args, **kwargs: canonical)
    def fake_loader(dataset, args):
        assert args.batch_size == 2
        return [{
            "case_id": [sample["case_id"] for sample in dataset.samples],
            "bus_img": torch.zeros(len(dataset), 1, 2, 2),
            "swe_img": torch.zeros(len(dataset), 3, 2, 2),
            "cdfi_img": torch.zeros(len(dataset), 3, 2, 2),
            "input_ids": torch.zeros(len(dataset), 3, dtype=torch.long),
            "attention_mask": torch.ones(len(dataset), 3, dtype=torch.long),
            "subtype_label": torch.zeros(len(dataset), dtype=torch.long),
        }]

    def fake_model(saved_args, device):
        assert saved_args.batch_size == 1
        return model

    monkeypatch.setattr(audit_stage2_module, "build_eval_loader", fake_loader)
    monkeypatch.setattr(audit_stage2_module, "build_model", fake_model)

    output_dir = audit_stage2_module.run_case_audit(_audit_args(tmp_path, batch_size=2))

    assert output_dir == tmp_path / "case_audit"
    assert model.loaded[1] is True
    assert model.evaluated is True
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "case_audit_summary.json", "case_predictions.csv", "error_audit.md"
    ]
    assert all(path.read_bytes() == source_bytes[path] for path in source_paths)


def test_run_case_audit_rejects_nonempty_output_unless_overwrite(tmp_path, monkeypatch):
    """默认输出目录已有内容时必须在构建模型前拒绝写入。"""
    _flat5_run(tmp_path)
    output_dir = tmp_path / "已有审计"
    output_dir.mkdir()
    (output_dir / "已有.txt").write_text("保留", encoding="utf-8")
    monkeypatch.setattr(
        audit_stage2_module,
        "build_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("不应构建模型")),
    )

    with pytest.raises(ValueError, match="[一-龥]"):
        audit_stage2_module.run_case_audit(_audit_args(tmp_path, output_dir=output_dir))


def test_run_case_audit_rejects_invalid_batch_size_and_unavailable_cuda(tmp_path, monkeypatch):
    """公共入口也必须拒绝无效批次和不可用的 CUDA 设备。"""
    _flat5_run(tmp_path)

    with pytest.raises(ValueError, match="[一-龥]"):
        audit_stage2_module.run_case_audit(_audit_args(tmp_path, batch_size=0))

    monkeypatch.setattr(audit_stage2_module.torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="[一-龥]"):
        audit_stage2_module.run_case_audit(_audit_args(tmp_path, device="cuda:0"))


def test_run_case_audit_propagates_task1_metric_reproduction_failure(tmp_path, monkeypatch):
    """任务 1 的指标复现失败必须阻止任何审计产物写入。"""
    _flat5_run(tmp_path, metrics={"malignant": {"accuracy": 0.0}})
    canonical = {"malignant_test": [
        {"case_id": "病例-B", "subtype_label": 0},
        {"case_id": "病例-A", "subtype_label": 0},
    ]}
    monkeypatch.setattr(audit_stage2_module, "MultiModalBreastDataset", _AuditDataset)
    monkeypatch.setattr(audit_stage2_module, "build_fair_splits", lambda *args, **kwargs: canonical)
    monkeypatch.setattr(audit_stage2_module, "build_eval_loader", lambda dataset, args: [{
        "case_id": ["病例-B", "病例-A"],
        "bus_img": torch.zeros(2, 1, 2, 2),
        "swe_img": torch.zeros(2, 3, 2, 2),
        "cdfi_img": torch.zeros(2, 3, 2, 2),
        "input_ids": torch.zeros(2, 3, dtype=torch.long),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "subtype_label": torch.zeros(2, dtype=torch.long),
    }])
    monkeypatch.setattr(audit_stage2_module, "build_model", lambda *args, **kwargs: _AuditModel())

    with pytest.raises(ValueError, match="无法复现"):
        audit_stage2_module.run_case_audit(_audit_args(tmp_path))

    assert not list((tmp_path / "case_audit").iterdir())


def test_write_audit_outputs_writes_three_files_and_no_high_confidence_branch(tmp_path):
    """审计产物必须按记录顺序写 CSV，并说明无高置信度错分。"""
    records = _records()
    summary = build_audit_summary(records, _saved_metrics(records))

    write_audit_outputs(tmp_path, records, summary)

    csv_path = tmp_path / "case_predictions.csv"
    summary_path = tmp_path / "case_audit_summary.json"
    markdown_path = tmp_path / "error_audit.md"
    assert csv_path.exists() and summary_path.exists() and markdown_path.exists()
    with csv_path.open(encoding="utf-8", newline="") as handle:
        assert [row["case_id"] for row in csv.DictReader(handle)] == ["病例-甲", "病例-乙", "病例-丙"]
    assert json.loads(summary_path.read_text(encoding="utf-8"))["malignant"] == summary["malignant"]
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "来源" in markdown
    assert "无高置信度错分" in markdown
    assert "仅供人工核查、不证明标签错误或模态因果" in markdown
