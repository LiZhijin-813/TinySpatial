# Reliable Stage 2 Baselines Design

Date: 2026-07-13
Project: TinySpatial-ContrastNet
Branch: exp/reliable-stage2-baselines

## Goal

Improve the reliability and early effectiveness of Stage 2 subtype experiments without expanding the paper's core novelty. This first engineering pass focuses on trustworthy evaluation, reproducible experiment entry points, and low-risk imbalance handling.

The main experimental direction is five-class training with malignant four-class subset evaluation:

- Five-class task: Luminal A, Luminal B, HER2+, TNBC, Benign
- Malignant subset: Luminal A, Luminal B, HER2+, TNBC

The first quick experiment should use `metadata_5class.csv` and the available `TinyUSFM.pth` by default. Stage 1 ACM-MIM checkpoints can be swapped in later through the existing pretrained path interface.

## Non-Goals

This pass will not add new paper-level innovations. It will not introduce hierarchical heads, supervised contrastive loss, prototype loss, or major architecture changes. It will not push to the remote repository. It will not commit model checkpoints or large training artifacts.

## Current Problems

1. Validation and test samples currently go through random image augmentation, so repeated evaluation can produce inconsistent metrics.
2. The evaluation utility is hard-coded for four classes and cannot correctly report five-class metrics.
3. Accuracy is currently the main early-stopping signal, which is fragile under class imbalance.
4. The training script lacks a clear switch for balanced sampling or loss variants.
5. Experiment outputs do not yet consistently save full arguments, best metrics, and per-epoch diagnostic fields.

## Design 1: Dataset and Evaluation Reliability

`MultiModalBreastDataset` will expose explicit training/evaluation behavior through an `augment` or equivalent flag.

Training behavior:

- BUS and SWE use shared random geometry transforms to preserve spatial alignment.
- CDFI uses independent training augmentation, including mild crop, flip, rotation, and color jitter.

Validation/test behavior:

- BUS and SWE use deterministic preprocessing only.
- CDFI uses deterministic preprocessing only.
- Re-reading the same validation/test item should produce the same tensors.

`evaluation.py` will be generalized to support any configured class count and class names. It will support both four-class and five-class names:

- Four-class: Luminal A, Luminal B, HER2+, TNBC
- Five-class: Luminal A, Luminal B, HER2+, TNBC, Benign

Evaluation output will include:

- accuracy
- balanced accuracy
- macro precision, macro recall, macro F1
- weighted F1
- per-class precision, recall, specificity, and F1
- confusion matrix
- prediction distribution and label distribution
- HER2+ recall and TNBC recall when those classes exist
- malignant-only subset metrics for five-class experiments

The training loop will use these metrics for logging and model selection.

## Design 2: Stage 2 Experiment Interface

`train_stage2.py` will become the first reliable experiment entry point. It will support:

- `--metadata_file metadata_5class.csv`
- `--num_classes 5`
- `--sampler none|balanced`
- `--loss ce|focal`
- `--monitor_metric acc|macro_f1|balanced_acc|malignant_macro_f1`
- `--eval_malignant_subset`

The default monitor metric for the first pass should be `macro_f1`, because it is less sensitive to class imbalance than raw accuracy.

Each run output directory should save:

- `args.json`
- `history.json`
- `metrics_best.json`
- `best_model.pth`
- `final_model.pth`

Checkpoint files remain ignored by Git through the existing `checkpoints/` ignore rule.

## Design 3: Minimal Training Strategy

The first training improvement will be class imbalance handling, not architecture expansion.

Default strategy:

- Use weighted cross entropy with label smoothing.
- Add optional balanced sampling through `WeightedRandomSampler`.
- Add optional focal loss, but do not enable it by default.
- Keep CAM disabled by default with `beta=0`.
- Keep the existing feature diversity regularizer as a configurable option.

Rationale:

- Balanced sampling directly changes what the model sees in each epoch, which may help TNBC and other minority classes.
- Keeping weighted CE as the default preserves a stable baseline.
- Focal loss is useful for later comparison but should not be mixed into the first default result.
- CAM and more advanced subtype-aware objectives should wait until the reliable baseline is established.

## Design 4: Git and Artifact Management

All code changes will happen on:

```text
exp/reliable-stage2-baselines
```

The first implementation commit should include only code, tests, and lightweight documentation. It should not include checkpoints or generated run directories.

Recommended commit message:

```text
improve stage2 reliability and metrics
```

The branch will not be pushed automatically. After implementation, the user will review the diff, commit hash, tests, and quick experiment output, then decide whether to push.

## Design 5: Quick Experiment

After implementation and smoke tests, run a short five-class experiment to confirm the pipeline and trend:

```bash
python code/train/train_stage2.py \
  --pretrained_path TinyUSFM.pth \
  --metadata_file metadata_5class.csv \
  --num_classes 5 \
  --sampler balanced \
  --monitor_metric macro_f1 \
  --epochs 5 \
  --batch_size 8 \
  --accum_steps 4 \
  --beta 0 \
  --gamma 1
```

The run is not expected to be a final paper result. It should answer whether the pipeline is stable, whether predictions collapse, and whether macro-F1 or malignant subset metrics show a usable trend.

## Acceptance Criteria

Code reliability:

- Dataset tests verify deterministic validation/test transforms.
- Stage 2 tests verify multi-class output and metric wiring.
- Evaluation supports four-class and five-class cases.

Experiment reliability:

- Repeated validation with the same checkpoint produces consistent predictions.
- Each epoch logs accuracy, macro-F1, balanced accuracy, per-class recalls, prediction distribution, label distribution, logits variance, and feature variance.
- Five-class runs also report malignant subset metrics.

Training signal:

- Balanced sampler is available and testable.
- Best checkpoint selection can use macro-F1.
- Quick-run logs make prediction collapse visible through `pred_dist`, `logits_var`, and `feat_var`.

## Next Step After Approval

After this design is approved, invoke the Superpowers writing-plans workflow to create the implementation plan. Implementation must follow that plan and commit changes locally on the experiment branch. No remote push happens without user approval.
