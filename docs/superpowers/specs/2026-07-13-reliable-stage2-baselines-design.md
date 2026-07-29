# Reliable Stage 2 Baselines Design

Original date: 2026-07-13
Revised after advisor review: 2026-07-29
Project: TinySpatial-ContrastNet
Branch: exp/reliable-stage2-baselines

## Goal

Establish a trustworthy Stage 2 experimental foundation before adding model
complexity. The first engineering pass must answer three questions in order:

1. Can a minimal BUS-only pipeline deliberately overfit a tiny stratified
   subset, proving that labels, loss, gradients, freezing, optimizer wiring, and
   the forward path are functional?
2. Does adding benign training data improve malignant subtype representation
   when every method is evaluated on exactly the same malignant validation and
   test cases?
3. Is a hierarchical dual-head objective more appropriate than a flat
   five-class Softmax for using benign samples as an auxiliary signal?

The paper-level primary task remains malignant four-class molecular subtyping:

- Luminal A
- Luminal B
- HER2+
- TNBC

Benign versus malignant recognition is an auxiliary task. Flat five-class
training remains only as a diagnostic comparison and is not the preferred final
formulation.

## Paper Positioning

The first paper claim should be described as an ROI-free, full-field,
four-modality fine-grained molecular subtyping framework. The term
"four-dimensional" must not be used because in medical ultrasound it can be
misread as 3D plus time.

The three intended paper contributions remain:

1. ROI-free full-field molecular subtyping with BUS, SWE, CDFI, and clinical
   text.
2. ACM-MIM physical-prior pretraining that uses retained SWE information to
   reconstruct highly masked BUS anatomy.
3. Alignment-aware heterogeneous fusion: joint BUS-SWE patch encoding and
   decoupled CDFI injection.

Text design, contrastive learning, LoRA, the benign auxiliary head, and later
dynamic modality weighting are supporting techniques unless experiments show
that one of them warrants a stronger claim.

## Non-Goals

This pass will not:

- Treat flat five-class Accuracy as evidence of improved malignant subtyping.
- Add supervised contrastive loss, prototype loss, or CAM to the first
  trustworthy baseline.
- Add dynamic modality weighting or Luminal/non-Luminal hierarchical subtype
  heads before the minimal and dual-head baselines pass their gates.
- Increase backbone capacity.
- Generate ROI annotations or lesion crops.
- Silently infer patient identity from case filenames and rewrite the official
  data split.
- Commit checkpoints, generated run directories, or large artifacts.
- Push implementation commits without user review.

## Verified Data Facts

The existing metadata files contain the same 767 malignant cases.

Canonical malignant split from `metadata.csv`:

- Train A: 534 cases
- Validation B: 151 cases
- Test C: 82 cases

Malignant cases inside `metadata_5class.csv`:

- Train: the same 534 cases as A
- Validation: 113 cases
- Test: 120 cases

The difference is that 38 malignant cases were moved from validation to test.
Therefore the historical four-class `val_acc=0.391` and five-class
`val_acc=0.539` are not a fair paired comparison.

Benign split in `metadata_5class.csv`:

- Train: 191 cases
- Validation: 41 cases
- Test: 42 cases

Only the 191 benign training cases may be added to model training. Benign
validation and test cases remain held out.

## Data Integrity and Split Contract

### Fair-comparison split

The first controlled experiments must preserve the malignant A/B/C split from
`metadata.csv`.

- Flat four-class training uses malignant A.
- Flat five-class and dual-head training use malignant A plus benign train.
- All three methods are evaluated on the same malignant B and C cases.
- Binary-head validation may additionally use benign validation.
- Binary-head testing may additionally use benign test.

The code must not use the five-class metadata's altered malignant validation and
test split for this comparison.

### Patient-level leakage audit

A filename-based audit found suspected shared base-case groups across splits:

- 2 groups in `metadata.csv`
- 31 groups in `metadata_5class.csv`

This is a warning, not a definitive patient-identity mapping. The implementation
will generate an explicit audit report listing suspected cross-split groups. It
must not silently move cases based only on a filename heuristic.

Paper-ready experiments require a confirmed patient/group identifier and a
patient-level split. If such an identifier is unavailable in metadata, the
legacy fair-comparison split may be used for engineering diagnosis, but it must
not be represented as patient-independent until the suspected groups are
resolved.

## Design 1: Mandatory Tiny-Subset Overfit Gate

Before any multimodal or architectural experiment, add a minimal BUS-only
training mode:

- BUS input only
- TinyUSFM encoder
- One linear four-class head
- 32 malignant cases, stratified as 8 per subtype
- A second 64-case check, stratified as 16 per subtype
- Deterministic preprocessing
- No random augmentation
- No label smoothing
- No class weighting
- No weight decay
- No CAM or feature-diversity loss
- Fully trainable encoder and head

The purpose is pipeline verification, not generalization.

Pass criterion:

- At least 98% training Accuracy, with a target of 100%
- Training loss approaches zero without NaN or Inf
- All four classes appear in the fixed subset and in model predictions
- Encoder and head parameters receive nonzero finite gradients

If the 32-case gate fails, stop model innovation work and inspect:

- case-to-label mapping
- label dtype and range
- logits shape and loss inputs
- train/eval mode
- random augmentation
- frozen parameters
- optimizer parameter groups
- gradient magnitude and optimizer steps
- forward outputs and class-index mapping

The 64-case check runs only after the 32-case check passes.

## Design 2: Deterministic Dataset and Evaluation

`MultiModalBreastDataset` will expose explicit training/evaluation behavior
through an `augment` or equivalent flag.

Training behavior:

- BUS and SWE share geometry transforms.
- CDFI uses independent mild training augmentation.

Validation, test, and overfit behavior:

- BUS, SWE, and CDFI use deterministic preprocessing.
- Re-reading the same item produces identical tensors.
- The overfit subset is saved as an explicit case-ID list so the test is
  reproducible.

Evaluation will support configured class names and task modes rather than a
hard-coded four-class global.

Required metrics:

- Accuracy
- Balanced Accuracy
- macro precision, recall, and F1
- weighted F1
- per-class precision, recall, specificity, and F1
- confusion matrix
- prediction and label distributions
- logits variance and feature variance
- HER2+ and TNBC recall

Repeated evaluation of one checkpoint on validation/test data must produce
identical predictions.

## Design 3: Controlled Task Modes

The Stage 2 model and training entry point will support three comparable modes.

### `flat4`

- Training data: malignant A
- Output: one four-class subtype head
- Purpose: conventional malignant subtype baseline

### `flat5`

- Training data: malignant A plus benign train
- Output: one five-class Softmax head
- Purpose: reproduce the benign-anchor phenomenon as a diagnostic baseline
- Status: comparison only, not the preferred final method

### `dual_head`

- Training data: malignant A plus benign train
- Shared multimodal representation
- Head 1: benign versus malignant logits
- Head 2: four malignant subtype logits
- Head 2 loss is computed only for malignant cases
- Purpose: use benign cases as representation supervision without placing
  benign and molecular subtypes at the same semantic level

The model forward API should return a structured dictionary containing the
available logits and diagnostic features. This avoids positional tuple changes
when task modes differ.

## Design 4: Dual-Head Objective

For each sample:

- `malignancy_label=0` for benign
- `malignancy_label=1` for malignant
- `subtype_label` is 0-3 for malignant
- benign `subtype_label` uses an explicit ignore value such as `-1`

The objective is:

```text
L_total = L_subtype + lambda_bm * L_benign_malignant
```

Default:

```text
lambda_bm = 0.3
```

Loss behavior:

- `L_benign_malignant` is averaged over every sample.
- `L_subtype` is averaged only over malignant samples.
- A benign-only batch returns zero subtype loss without invalid reduction.
- Class weighting is configured separately for the binary and subtype heads.
- Label smoothing is disabled for the first controlled comparison.
- CAM and feature-diversity regularization are disabled for the first
  controlled comparison.

The first sensitivity check compares `lambda_bm` values 0.1, 0.3, and 0.5 only
after the default dual-head run is stable.

## Design 5: Fair Evaluation Semantics

All `flat4`, `flat5`, and `dual_head` comparisons use the same malignant B and C
cases.

For `flat5`, report two malignant evaluations:

1. End-to-end: a malignant case predicted as benign is incorrect.
2. Conditional subtype: restrict/renormalize to the four malignant logits to
   measure subtype representation separately.

For `dual_head`, report:

1. Binary-head performance on combined malignant and held-out benign cases.
2. Conditional malignant subtype performance from Head 2.
3. End-to-end malignant performance where a binary benign prediction is
   incorrect before subtype classification.

Primary model-selection metric for subtype experiments:

```text
malignant_macro_f1
```

Secondary metrics:

- malignant Balanced Accuracy
- per-subtype recall
- HER2+ recall
- TNBC recall
- binary AUC, sensitivity, and specificity
- prediction distribution
- logits variance
- feature variance

Overall flat five-class Accuracy must not be the primary checkpoint or paper
metric.

## Design 6: Experiment Interface and Artifacts

`train_stage2.py` will provide one reproducible interface with arguments
equivalent to:

- `--task_mode overfit|flat4|flat5|dual_head`
- `--malignant_metadata metadata.csv`
- `--benign_metadata metadata_5class.csv`
- `--augment` / `--no_augment`
- `--overfit_samples 32|64`
- `--lambda_bm 0.3`
- `--sampler none|balanced`
- `--monitor_metric malignant_macro_f1|macro_f1|balanced_acc|acc`
- `--eval_malignant_subset`

Focal loss may remain available as an optional later comparison, but weighted
cross entropy is the default baseline loss.

Each run directory saves:

- `args.json`
- `split_manifest.json`
- `history.json`
- `metrics_best.json`
- `best_model.pth`
- `final_model.pth`

The split manifest records exact case IDs used for each task and split.

## Design 7: Controlled Baseline Matrix

### Gate S0

Run the 32-case BUS-only overfit test. Run 64 cases only after it passes.

### Baseline B0

- Mode: `flat4`
- Train: malignant A
- Evaluate: malignant B/C

### Diagnostic B1

- Mode: `flat5`
- Train: malignant A plus benign train
- Evaluate: the same malignant B/C
- Separately evaluate benign validation/test

### Candidate B2

- Mode: `dual_head`
- Train: malignant A plus benign train
- Evaluate: the same malignant B/C
- Separately evaluate the binary head

The first comparison may use one fixed seed to validate behavior. Only stable
configurations proceed to three-seed runs.

B2 is promoted to the working Stage 2 formulation only if:

- it avoids single-class collapse,
- malignant Macro-F1 improves over B0 and the conditional B1 result,
- the improvement is not confined to benign/malignant Accuracy,
- and the trend holds in at least two of three seeds.

## Later Experiments, Not First-Pass Scope

After the gates and controlled baselines pass:

1. Modality ablation:
   BUS; BUS+SWE; BUS+CDFI; BUS+SWE+CDFI; all imaging plus text.
2. Pretraining ablation:
   TinyUSFM; BUS-only MIM; ordinary BUS+SWE MIM; ACM-MIM.
3. Fusion ablation:
   simple concatenation; current alignment-aware fusion; IP-Adapter layer and
   scale variants.
4. Hierarchical subtype auxiliary supervision:
   Luminal versus non-Luminal, Luminal A versus B, and HER2+ versus TNBC.
   The flat four-class head remains primary initially to avoid tree error
   propagation.
5. Dynamic modality weighting:
   add sample-adaptive gates only if modality ablations show inconsistent SWE
   or CDFI utility. Without quality annotations, describe this as adaptive
   weighting rather than supervised quality-aware weighting.

## Testing Requirements

Dataset and split tests:

- deterministic validation/test tensors
- exact malignant A/B/C preservation across task modes
- only benign train is added to training
- benign validation/test never enter training
- split manifests contain no duplicate case IDs
- suspected patient/base-case conflicts are reported

Overfit tests:

- fixed balanced 32/64 case selection
- all expected trainable parameters receive gradients
- disabled augmentation is deterministic
- checkpoint resume preserves the fixed subset

Loss tests:

- binary loss uses every sample
- subtype loss ignores benign samples
- benign-only and malignant-only batches remain finite
- separate class weights have the correct dimensions

Evaluation tests:

- four-class, five-class, binary, conditional, and end-to-end metrics
- malignant B/C case IDs are identical across B0/B1/B2
- repeated evaluation is deterministic
- confusion-matrix dimensions and class names match the configured task

Artifact tests:

- args, split manifest, history, and best metrics are written
- best checkpoint selection uses the configured metric

## Acceptance Criteria

Pipeline integrity:

- 32-case overfit gate reaches at least 98% training Accuracy.
- 64-case gate runs only after the 32-case gate passes.

Data integrity:

- Controlled comparisons use identical malignant B/C cases.
- Only 191 benign training cases are used for training.
- Suspected patient-level cross-split groups are explicitly reported.

Evaluation reliability:

- Validation/test preprocessing is deterministic.
- Repeated evaluation produces identical predictions.
- Malignant Macro-F1 and per-class recall are available for every baseline.

Experimental interpretability:

- Flat five-class Accuracy is not used as proof of subtype improvement.
- Dual-head binary and subtype metrics are reported separately.
- Collapse is visible through prediction distribution, logits variance, and
  feature variance.

## Git and Artifact Management

All work remains on:

```text
exp/reliable-stage2-baselines
```

The revised design document is committed separately from implementation. Future
implementation commits include only code, tests, and lightweight documentation.
Checkpoints and generated runs remain outside Git.

The remote branch is not pushed again without user approval.

## Next Step After Approval

After the user reviews this revised design, invoke the Superpowers
`writing-plans` workflow. The implementation plan must begin with:

1. split and patient-group audit,
2. deterministic evaluation,
3. 32/64-case overfit gate,
4. fair B0/B1/B2 baselines,
5. only then multimodal and paper-level ablations.
