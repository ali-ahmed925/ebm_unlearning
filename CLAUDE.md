# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements **machine unlearning for label-conditioned Energy-Based Models (EBMs)**. The goal is to selectively "forget" a class from a pretrained EBM by reshaping the energy landscape — increasing energy (lowering probability) for forget-set samples while preserving energy structure for the retain set.

## Environment Setup

```bash
pip install -r requirements.txt
```

Key dependencies: PyTorch 2.5.1, torchvision 0.20.1, TensorBoard 2.16.2. `protobuf` is pinned to `3.20.3` due to a TensorBoard 2.16 incompatibility.

## Running the Pipeline

**All hyperparameters** are centralized in `configs/config.yaml`. The primary workflow uses Jupyter notebooks in order:

1. `notebooks/01_data_setup.ipynb` — Download and verify MNIST/CIFAR-10/CIFAR-100
2. `notebooks/02_ebm_pretraining.ipynb` — Pretrain EBM on full dataset
3. `notebooks/03_unlearning.ipynb` — Unlearn the forget class
4. `notebooks/04_evaluation.ipynb` — Evaluate accuracy and energy gaps
5. `notebooks/05_additional_experiments.ipynb` — Extended experiments
6. `notebooks/06_domainnet_pretraining.ipynb` — Pretrain on DomainNet (10-class, multi-domain)
7. `notebooks/07_domainnet_unlearning.ipynb` — Unlearn on DomainNet (cross-domain generalization)

**TensorBoard** (logs written by notebooks):
```bash
tensorboard --logdir outputs/tensorboard
```

**CLI evaluation scripts** (standalone, each takes `--checkpoint`, `--dataset`, `--forget-label`):
```bash
# CIFAR-10
python eval_pretrained_ebm.py --checkpoint outputs/checkpoints/ebm_pretrained.pt --dataset cifar10 --forget-label 0
python eval_unlearned_ebm.py  --checkpoint outputs/checkpoints/ebm_unlearned.pt  --dataset cifar10 --forget-label 0
python eval_classification_ebm.py --checkpoint outputs/checkpoints/ebm_unlearned.pt --dataset cifar10 --forget-label 0

# CIFAR-100 (requires --num-classes 100 --backbone resnet18)
python eval_classification_ebm.py --checkpoint outputs/checkpoints/ebm_pretrained_cifar100.pt \
  --dataset cifar100 --num-classes 100 --backbone resnet18 --forget-label 11

# DomainNet (10-class; add --domain sketch to filter by domain, --filter-class-name tiger to filter by class)
#
# IMPORTANT: DomainNetSubset sorts classes alphabetically (src/data/domainnet.py:54),
# so the forget label is the index in sorted(classes), NOT the position in the
# EXPERIMENT_CLASSES list. For the 10-class subset tiger = 7 (0 would be airplane).
# For the 26-class subset (configs/config_domainnet_subset.yaml) tiger = 23.
python eval_classification_ebm.py --checkpoint outputs/checkpoints/ebm_pretrained_domainnet.pt \
  --dataset domainnet --num-classes 10 --backbone resnet18 --forget-label 7
```

**Analysis tools:**
```bash
python check_similarity.py              # PCA subspace overlap between classes (uses CLIP ViT-B/32)
python check_prediction_distribution.py # Prediction distribution across classes
```

Checkpoints and logs are written to `outputs/`.

## Architecture

### Core Concept

- An EBM `E(x, y)` assigns a scalar energy to each (image, label) pair. Lower energy = higher probability.
- **Inference** (no labels): `ŷ = argmin_y E(x, y)` — prediction is the label with lowest energy.
- **Unlearning**: freeze the pretrained model; train a copy to increase energy for forget-set samples and preserve energy for retain-set samples.

### Model (`src/models/ebm.py`)

`EnergyModel` supports two backbones (`conv` or `resnet18`). Image features are combined with a learned label embedding by **element-wise product** (not concatenation), then passed through a `Linear(embed_dim, 1)` energy head:

```
E(x, y) = W · (f(x) ⊙ e_y) + b
```

i.e. a bilinear form with a diagonal interaction matrix. ResNet variant supports partial freezing of early stages.

A useful consequence: energies for *all* labels factor as `f(x) @ (W ⊙ e)ᵀ + b`, so a full label sweep costs one backbone pass per image rather than one per (image, label) pair. `src/evaluation/locality.py:all_class_energies` uses this; the older `classification.predict_argmin_energy` does the naive replication and is `num_classes` times slower.

### Training (`src/training/`)

- **`pretrain.py`** (`PretrainConfig`, `pretrain_ebm()`): Supervised energy contrast loss with K negative label samples per positive. Uses a shifting trick to guarantee `y_neg ≠ y_true`. Supports early stopping, validation, and checkpoint saving.
- **`unlearn.py`** (`UnlearnConfig`, `unlearn()`): Cycles through forget and retain loaders continuously. Pretrained model is **frozen**; only the unlearned model's weights are updated. Tracks MIA proxy metrics during training.

### Loss Functions (`src/losses/`)

All four unlearning losses are combined via configurable weights (`LossWeights` in `total.py`):

| Loss | File | Purpose |
|------|------|---------|
| `forget_loss` | `forget.py` | Margin contrastive — push forget-set energy **up** |
| `retain_loss` | `retain.py` | Normalized MSE vs. pretrained — preserve retain-set energy |
| `margin_loss` | `margin.py` | Softplus penalty — enforce `E(retain) > E(forget)` globally |
| `energy_l2` | `energy_reg.py` | L2 regularization — prevent unbounded energy growth |

Total: `L = λ_f·L_forget + λ_r·L_retain + λ_m·L_margin + λ_e·L_energy`

### Data (`src/data/`)

- **`dataset.py`**: Loads MNIST, CIFAR-10, or CIFAR-100 with correct normalization. `IndexedSubset` preserves original dataset indices.
- **`split.py`**: Splits by forget class into forget/retain sets, then each into train/holdout (default 80/20).
- **`domainnet.py`**: 10-class DomainNet subset (`EXPERIMENT_CLASSES`; tiger = index 0). Supports 4 domains (real/sketch/clipart/painting). `domain_labels` tensor enables domain-aware splitting for cross-domain generalization experiments.

### Evaluation (`src/evaluation/`)

`evaluate_classification()` in `classification.py` computes overall accuracy, per-class accuracy, confusion matrix, and separately reports forget-class vs. retain-class accuracy.

### Utilities (`src/utils/`)

- **`tracking.py`**: Pluggable tracker interface with `TensorBoardTracker`, `WandbTracker`, and `NullTracker`.
- **`seed.py`**: Seed management (uses CPU-based `torch.Generator` for reproducibility).

## Implementation Notes

- **Chunked computation**: Batched negative label operations support `neg_chunk`/`y_chunk` parameters to avoid OOM on large label spaces.
- **Retain loss normalization**: Divides MSE by mean `|E_pretrained|` to prevent scale drift when energies grow large.
- **Three complementary margin mechanisms**: `forget_loss` (per-sample contrastive), `retain_loss` (anchor to pretrained), and `margin_loss` (global separation) work together — tuning one weight affects the others.
- **Generator seeding**: Negative label sampling uses a seeded CPU generator passed through the call stack for reproducibility.
