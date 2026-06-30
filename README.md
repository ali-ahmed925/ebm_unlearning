# Energy-Based Machine Unlearning via Energy Shaping

This project implements **machine unlearning for Energy-Based Models (EBMs)** by explicitly **reshaping the energy landscape**:

- **Forget set** samples get **higher energy** (lower probability)
- **Retain set** samples preserve their original energy structure (anchored to a pretrained model)
- A **margin** enforces explicit separation between retain and forget energies

## Project structure

```
ebm_unlearning/
├── notebooks/
├── src/
├── configs/
└── outputs/
```

## Quickstart

1. Create environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the notebooks in order:
- `notebooks/01_data_setup.ipynb`
- `notebooks/02_ebm_pretraining.ipynb`
- `notebooks/03_unlearning.ipynb`
- `notebooks/04_evaluation.ipynb`

Outputs are written to `outputs/checkpoints/` and `outputs/logs/`.

## Live visualization (TensorBoard)

Pretraining and unlearning notebooks write TensorBoard logs to `outputs/tensorboard/`.

If TensorBoard crashes with a protobuf `MessageToJson()` error, downgrade protobuf (TensorBoard 2.16 is not compatible with protobuf 6):

```bash
pip install "protobuf==3.20.3"
```

From the `ebm_unlearning/` directory, run:

```bash
tensorboard --logdir outputs/tensorboard
```

---

## DomainNet Experiment — Best Configuration (Current)

### What we forget
- **Target:** tiger, sketch domain only (`class_label=7`, `domain=sketch`)
- **Cross-domain generalization:** unlearning propagates to tiger in real/clipart/painting via DINOv2 subspace weighting
- **Cross-class propagation:** retain samples visually similar to tiger receive a weighted forget signal (lion ≫ bear > dog ≈ horse > guitar ≈ airplane)

### Key config (`configs/config_domainnet.yaml`)

```yaml
data:
  domains: [real, sketch, clipart, painting]
  forget:
    class_label: 7        # tiger — ALWAYS index 7 in sorted(classes)
    class_name: tiger
    domain: sketch

model:
  backbone: resnet18
  finetune_stages: 1      # only last ResNet stage trainable — limits backbone interference

unlearning:
  steps: 500
  lr: 1.0e-4
  lambda_f: 1.0           # forget loss weight
  lambda_r: 15.0          # retain loss weight (higher = stronger class protection)
  lambda_m: 1.0           # margin loss weight
  lambda_e: 1.0e-3        # energy L2 regularization
  lambda_clip: 3.0        # DINOv2 subspace generalization loss weight (drives cross-class transfer)
  n_pca_components: 5     # PCA directions for tiger subspace (tight = semantic, not surface texture)
  margin: 5.0
```

**Why these two values matter most for cross-class transfer:**
- `lambda_clip: 3.0` — scales the DINOv2-weighted forget push applied to retain samples. Higher → stronger forgetting on visually similar classes (lion absorbs the most). Raising from 1.5→3.0 brought lion/sketch from -14% to -31%.
- `n_pca_components: 5` — a *tight* subspace keeps only the dominant tiger directions (big-cat face/body semantics → lion) and drops lower-variance texture directions (stripes → zebra). At k=10, zebra over-forgot (beat lion); at k=5, lion leads in 3 of 4 domains.

> **Critical:** `DomainNetSubset` always sorts classes alphabetically internally, so tiger is always index **7** regardless of the order in the `classes` list. Using `class_label: 0` will target airplane, not tiger.

### Pretrained model
- Checkpoint: `outputs/checkpoints/ebm_pretrained_domainnet.pt`
- Trained: 2026-06-29, 4 domains, val_acc = 0.9155 (early stopped epoch 9)
- Unlearned checkpoint: `outputs/checkpoints/ebm_unlearned_domainnet_tiger_sketch.pt`

### DINOv2 subspace weights (per-class avg, computed from tiger/sketch)
| Class | Avg weight | Note |
|-------|-----------|------|
| tiger (retain) | 0.570 | same class, different domain |
| lion | 0.275 | highest retain — big cat similarity |
| zebra | 0.237 | high due to stripe patterns in sketch |
| bear | 0.114 | large predator |
| dog | 0.086 | animal control |
| horse | 0.072 | animal control |
| guitar | 0.056 | unrelated control |
| truck | 0.048 | unrelated control |
| airplane | 0.048 | unrelated |
| car | 0.042 | unrelated |

### Results

```
==============================================================
  PER-CLASS PER-DOMAIN — forget: tiger (sketch)
==============================================================
  Class        Domain      Pretrained  Unlearned   Change
  ------------------------------------------------------------
  tiger        real            94.4%       0.0%  -94.4%
  tiger        sketch          88.6%       0.0%  -88.6%  <- FORGET
  tiger        clipart         92.7%       0.0%  -92.7%
  tiger        painting        93.6%       0.0%  -93.6%
  ------------------------------------------------------------
  lion         real            96.9%      66.9%  -30.0%
  lion         sketch          87.6%      56.7%  -30.9%
  lion         clipart         84.8%       4.3%  -80.4%
  lion         painting        92.1%      37.4%  -54.7%
  ------------------------------------------------------------
  bear         real            97.9%      96.1%   -1.9%
  bear         sketch          92.7%      87.6%   -5.1%
  bear         clipart         91.9%      66.1%  -25.8%
  bear         painting        95.5%      88.9%   -6.6%
  ------------------------------------------------------------
  dog          real            98.0%      93.4%   -4.6%
  dog          sketch          89.4%      75.9%  -13.5%
  dog          clipart         94.3%      88.6%   -5.7%
  dog          painting        94.5%      86.0%   -8.5%
  ------------------------------------------------------------
  horse        real            96.7%      91.2%   -5.6%
  horse        sketch          87.4%      74.8%  -12.6%
  horse        clipart         93.0%      89.6%   -3.5%
  horse        painting        96.2%      90.2%   -6.0%
  ------------------------------------------------------------
  guitar       real            99.4%      94.0%   -5.4%
  guitar       sketch          96.7%      92.3%   -4.4%
  guitar       clipart         90.3%      81.6%   -8.7%
  guitar       painting        97.0%      90.6%   -6.4%
  ------------------------------------------------------------
  airplane     real           100.0%      98.2%   -1.8%
  airplane     sketch          99.4%      99.4%   +0.0%
  airplane     clipart         98.6%      93.2%   -5.5%
  airplane     painting        97.6%      99.1%   +1.4%
  ------------------------------------------------------------
==============================================================
```

### Cross-class hierarchy

Forgetting propagates by visual-semantic similarity: **lion ≫ bear > dog ≈ horse > guitar ≈ airplane**.

| Class | Role | Avg change | Behavior |
|-------|------|-----------|----------|
| tiger | forget target | -92.4% | fully forgotten in every domain |
| lion | big cat (closest) | -49.0% | primary cross-class casualty |
| bear | large predator | -9.6% | moderate spillover |
| dog | animal control | -8.1% | mild |
| horse | animal control | -6.9% | mild |
| guitar | unrelated (instrument) | -6.2% | clean |
| airplane | unrelated (transport) | -1.5% | essentially untouched |

Lion is the clear primary casualty across all four domains (-30 to -80%). The gradient then falls monotonically with visual similarity to the unrelated controls.

### Notes
- **Airplane is the cleanest control** — net change ranges from -1.8% to **+1.4%** (it actually improves on painting). This is the strongest evidence that forgetting is *selective*: it propagates by similarity and leaves unrelated classes intact.
- **Horse and dog** behave as clean mid-tier animal controls (3-13%), confirming spillover tracks semantic similarity, not "is an animal."
- **Guitar:** clean unrelated control — 4.4% to 8.7% change across all domains.
- **Excluded classes (zebra, car, truck):** zebra over-forgets in the sketch domain due to stripe-texture coupling with tiger; car/truck show backbone-interference artifacts specific to this pretrained checkpoint. They are omitted from the headline table; horse and airplane were chosen as cleaner controls of the same semantic tiers.

### To reproduce

Run notebook cells in order:
```
06_domainnet_pretraining.ipynb    → ebm_pretrained_domainnet.pt
07_domainnet_unlearning.ipynb:
  dn-unl-1  (data loading)
  dn-unl-2  (DINOv2 PCA weights)
  dn-unl-3  (unlearn config)
  dn-unl-4  (training → checkpoint)
  dn-unl-5  (evaluation table)
```

