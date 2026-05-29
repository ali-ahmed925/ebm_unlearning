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


