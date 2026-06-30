# PatchTST for Vehicle Acceleration Prediction — Stage 1

---

## What this repository does

This code implements **PatchTST** (Nie et al., ICLR 2023) as the Transformer-based predictor for Stage 1 of the dissertation, extending the future work described in Section 5 of the PhD proposal.

**Task:** Given the last 10 timesteps (1.0 s) of four car-following features for a follower vehicle, predict its acceleration at the very next timestep (t+11, i.e. 0.1 s ahead).

**Three models are trained and compared for each of the 30 vehicles:**

| Model | Training data | Description |
|---|---|---|
| **User-Specific PatchTST** | 248 rows — that vehicle only | Privacy-preserving local learning baseline |
| **CDT with Full Data** | 7,440 rows — all 30 vehicles pooled | Centralised upper-bound baseline |
| **CDT with Limited Data** | ~270 rows — 9 random rows × 30 vehicles | Fair matched-data centralised comparison |

This mirrors the exact framework from the dissertation's Stage 1 study, replacing LSTM with PatchTST as the data-driven predictor while keeping all other experimental conditions identical.

---

## Dataset

**Enhanced NGSIM (Next Generation Simulation)**  
U.S. Department of Transportation, Federal Highway Administration

- **Locations:** Southbound US-101 (Los Angeles, CA) + Eastbound I-80 (Emeryville, CA)
- **Resolution:** 0.1 seconds (10 Hz)
- **Total vehicles in database:** 1,000
- **Vehicles used in this study:** 30 (randomly selected)
- **Observations per vehicle:** 350

**Download:** https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm  
See `data/README.md` for the required CSV format.

### Input features (4 channels)

| Feature | Description | Unit |
|---|---|---|
| `velocity` | Follower vehicle speed | m/s |
| `delta_x` | Positional gap between follower and leader | m |
| `delta_v` | Speed difference: follower − leader | m/s |
| `acceleration` | Current follower vehicle acceleration | m/s² |

### Dataset split

| Split | Rows | Sliding-window blocks |
|---|---|---|
| Training (per vehicle) | 248 (~70 %) | **238** |
| Test (per vehicle) | 102 (~30 %) | **92** |
| CDT Full Data (all 30 vehicles) | 7,440 | 7,430 |
| CDT Limited Data | ~270 | 260 |

### Sliding window

```
Block 0  :  X = [t1, t2, ..., t10]  →  y = t11
Block 1  :  X = [t2, t3, ..., t11]  →  y = t12
...
Block 237:  X = [t239, ..., t248]   →  y = t249
```

Each block feeds 10 × 4 features into PatchTST and predicts a single scalar acceleration value.

---

## Model: PatchTST

**Full name:** Patch Time Series Transformer  
**Paper:** Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers," ICLR 2023  
**HuggingFace:** https://huggingface.co/ibm-granite/granite-timeseries-patchtst

### Architecture

```
Input: (batch, 10 timesteps, 4 features)
    │
    ├─ Patch tokenisation  (patch_length=2, stride=2)
    │   10 steps ÷ 2 = 5 patch tokens per channel
    │
    ├─ Channel-independent embedding  (d_model=64)
    │   4 channels share weights, processed independently
    │
    ├─ Positional encoding  (sinusoidal)
    │
    ├─ Transformer encoder × 2 layers
    │   4 attention heads · FFN dim=128 · dropout=0.1
    │
    ├─ Prediction head  (prediction_length=1)
    │   Output: (batch, 1, 4)
    │
    └─ Regression head  Linear(4 → 1)
        Output: (batch,)  predicted acceleration  (m/s²)
```

**Why PatchTST over LSTM?**  
- Patching creates richer tokens than raw scalars, letting attention compare temporal segments directly  
- Self-attention captures global dependencies across the 10-step window in parallel, vs. LSTM's sequential recurrence  
- Channel-independence matches the dissertation's feature structure and enables direct comparison with the LSTM baseline  
- Academically established benchmark (ICLR 2023, widely cited)

### BibTeX

```bibtex
@inproceedings{nie2023time,
  title     = {A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
  author    = {Yuqi Nie and Nam H. Nguyen and Phanwadee Sinthong and Jayant Kalagnanam},
  booktitle = {International Conference on Learning Representations},
  year      = {2023},
  url       = {https://arxiv.org/abs/2211.14730}
}
```

---

## Evaluation protocol

1. **Bootstrap MSE:** 50 independent trials, each drawing 20 samples **without replacement** from the 92 test blocks
2. **Confidence interval:** Student-t 95 % CI computed from the 50 MSE values (`df = 49`)
3. **Statistical significance:** CI non-overlap between two models ≡ ANOVA alternative hypothesis H₁ (statistically significant difference)

---

## Repository structure

```
PatchTST-Vehicle-Acceleration/
│
├── main.py                  ← Entry point — run this
├── config.py                ← All hyperparameters and paths (edit here)
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── data_utils.py        ← Dataset class, CDT builder
│   ├── model.py             ← PatchTSTRegressor (HuggingFace backbone + head)
│   ├── train.py             ← Training loop (Adam, early stopping, checkpoint)
│   ├── evaluate.py          ← Bootstrap MSE, 95 % CI, CI overlap test
│   ├── plot.py              ← Per-vehicle 2-panel PNG
│   └── save_results.py      ← Excel (30-sheet + summary) and CSV writers
│
├── data/
│   └── README.md            ← NGSIM download instructions + required format
│
├── outputs/                 ← All generated files go here (gitignored)
│   ├── checkpoints/         ← Best model weights (.pt)
│   ├── plots/               ← Per-vehicle PNG figures
│   ├── predictions_all_vehicles.xlsx
│   ├── metrics_summary.xlsx
│   └── patchtst_results.csv
│
└── tests/
    └── test_smoke.py        ← 17 pytest smoke tests
```

---

## Installation

```bash
git clone https://github.com/<sepidemojalal>/PatchTST-Vehicle-Acceleration.git
cd PatchTST-Vehicle-Acceleration
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0, HuggingFace Transformers ≥ 4.36

---

## Usage

### Run with real NGSIM data

```bash
python main.py --data data/your_ngsim_file.csv
```

### Run with synthetic data (no CSV needed)

```bash
python main.py
```

### Quick test (5 vehicles, 20 epochs)

```bash
python main.py --vehicles 5 --epochs 20
```

### All command-line options

```
python main.py --help

  --data      Path to NGSIM CSV file (omit to use synthetic data)
  --vehicles  Number of vehicles to process          [default: 30]
  --epochs    Maximum training epochs per model      [default: 300]
  --batch     Mini-batch size                        [default: 32]
  --trials    Bootstrap trials for MSE CI            [default: 50]
  --d_model   Transformer embedding dimension        [default: 64]
  --heads     Number of attention heads              [default: 4]
  --layers    Number of Transformer encoder layers   [default: 2]
```

---

## Outputs

All outputs are written to `outputs/` automatically:

| File / Folder | Description |
|---|---|
| `outputs/checkpoints/vehicle_N.pt` | Best weights for User-Specific model, vehicle N |
| `outputs/checkpoints/cdt_full.pt` | Best weights for CDT with Full Data |
| `outputs/checkpoints/cdt_limited.pt` | Best weights for CDT with Limited Data |
| `outputs/plots/vehicle_N.png` | 2-panel PNG: loss curves + test predictions |
| `outputs/predictions_all_vehicles.xlsx` | 30 sheets · 11 columns each |
| `outputs/metrics_summary.xlsx` | MSE + 95 % CI for all models and vehicles |
| `outputs/patchtst_results.csv` | Raw per-vehicle results for downstream analysis |

### Predictions Excel columns

| Column | Description |
|---|---|
| `block_index` | Test block index (0–91) |
| `ground_truth` | Actual acceleration (m/s²) |
| `pred_user_specific` | User-Specific PatchTST prediction |
| `pred_cdt_full` | CDT with Full Data prediction |
| `pred_cdt_limited` | CDT with Limited Data prediction |
| `error_*` | Signed error: prediction − ground truth |
| `sq_error_*` | Squared error per block (mean = MSE) |

### Metrics Excel columns

| Column | Description |
|---|---|
| `vehicle_id` | Vehicle identifier (1–30), plus MEAN row |
| `best_model` | Model with lowest MSE for this vehicle |
| `ranking` | Models ranked lowest → highest MSE |
| `US_MSE_mean` | Bootstrapped mean MSE — User-Specific |
| `US_MSE_CI_low/high` | 95 % CI bounds — User-Specific |
| `CDT_MSE_mean/CI_*` | Same for CDT with Full Data |
| `LTD_MSE_mean/CI_*` | Same for CDT with Limited Data |
| `US_CDT_CI_overlap` | True = null H₀; False = alt H₁ (significant) |

---

## Key configuration values

All values live in `config.py`. The most commonly adjusted:

```python
# Sliding window (Stage 1)
INPUT_STEPS   = 10   # input timesteps (1.0 s of history)
HORIZON_STEPS = 1    # 1-step ahead prediction (next timestep)

# PatchTST
D_MODEL    = 64      # embedding dimension
NHEAD      = 4       # attention heads
NUM_LAYERS = 2       # encoder layers
DROPOUT    = 0.1

# Training
EPOCHS              = 300
LR                  = 1e-3
EARLY_STOP_PATIENCE = 10
BATCH_SIZE          = 32

# Evaluation
N_BOOTSTRAP_TRIALS = 50
BOOTSTRAP_SAMPLE   = 20

# CDT limited dataset
LIMITED_N    = 9
LIMITED_SEED = 12
```

---

## Running tests

```bash
pytest tests/ -v
```

17 smoke tests cover: synthetic data generation, dataset block counts, input/output shapes, one-step target alignment, model forward pass, parameter counting, prediction shapes, bootstrap MSE structure, perfect-prediction MSE = 0, CI overlap logic, and CDT dataset sizes.

---

