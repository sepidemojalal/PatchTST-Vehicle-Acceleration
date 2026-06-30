# =============================================================================
# config.py
# Central configuration for the PatchTST one-step acceleration prediction
# experiment (Stage 1 of Sepide Mojalal's PhD dissertation, Rowan University).
#
# All hyperparameters, dataset constants, and output paths live here.
# Edit this file to change any setting — every other module imports from here.
# =============================================================================

import os
import torch

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42          # master seed for torch, numpy, random

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset (Enhanced-NGSIM) ──────────────────────────────────────────────────
# Source: U.S. DOT FHWA Enhanced NGSIM Vehicle Trajectories and Supporting Data
# Locations: southbound US-101 (Los Angeles, CA) + eastbound I-80 (Emeryville, CA)
# Resolution: 0.1 s per observation (10 Hz)

N_VEHICLES    = 30      # vehicles randomly selected from the 1,000-vehicle pool
N_OBS         = 350     # observations per vehicle
TRAIN_OBS     = 248     # training rows per vehicle  (≈ 70 %)
TEST_OBS      = 102     # test rows per vehicle      (≈ 30 %)

# Four input features matching all other models in the dissertation
# (Helly, IDM, LSTM — same feature set, same preprocessing)
FEATURES = [
    "velocity",      # follower vehicle velocity (m/s)
    "delta_x",       # positional gap between follower and leader (m)
    "delta_v",       # speed difference between follower and leader (m/s)
    "acceleration",  # current acceleration of the follower vehicle (m/s²)
]
TARGET = "acceleration"

# ── CDT limited-data configuration ───────────────────────────────────────────
# Matches the paper: 9 frames randomly selected per vehicle (seed=12),
# giving a limited aggregated dataset of ~270 rows — comparable in size to
# a single User-Specific dataset (248 rows), enabling a fair comparison.
LIMITED_N    = 9    # frames randomly sampled per vehicle
LIMITED_SEED = 12   # fixed seed for reproducibility (matches dissertation)

# ── Sliding window ────────────────────────────────────────────────────────────
# Stage 1: one-step prediction
#   Input  : 10 consecutive timesteps (t … t+9)   →  1.0 s of history
#   Target : acceleration at timestep t+10         →  immediate next step
#
# Block counts:
#   Train : 248 rows − 10 input steps = 238 blocks per vehicle
#   Test  :  102 rows − 10 input steps =  92 blocks per vehicle
#   CDT Full    : 7440 rows − 10 = 7430 blocks
#   CDT Limited :  270 rows − 10 =  260 blocks
INPUT_STEPS   = 10   # number of input timesteps (1.0 s at 0.1 s resolution)
HORIZON_STEPS = 1    # prediction horizon: 1 step = next timestep (0.1 s ahead)

# ── PatchTST architecture ────────────────────────────────────────────────────
# Nie et al. "A Time Series is Worth 64 Words" (ICLR 2023)
# HuggingFace: ibm-granite/granite-timeseries-patchtst
#
# With INPUT_STEPS=10 and PATCH_LENGTH=2:
#   number of patch tokens = ceil(10 / 2) = 5 tokens per channel
NUM_INPUT_CHANNELS  = len(FEATURES)   # 4
PATCH_LENGTH        = 2               # timesteps grouped into one token
PATCH_STRIDE        = 2               # non-overlapping patches
D_MODEL             = 64              # Transformer embedding dimension
NHEAD               = 4               # attention heads  (D_MODEL must be divisible)
NUM_LAYERS          = 2               # stacked Transformer encoder layers
FFN_DIM             = 128             # feed-forward hidden dimension  (2 × D_MODEL)
DROPOUT             = 0.1             # attention + FF dropout rate

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE          = 32
EPOCHS              = 300             # maximum; early stopping usually triggers earlier
LR                  = 1e-3            # Adam initial learning rate
LR_PATIENCE         = 5               # ReduceLROnPlateau patience (epochs)
LR_FACTOR           = 0.5             # LR reduction factor
EARLY_STOP_PATIENCE = 10              # stop if val MSE does not improve for N epochs
GRAD_CLIP           = 1.0             # gradient clipping (ℓ₂ norm)

# ── Evaluation ────────────────────────────────────────────────────────────────
# Bootstrap MSE protocol — matches dissertation Stage 1 evaluation exactly:
#   50 independent trials, each drawing 20 samples without replacement from
#   the 92 test blocks.  Student-t 95 % CI computed across the 50 MSE values.
#   CI non-overlap ≡ ANOVA alternative hypothesis (statistically significant
#   difference between models).
N_BOOTSTRAP_TRIALS = 50
BOOTSTRAP_SAMPLE   = 20   # samples per trial (without replacement)

# ── Output paths ──────────────────────────────────────────────────────────────
OUTPUT_DIR       = "outputs"
CHECKPOINTS_DIR  = os.path.join(OUTPUT_DIR, "checkpoints")
PLOTS_DIR        = os.path.join(OUTPUT_DIR, "plots")
PREDICTIONS_XLSX = os.path.join(OUTPUT_DIR, "predictions_all_vehicles.xlsx")
METRICS_XLSX     = os.path.join(OUTPUT_DIR, "metrics_summary.xlsx")
RESULTS_CSV      = os.path.join(OUTPUT_DIR, "patchtst_results.csv")
