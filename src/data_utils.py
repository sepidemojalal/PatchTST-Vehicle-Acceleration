# =============================================================================
# src/data_utils.py
# Data loading, synthetic NGSIM generator, PyTorch Dataset, and CDT builders.
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import config


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic NGSIM-like data generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_ngsim(
        n_vehicles: int = config.N_VEHICLES,
        n_obs:      int = config.N_OBS,
        seed:       int = config.SEED) -> pd.DataFrame:
    """
    Generate synthetic car-following trajectories that mimic NGSIM statistics.

    Each vehicle is assigned random IDM-inspired parameters (base velocity and
    aggression scale), creating non-IID behavioural heterogeneity across
    vehicles — the core challenge this dissertation addresses.

    This generator is used ONLY when no real NGSIM CSV is provided.
    For real experiments, supply the CSV via --data.

    Columns produced
    ----------------
    vehicle_id    : integer vehicle identifier  (1 … n_vehicles)
    vehicle_index : timestep index within vehicle  (0 … n_obs-1)
    velocity      : follower vehicle velocity  (m/s)
    delta_x       : gap between follower and leader  (m)
    delta_v       : speed difference between follower and leader  (m/s)
    acceleration  : follower vehicle acceleration  (m/s²)
    """
    rng     = np.random.default_rng(seed)
    records = []

    for vid in range(1, n_vehicles + 1):
        # Vehicle-specific parameters — creates non-IID data across vehicles
        base_vel   = rng.uniform(10, 30)    # desired free-flow speed  (m/s)
        aggression = rng.uniform(0.5, 2.0)  # acceleration sensitivity scale

        vel = base_vel
        acc = 0.0
        gap = rng.uniform(10, 40)
        dv  = 0.0

        for idx in range(n_obs):
            # IDM-inspired acceleration with small Gaussian noise
            desired_gap = 4.0 + 1.5 * vel
            acc_new = float(np.clip(
                aggression * (
                    1.0
                    - (vel / (base_vel + 5.0)) ** 4
                    - (desired_gap / max(gap, 1.0)) ** 2
                ) + rng.normal(0.0, 0.05),
                -4.0, 3.0
            ))

            records.append({
                "vehicle_id"   : vid,
                "vehicle_index": idx,
                "velocity"     : round(vel, 4),
                "delta_x"      : round(gap, 4),
                "delta_v"      : round(dv,  4),
                "acceleration" : round(acc, 4),
            })

            # Euler integration for next state
            vel = float(np.clip(vel + acc_new * 0.1, 0.0, 45.0))
            gap = float(np.clip(gap + dv  * 0.1, 2.0, 100.0))
            dv  = rng.uniform(-1.0, 1.0) * 0.3 + dv * 0.8
            acc = acc_new

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset  —  sliding-window one-step prediction
# ─────────────────────────────────────────────────────────────────────────────

class AccelerationDataset(Dataset):
    """
    One-step-ahead acceleration prediction dataset.

    Sliding window formulation (Stage 1):
        X[i] = feature matrix for rows i … i+INPUT_STEPS-1   shape (10, 4)
        y[i] = acceleration at row i+INPUT_STEPS              scalar

    Example for INPUT_STEPS=10:
        Block 0  : X = rows [0..9]  →  y = row 10  (acceleration at t11)
        Block 1  : X = rows [1..10] →  y = row 11
        …
        Block 237: X = rows [228..237] → y = row 238     (last train block)

    Block counts:
        248 train rows → 238 blocks   (248 - 10)
        102 test rows  →  92 blocks   (102 - 10)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns defined in config.FEATURES + config.TARGET.
        Rows must be ordered by timestep (ascending vehicle_index).
    """

    def __init__(self, df: pd.DataFrame):
        arr = df[config.FEATURES].values.astype(np.float32)
        acc = df[config.TARGET].values.astype(np.float32)
        n   = len(arr)

        X_list, y_list = [], []
        for i in range(n - config.INPUT_STEPS):
            X_list.append(arr[i : i + config.INPUT_STEPS])
            y_list.append(acc[i + config.INPUT_STEPS])

        if len(X_list) == 0:
            self.X = torch.zeros((0, config.INPUT_STEPS, len(config.FEATURES)))
            self.y = torch.zeros((0,))
        else:
            self.X = torch.tensor(np.array(X_list), dtype=torch.float32)
            self.y = torch.tensor(np.array(y_list), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def make_loader(
        dataset:    AccelerationDataset,
        batch_size: int,
        shuffle:    bool) -> DataLoader:
    """
    Create a DataLoader, capping batch_size to dataset length.
    Raises ValueError if dataset is empty.
    """
    if len(dataset) == 0:
        raise ValueError("Cannot create a DataLoader from an empty dataset.")
    bs = max(1, min(batch_size, len(dataset)))
    return DataLoader(dataset, batch_size=bs, shuffle=shuffle, drop_last=False)


# ─────────────────────────────────────────────────────────────────────────────
# CDT dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def build_cdt_datasets(
        df:          pd.DataFrame,
        vehicle_ids: list) -> tuple:
    """
    Build the two centralised (CDT) training DataFrames.

    CDT with Full Data
        Concatenates the training rows of all vehicles.
        Represents the highest-data centralised baseline.
        Size: 248 × n_vehicles rows   (7,440 rows for 30 vehicles).

    CDT with Limited Data
        Selects LIMITED_N (9) rows at random per vehicle (seed=LIMITED_SEED=12),
        producing a dataset of ~270 rows whose size matches a single vehicle's
        training set — enabling a fair matched-data comparison against
        User-Specific models.

    Parameters
    ----------
    df          : full raw DataFrame (all vehicles)
    vehicle_ids : list of vehicle IDs to include

    Returns
    -------
    cdt_full_df    : pd.DataFrame  (all training rows pooled)
    cdt_limited_df : pd.DataFrame  (~9 rows × n_vehicles)
    """
    rng_ltd            = np.random.default_rng(config.LIMITED_SEED)
    cdt_train_frames   = []
    cdt_limited_frames = []

    for vid in vehicle_ids:
        vdf   = df[df["vehicle_id"] == vid].reset_index(drop=True)
        train = vdf.iloc[: config.TRAIN_OBS]
        cdt_train_frames.append(train)

        n_sample = min(config.LIMITED_N, len(train))
        idx      = rng_ltd.choice(len(train), size=n_sample, replace=False)
        cdt_limited_frames.append(train.iloc[sorted(idx)])

    cdt_full_df    = pd.concat(cdt_train_frames,   ignore_index=True)
    cdt_limited_df = pd.concat(cdt_limited_frames, ignore_index=True)

    return cdt_full_df, cdt_limited_df
