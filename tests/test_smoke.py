# =============================================================================
# tests/test_smoke.py
# Smoke tests — verify the full pipeline runs end-to-end without errors
# on a minimal 2-vehicle dataset.  No GPU or real data required.
#
# Run with:  pytest tests/ -v
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import pytest

import config
from src.data_utils import (generate_synthetic_ngsim,
                             AccelerationDataset,
                             make_loader,
                             build_cdt_datasets)
from src.model    import PatchTSTRegressor
from src.evaluate import predict, bootstrap_mse, ci_overlap


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_df():
    """2 vehicles, 80 observations each — just enough for sliding window."""
    return generate_synthetic_ngsim(n_vehicles=2, n_obs=80, seed=0)


@pytest.fixture(scope="module")
def vehicle_ds(small_df):
    vdf = small_df[small_df["vehicle_id"] == 1].reset_index(drop=True)
    return AccelerationDataset(vdf)


@pytest.fixture(scope="module")
def tiny_model():
    """Minimal model for fast testing (no GPU needed)."""
    return PatchTSTRegressor(d_model=16, nhead=2, num_layers=1, ffn_dim=32)


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def test_synthetic_columns(small_df):
    """All required columns must be present."""
    required = set(config.FEATURES + ["vehicle_id", "vehicle_index"])
    assert required.issubset(set(small_df.columns))


def test_synthetic_vehicle_count(small_df):
    assert small_df["vehicle_id"].nunique() == 2


def test_synthetic_obs_count(small_df):
    assert (small_df["vehicle_id"].value_counts() == 80).all()


def test_dataset_block_count(small_df):
    """Block count = n_rows - INPUT_STEPS  (no HORIZON offset for 1-step)."""
    vdf = small_df[small_df["vehicle_id"] == 1].reset_index(drop=True)
    ds  = AccelerationDataset(vdf)
    assert len(ds) == 80 - config.INPUT_STEPS


def test_dataset_input_shape(vehicle_ds):
    X, y = vehicle_ds[0]
    assert X.shape == (config.INPUT_STEPS, len(config.FEATURES))


def test_dataset_target_is_scalar(vehicle_ds):
    _, y = vehicle_ds[0]
    assert y.shape == ()


def test_dataset_target_is_next_step(small_df):
    """y[0] must equal the acceleration at row INPUT_STEPS (next timestep)."""
    vdf = small_df[small_df["vehicle_id"] == 1].reset_index(drop=True)
    ds  = AccelerationDataset(vdf)
    expected = float(vdf["acceleration"].iloc[config.INPUT_STEPS])
    assert abs(float(ds.y[0]) - expected) < 1e-5


def test_dataloader_batch_shape(vehicle_ds):
    loader = make_loader(vehicle_ds, batch_size=4, shuffle=False)
    Xb, yb = next(iter(loader))
    assert Xb.ndim == 3                       # (batch, steps, features)
    assert yb.ndim == 1                       # (batch,)
    assert Xb.shape[1] == config.INPUT_STEPS
    assert Xb.shape[2] == len(config.FEATURES)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def test_model_forward_shape(tiny_model, vehicle_ds):
    loader = make_loader(vehicle_ds, batch_size=4, shuffle=False)
    Xb, _  = next(iter(loader))
    out    = tiny_model(Xb)
    assert out.shape == (Xb.shape[0],)        # (batch,)


def test_model_output_is_scalar_per_sample(tiny_model, vehicle_ds):
    loader = make_loader(vehicle_ds, batch_size=1, shuffle=False)
    Xb, _  = next(iter(loader))
    out    = tiny_model(Xb)
    assert out.shape == (1,)


def test_model_parameter_count(tiny_model):
    assert tiny_model.count_parameters() > 0


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_shape(tiny_model, vehicle_ds):
    preds = predict(tiny_model, vehicle_ds)
    assert preds.shape == (len(vehicle_ds),)


def test_bootstrap_mse_structure():
    y_t = np.random.randn(92).astype(np.float32)
    y_p = np.random.randn(92).astype(np.float32)
    res = bootstrap_mse(y_t, y_p, n_trials=10, sample_size=10)
    assert set(res.keys()) == {"mean", "ci_low", "ci_high", "std"}
    assert res["ci_low"] <= res["mean"] <= res["ci_high"]
    assert res["std"] >= 0


def test_bootstrap_mse_perfect_prediction():
    y = np.ones(92, dtype=np.float32)
    res = bootstrap_mse(y, y, n_trials=10, sample_size=10)
    assert abs(res["mean"]) < 1e-9


def test_ci_overlap_overlapping():
    a = {"ci_low": 0.10, "ci_high": 0.50}
    b = {"ci_low": 0.30, "ci_high": 0.80}
    assert ci_overlap(a, b) is True


def test_ci_overlap_non_overlapping():
    a = {"ci_low": 0.10, "ci_high": 0.40}
    b = {"ci_low": 0.60, "ci_high": 0.90}
    assert ci_overlap(a, b) is False


def test_cdt_dataset_sizes(small_df):
    vehicle_ids    = [1, 2]
    full_df, ltd_df = build_cdt_datasets(small_df, vehicle_ids)
    # Full: 2 vehicles × TRAIN_OBS rows each  (capped at available rows)
    expected_full = min(config.TRAIN_OBS, 80) * 2
    assert len(full_df) == expected_full
    # Limited: min(LIMITED_N, TRAIN_OBS) rows × 2 vehicles
    assert len(ltd_df) == min(config.LIMITED_N, 80) * 2
