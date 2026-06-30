# =============================================================================
# src/evaluate.py
# Inference and bootstrapped MSE evaluation with 95 % confidence intervals.
#
# Evaluation protocol — identical to the dissertation's Stage 1:
#   50 bootstrap trials, each drawing 20 samples WITHOUT replacement from
#   the 92-block test set.  Student-t 95 % CI is computed from the 50 MSE
#   values.  CI non-overlap between two models is used as the ANOVA-equivalent
#   test of statistical significance (alternative hypothesis).
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy import stats

import config
from src.data_utils import AccelerationDataset


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict(model: torch.nn.Module,
            dataset: AccelerationDataset) -> np.ndarray:
    """
    Run inference over all blocks in `dataset`.

    Parameters
    ----------
    model   : trained PatchTSTRegressor (moved to config.DEVICE internally)
    dataset : AccelerationDataset to evaluate

    Returns
    -------
    np.ndarray, shape (n_blocks,)
        Predicted acceleration values in m/s².
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    preds  = []
    with torch.no_grad():
        for Xb, _ in loader:
            out = model(Xb.to(config.DEVICE)).cpu().numpy()
            preds.extend(out.tolist())
    return np.array(preds, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap MSE  (matches dissertation protocol exactly)
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_mse(
        y_true     : np.ndarray,
        y_pred     : np.ndarray,
        n_trials   : int = config.N_BOOTSTRAP_TRIALS,
        sample_size: int = config.BOOTSTRAP_SAMPLE,
        seed       : int = config.SEED,
) -> dict:
    """
    Bootstrapped MSE with 95 % Student-t confidence interval.

    Protocol
    --------
    • n_trials independent draws (default 50)
    • Each draw: sample_size instances WITHOUT replacement (default 20)
    • MSE computed over the drawn subset
    • CI: Student-t interval at df = n_trials − 1

    Parameters
    ----------
    y_true      : ground-truth accelerations  (n_blocks,)
    y_pred      : predicted accelerations     (n_blocks,)
    n_trials    : number of bootstrap repetitions
    sample_size : instances sampled per trial (without replacement)
    seed        : RNG seed for reproducibility

    Returns
    -------
    dict with keys
        mean    : mean MSE across trials
        ci_low  : lower bound of 95 % CI
        ci_high : upper bound of 95 % CI
        std     : standard deviation of MSE values across trials
    """
    rng  = np.random.default_rng(seed)
    mses = []
    n    = len(y_true)

    for _ in range(n_trials):
        idx  = rng.choice(n, size=min(sample_size, n), replace=False)
        mse  = float(np.mean((y_true[idx] - y_pred[idx]) ** 2))
        mses.append(mse)

    mses = np.array(mses)
    ci   = stats.t.interval(
        0.95,
        df    = n_trials - 1,
        loc   = np.mean(mses),
        scale = stats.sem(mses),
    )
    return {
        "mean"   : float(np.mean(mses)),
        "ci_low" : float(ci[0]),
        "ci_high": float(ci[1]),
        "std"    : float(np.std(mses)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CI overlap  (ANOVA-equivalent significance check)
# ─────────────────────────────────────────────────────────────────────────────

def ci_overlap(a: dict, b: dict) -> bool:
    """
    Return True if the 95 % bootstrap CIs of two models overlap.

    Interpretation (matches dissertation)
    ──────────────────────────────────────
    • Overlap     → null hypothesis: no statistically significant difference
    • No overlap  → alternative hypothesis: one model is significantly better
    """
    return a["ci_low"] <= b["ci_high"] and b["ci_low"] <= a["ci_high"]
