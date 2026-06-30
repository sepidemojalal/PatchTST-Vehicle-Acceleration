# =============================================================================
# src/plot.py
# Per-vehicle 2-panel PNG figure.
#
# Panel 1 — User-Specific PatchTST training & validation loss curves
# Panel 2 — Ground truth vs predictions from all three models on the test set
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe on HPC / headless)
import matplotlib.pyplot as plt

import config


def plot_vehicle(
        train_losses : list,
        val_losses   : list,
        y_true       : np.ndarray,
        y_us         : np.ndarray,
        y_cdt        : np.ndarray,
        y_ltd        : np.ndarray,
        vehicle_id   : int,
        mse_us       : float,
        mse_cdt      : float,
        mse_ltd      : float,
) -> str:
    """
    Save a 2-panel figure to outputs/plots/vehicle_{vehicle_id}.png.

    Panel 1 — Loss curves
        Training and validation MSE per epoch for the User-Specific model.
        Shows convergence behaviour and whether early stopping was triggered.

    Panel 2 — Predictions vs ground truth
        Test-set acceleration (m/s²) for all 92 test blocks.
        Ground truth (black), User-Specific (blue), CDT Full (orange),
        CDT Limited (green).  MSE shown in legend for quick comparison.

    Parameters
    ----------
    train_losses : per-epoch training MSE  (User-Specific model)
    val_losses   : per-epoch validation MSE  (User-Specific model)
    y_true       : ground-truth accelerations on the 92 test blocks
    y_us         : User-Specific PatchTST predictions
    y_cdt        : CDT with Full Data predictions
    y_ltd        : CDT with Limited Data predictions
    vehicle_id   : integer ID used in title and filename
    mse_us/cdt/ltd : bootstrapped mean MSE for each model

    Returns
    -------
    str : path to the saved PNG file
    """
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    fig.suptitle(
        f"Vehicle {vehicle_id}  ·  PatchTST (Nie et al., ICLR 2023)  ·  "
        f"One-Step Acceleration Prediction\n"
        f"Input: [velocity, Δx, Δv, accel] × {config.INPUT_STEPS} steps"
        f"  →  accel at t+{config.INPUT_STEPS}  (next timestep, 0.1 s ahead)",
        fontsize=10,
        fontweight="bold",
    )

    # ── Panel 1: Training & validation loss ───────────────────────────────────
    ax = axes[0]
    ax.plot(train_losses, lw=1.6, color="#2077B4",
            label="Train MSE  (User-Specific)")
    ax.plot(val_losses,   lw=1.6, color="#D62728",
            label="Val MSE    (User-Specific)", linestyle="--")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("MSE  (m/s²)²", fontsize=10)
    ax.set_title("User-Specific PatchTST — Training & Validation Loss",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Panel 2: Predictions vs ground truth ──────────────────────────────────
    ax  = axes[1]
    idx = np.arange(len(y_true))
    ax.plot(idx, y_true, lw=2.0,  color="black",     label="Ground truth")
    ax.plot(idx, y_us,   lw=1.5,  color="#2077B4",
            label=f"User-Specific PatchTST  (MSE = {mse_us:.4f})")
    ax.plot(idx, y_cdt,  lw=1.5,  color="#FF7F0E",
            label=f"CDT with Full Data       (MSE = {mse_cdt:.4f})",
            linestyle="--")
    ax.plot(idx, y_ltd,  lw=1.5,  color="#2CA02C",
            label=f"CDT with Limited Data    (MSE = {mse_ltd:.4f})",
            linestyle=":")
    ax.set_xlabel("Test Block Index  (block 0 = rows 249–258 of vehicle data)",
                  fontsize=10)
    ax.set_ylabel("Acceleration  (m/s²)", fontsize=10)
    ax.set_title(
        f"Test-Set Predictions  —  92 blocks  "
        f"(rows 249–350 of vehicle {vehicle_id})",
        fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, f"vehicle_{vehicle_id}.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
