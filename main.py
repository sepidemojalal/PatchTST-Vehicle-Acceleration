# =============================================================================
# main.py
# Entry point for PatchTST one-step vehicle acceleration prediction.
# PhD dissertation Stage 1 — Sepide Mojalal, Rowan University, 2026.
#
# Experiment overview
# ───────────────────
# Dataset : Enhanced NGSIM (US-101 + I-80)
#           30 vehicles · 350 obs each · 0.1 s resolution
#
# Task    : Given 10 consecutive timesteps of car-following features
#           [velocity, Δx, Δv, acceleration], predict the follower
#           vehicle's acceleration at the very next timestep (t+1).
#
# Models  :
#   1. User-Specific PatchTST — trained on that vehicle's 248 rows only.
#      Embodies the privacy-preserving local-learning paradigm.
#   2. CDT with Full Data     — trained on all 30 × 248 = 7,440 rows pooled.
#      Represents the centralised upper-bound baseline.
#   3. CDT with Limited Data  — trained on ~270 rows (~9/vehicle, seed=12).
#      Enables a fair matched-data comparison against User-Specific.
#
# Evaluation:
#   Bootstrap MSE (50 trials × 20 samples without replacement) per vehicle.
#   Student-t 95 % CI computed from trial MSEs.
#   CI non-overlap ≡ statistically significant model difference (ANOVA H₁).
#
# Outputs (all written to outputs/):
#   checkpoints/vehicle_N.pt          best weights, User-Specific model
#   checkpoints/cdt_full.pt           best weights, CDT Full model
#   checkpoints/cdt_limited.pt        best weights, CDT Limited model
#   plots/vehicle_N.png               2-panel figure per vehicle
#   predictions_all_vehicles.xlsx     30-sheet Excel (one sheet/vehicle)
#   metrics_summary.xlsx              MSE + CI summary + MEAN row
#   patchtst_results.csv              raw per-vehicle results
#
# Usage
# ─────
#   python main.py                               # synthetic NGSIM data
#   python main.py --data data/ngsim.csv         # real NGSIM data
#   python main.py --vehicles 5 --epochs 20      # quick test
# =============================================================================

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch

import config
from src.data_utils   import (generate_synthetic_ngsim,
                               AccelerationDataset,
                               make_loader,
                               build_cdt_datasets)
from src.model        import PatchTSTRegressor
from src.train        import train_model
from src.evaluate     import predict, bootstrap_mse, ci_overlap
from src.plot         import plot_vehicle
from src.save_results import (save_predictions_excel,
                               save_metrics_excel,
                               save_results_csv)


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.SEED)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PatchTST one-step vehicle acceleration prediction "
                    "(Sepide Mojalal, Rowan University)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data", type=str, default=None,
        help="Path to NGSIM CSV.  Required columns: vehicle_id, vehicle_index, "
             "velocity, delta_x, delta_v, acceleration.  "
             "If omitted, synthetic NGSIM-like data is generated.",
    )
    p.add_argument(
        "--vehicles", type=int, default=config.N_VEHICLES,
        help="Number of vehicles to process.",
    )
    p.add_argument(
        "--epochs", type=int, default=config.EPOCHS,
        help="Maximum training epochs per model (early stopping usually triggers "
             "before this limit).",
    )
    p.add_argument(
        "--batch", type=int, default=config.BATCH_SIZE,
        help="Mini-batch size.",
    )
    p.add_argument(
        "--trials", type=int, default=config.N_BOOTSTRAP_TRIALS,
        help="Bootstrap trials for MSE confidence interval.",
    )
    p.add_argument(
        "--d_model", type=int, default=config.D_MODEL,
        help="Transformer embedding dimension (must be divisible by --heads).",
    )
    p.add_argument(
        "--heads", type=int, default=config.NHEAD,
        help="Number of attention heads.",
    )
    p.add_argument(
        "--layers", type=int, default=config.NUM_LAYERS,
        help="Number of Transformer encoder layers.",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Experiment
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment(
        df          : pd.DataFrame,
        vehicle_ids : list,
        epochs      : int,
        batch_size  : int,
        n_trials    : int,
        model_kwargs: dict,
) -> tuple:
    """
    Full three-model experiment across all vehicles.

    Returns
    -------
    results_df : pd.DataFrame  — one row per vehicle with all MSE metrics
    all_preds  : dict          — {vid: {y_true, y_us, y_cdt, y_ltd}}
    """
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR,       exist_ok=True)

    # ── 1. Build CDT datasets ─────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Building CDT datasets …")
    print("=" * 62)

    cdt_full_df, cdt_limited_df = build_cdt_datasets(df, vehicle_ids)

    cdt_full_ds    = AccelerationDataset(cdt_full_df)
    cdt_limited_ds = AccelerationDataset(cdt_limited_df)

    print(f"  CDT Full Data    : {len(cdt_full_df):6d} rows → "
          f"{len(cdt_full_ds):5d} training blocks")
    if len(cdt_limited_ds) > 0:
        print(f"  CDT Limited Data : {len(cdt_limited_df):6d} rows → "
              f"{len(cdt_limited_ds):5d} training blocks")
    else:
        print(f"  CDT Limited Data : {len(cdt_limited_df):6d} rows → "
              "too small for sliding window (random init weights will be used)")

    # ── 2. Train CDT Full ─────────────────────────────────────────────────────
    print("\n  Training CDT with Full Data …")
    cdt_full_mdl  = PatchTSTRegressor(**model_kwargs)
    cdt_full_ckpt = os.path.join(config.CHECKPOINTS_DIR, "cdt_full.pt")
    train_model(
        cdt_full_mdl,
        make_loader(cdt_full_ds, batch_size, shuffle=True),
        make_loader(cdt_full_ds, batch_size, shuffle=False),
        checkpoint = cdt_full_ckpt,
        epochs     = epochs,
        verbose    = True,
    )
    print(f"  Checkpoint saved → {cdt_full_ckpt}")

    # ── 3. Train CDT Limited ──────────────────────────────────────────────────
    print("\n  Training CDT with Limited Data …")
    cdt_lim_mdl  = PatchTSTRegressor(**model_kwargs)
    cdt_lim_ckpt = os.path.join(config.CHECKPOINTS_DIR, "cdt_limited.pt")
    if len(cdt_limited_ds) > 0:
        train_model(
            cdt_lim_mdl,
            make_loader(cdt_limited_ds, batch_size, shuffle=True),
            make_loader(cdt_limited_ds, batch_size, shuffle=False),
            checkpoint = cdt_lim_ckpt,
            epochs     = epochs,
            verbose    = True,
        )
        print(f"  Checkpoint saved → {cdt_lim_ckpt}")
    else:
        torch.save(cdt_lim_mdl.state_dict(), cdt_lim_ckpt)
        print("  WARNING: dataset too small — random-init weights saved to "
              f"{cdt_lim_ckpt}")

    # ── 4. Per-vehicle User-Specific loop ─────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Per-vehicle User-Specific PatchTST …")
    print("=" * 62)

    results   : list = []
    all_preds : dict = {}

    for v_idx, vid in enumerate(vehicle_ids, 1):
        print(f"\n  ── Vehicle {vid:3d}  ({v_idx}/{len(vehicle_ids)}) ──")

        vdf      = df[df["vehicle_id"] == vid].reset_index(drop=True)
        train_df = vdf.iloc[: config.TRAIN_OBS]
        test_df  = vdf.iloc[config.TRAIN_OBS : config.TRAIN_OBS + config.TEST_OBS]

        train_ds = AccelerationDataset(train_df)
        test_ds  = AccelerationDataset(test_df)

        if len(train_ds) == 0 or len(test_ds) == 0:
            print("    Skipping — insufficient data.")
            continue

        print(f"    Train : {len(train_df)} rows → {len(train_ds)} blocks")
        print(f"    Test  : {len(test_df)} rows → {len(test_ds)} blocks")

        # Train User-Specific model
        us_mdl  = PatchTSTRegressor(**model_kwargs)
        us_ckpt = os.path.join(config.CHECKPOINTS_DIR, f"vehicle_{vid}.pt")
        t_losses, v_losses = train_model(
            us_mdl,
            make_loader(train_ds, batch_size, shuffle=True),
            make_loader(test_ds,  batch_size, shuffle=False),
            checkpoint = us_ckpt,
            epochs     = epochs,
            verbose    = False,
        )
        print(f"    Checkpoint   → {us_ckpt}")

        # Predictions on test set (all three models share the same test blocks)
        y_true = test_ds.y.numpy()
        y_us   = predict(us_mdl,      test_ds)
        y_cdt  = predict(cdt_full_mdl, test_ds)
        y_ltd  = predict(cdt_lim_mdl,  test_ds)

        all_preds[vid] = {
            "y_true": y_true,
            "y_us"  : y_us,
            "y_cdt" : y_cdt,
            "y_ltd" : y_ltd,
        }

        # Bootstrap MSE + 95 % CI (50 trials × 20 samples without replacement)
        mse_us  = bootstrap_mse(y_true, y_us,  n_trials=n_trials)
        mse_cdt = bootstrap_mse(y_true, y_cdt, n_trials=n_trials)
        mse_ltd = bootstrap_mse(y_true, y_ltd, n_trials=n_trials)

        # Ranking (ascending MSE = best first)
        ranked = sorted(
            [("User-Specific", mse_us["mean"]),
             ("CDT Full",      mse_cdt["mean"]),
             ("CDT Limited",   mse_ltd["mean"])],
            key=lambda x: x[1],
        )
        ranking_str    = " < ".join(m[0] for m in ranked)
        us_cdt_overlap = ci_overlap(mse_us, mse_cdt)

        print(f"    Ranking      : {ranking_str}")
        print(f"    US   MSE = {mse_us['mean']:.6f}  "
              f"95%CI [{mse_us['ci_low']:.6f}, {mse_us['ci_high']:.6f}]")
        print(f"    CDT  MSE = {mse_cdt['mean']:.6f}  "
              f"95%CI [{mse_cdt['ci_low']:.6f}, {mse_cdt['ci_high']:.6f}]")
        print(f"    LTD  MSE = {mse_ltd['mean']:.6f}  "
              f"95%CI [{mse_ltd['ci_low']:.6f}, {mse_ltd['ci_high']:.6f}]")
        print(f"    CI overlap (US vs CDT): {us_cdt_overlap}  "
              f"({'null H₀ supported' if us_cdt_overlap else 'alt H₁ supported — significant diff'})")

        # Per-vehicle 2-panel plot
        plot_path = plot_vehicle(
            train_losses = t_losses,
            val_losses   = v_losses,
            y_true       = y_true,
            y_us         = y_us,
            y_cdt        = y_cdt,
            y_ltd        = y_ltd,
            vehicle_id   = vid,
            mse_us       = mse_us["mean"],
            mse_cdt      = mse_cdt["mean"],
            mse_ltd      = mse_ltd["mean"],
        )
        print(f"    Plot         → {plot_path}")

        results.append({
            "vehicle_id"        : vid,
            "train_blocks"      : len(train_ds),
            "test_blocks"       : len(test_ds),
            "best_model"        : ranked[0][0],
            "ranking"           : ranking_str,
            "us_cdt_ci_overlap" : us_cdt_overlap,
            "mse_us_mean"       : mse_us["mean"],
            "mse_us_ci_low"     : mse_us["ci_low"],
            "mse_us_ci_high"    : mse_us["ci_high"],
            "mse_cdt_mean"      : mse_cdt["mean"],
            "mse_cdt_ci_low"    : mse_cdt["ci_low"],
            "mse_cdt_ci_high"   : mse_cdt["ci_high"],
            "mse_ltd_mean"      : mse_ltd["mean"],
            "mse_ltd_ci_low"    : mse_ltd["ci_low"],
            "mse_ltd_ci_high"   : mse_ltd["ci_high"],
        })

    return pd.DataFrame(results), all_preds


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(results_df: pd.DataFrame) -> None:
    print("\n" + "=" * 62)
    print("  RESULTS SUMMARY")
    print("=" * 62)

    n = len(results_df)
    if n == 0:
        print("  No results to show.")
        return

    first = results_df.iloc[0]
    print(f"\n  Vehicles evaluated      : {n}")
    print(f"  Input steps             : {config.INPUT_STEPS}  (1.0 s lookback)")
    print(f"  Prediction horizon      : 1 step  (next timestep, 0.1 s)")
    print(f"  Train blocks/vehicle    : {int(first['train_blocks'])}")
    print(f"  Test  blocks/vehicle    : {int(first['test_blocks'])}")
    print(f"  Bootstrap trials        : {config.N_BOOTSTRAP_TRIALS} "
          f"× {config.BOOTSTRAP_SAMPLE} samples")

    print(f"\n  Mean MSE ± std across {n} vehicles:")
    for label, col in [
        ("User-Specific PatchTST", "mse_us_mean"),
        ("CDT with Full Data     ", "mse_cdt_mean"),
        ("CDT with Limited Data  ", "mse_ltd_mean"),
    ]:
        vals = results_df[col]
        print(f"    {label} : {vals.mean():.6f}  (std {vals.std():.6f})")

    print(f"\n  Best model per vehicle:")
    print(results_df["best_model"].value_counts().to_string())

    olap = int(results_df["us_cdt_ci_overlap"].sum())
    print(f"\n  CI overlap (US vs CDT)     : {olap}/{n}  "
          f"(null H₀ — no significant difference)")
    print(f"  US statistically distinct  : {n - olap}/{n}  "
          f"(alt H₁ — significant difference)")

    print(f"\n{'─' * 62}")
    print(f"  {'VID':>4}  {'US MSE':>10}  {'CDT MSE':>10}  "
          f"{'LTD MSE':>10}  {'Best':>14}  {'CI overlap':>10}")
    print(f"{'─' * 62}")
    for _, r in results_df.iterrows():
        print(f"  {int(r['vehicle_id']):>4}  "
              f"{r['mse_us_mean']:>10.6f}  "
              f"{r['mse_cdt_mean']:>10.6f}  "
              f"{r['mse_ltd_mean']:>10.6f}  "
              f"{r['best_model']:>14}  "
              f"{'Yes' if r['us_cdt_ci_overlap'] else 'No':>10}")
    print(f"{'─' * 62}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.data and os.path.exists(args.data):
        print(f"\n  Loading NGSIM data from: {args.data}")
        df = pd.read_csv(args.data)
        required = {
            "vehicle_id", "vehicle_index",
            "velocity", "delta_x", "delta_v", "acceleration",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}\n"
                f"Expected columns: {sorted(required)}"
            )
        print(f"  Loaded {len(df):,} rows · "
              f"{df['vehicle_id'].nunique()} unique vehicles")
    else:
        if args.data:
            print(f"\n  WARNING: --data file not found: {args.data}")
        print("  Generating synthetic NGSIM-like data …")
        df = generate_synthetic_ngsim(
            n_vehicles = args.vehicles,
            n_obs      = config.N_OBS,
            seed       = config.SEED,
        )
        print(f"  Generated {len(df):,} rows · "
              f"{df['vehicle_id'].nunique()} vehicles")

    vehicle_ids  = sorted(df["vehicle_id"].unique())[: args.vehicles]
    model_kwargs = dict(
        d_model    = args.d_model,
        nhead      = args.heads,
        num_layers = args.layers,
        ffn_dim    = args.d_model * 2,
        dropout    = config.DROPOUT,
    )

    # ── Banner ────────────────────────────────────────────────────────────────
    print(f"\n{'═' * 62}")
    print(f"  PatchTST — One-Step Vehicle Acceleration Prediction")
    print(f"  Sepide Mojalal · Rowan University · Stage 1")
    print(f"{'═' * 62}")
    print(f"  Device      : {config.DEVICE}")
    print(f"  Vehicles    : {len(vehicle_ids)}")
    print(f"  Input       : {config.INPUT_STEPS} steps  (t … t+9)  →  accel at t+10")
    print(f"  Train rows  : {config.TRAIN_OBS}/vehicle  → "
          f"{config.TRAIN_OBS - config.INPUT_STEPS} blocks")
    print(f"  Test rows   : {config.TEST_OBS}/vehicle   → "
          f"{config.TEST_OBS - config.INPUT_STEPS} blocks")
    print(f"  Epochs      : {args.epochs}  (early stop patience={config.EARLY_STOP_PATIENCE})")
    print(f"  Batch size  : {args.batch}")
    print(f"  d_model     : {args.d_model}  |  heads: {args.heads}  "
          f"|  layers: {args.layers}")
    print(f"  Bootstrap   : {args.trials} trials × {config.BOOTSTRAP_SAMPLE} "
          "samples  (without replacement)")
    print(f"\n  Output directory: {config.OUTPUT_DIR}/")

    # ── Run ───────────────────────────────────────────────────────────────────
    results_df, all_preds = run_experiment(
        df          = df,
        vehicle_ids = vehicle_ids,
        epochs      = args.epochs,
        batch_size  = args.batch,
        n_trials    = args.trials,
        model_kwargs= model_kwargs,
    )

    # ── Summary & save ────────────────────────────────────────────────────────
    print_summary(results_df)

    print(f"\n{'=' * 62}")
    print("  Saving outputs …")
    print(f"{'=' * 62}")
    save_results_csv(results_df)
    save_predictions_excel(all_preds)
    save_metrics_excel(results_df)
    print(f"\n  All outputs written to  {config.OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()
