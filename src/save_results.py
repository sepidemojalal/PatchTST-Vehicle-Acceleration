# =============================================================================
# src/save_results.py
# Save all experiment results to Excel and CSV.
#
# predictions_all_vehicles.xlsx
#     One sheet per vehicle (Vehicle_1 … Vehicle_30).
#     Columns: block_index, ground_truth, pred_user_specific,
#              pred_cdt_full, pred_cdt_limited,
#              error_us, error_cdt, error_ltd,
#              sq_error_us, sq_error_cdt, sq_error_ltd
#     (sq_error = squared error per block; MSE = mean of this column)
#
# metrics_summary.xlsx
#     Single sheet "MSE_Summary" with one row per vehicle and a MEAN row.
#     Columns: vehicle_id, best_model, ranking,
#              US_MSE_mean, US_MSE_CI_low, US_MSE_CI_high,
#              CDT_MSE_mean, CDT_MSE_CI_low, CDT_MSE_CI_high,
#              LTD_MSE_mean, LTD_MSE_CI_low, LTD_MSE_CI_high,
#              US_CDT_CI_overlap
#
# patchtst_results.csv
#     Same columns as metrics_summary.xlsx but in CSV for downstream analysis.
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

import config


def _autosize(ws):
    """Auto-size all columns in an openpyxl worksheet."""
    for col in ws.columns:
        width = max(len(str(cell.value or "")) for cell in col) + 3
        ws.column_dimensions[col[0].column_letter].width = width


# ─────────────────────────────────────────────────────────────────────────────
# 30-sheet predictions Excel
# ─────────────────────────────────────────────────────────────────────────────

def save_predictions_excel(all_preds: dict) -> None:
    """
    Write one Excel file with one sheet per vehicle.

    Column descriptions
    -------------------
    block_index      : 0-based test block index  (0 … 91)
    ground_truth     : actual acceleration at the target timestep  (m/s²)
    pred_*           : model prediction  (m/s²)
    error_*          : signed prediction error  (pred − truth)
    sq_error_*       : squared error per block  (mean = MSE for that model)

    Parameters
    ----------
    all_preds : {vehicle_id: {"y_true", "y_us", "y_cdt", "y_ltd"}}
    """
    os.makedirs(os.path.dirname(config.PREDICTIONS_XLSX) or ".", exist_ok=True)

    with pd.ExcelWriter(config.PREDICTIONS_XLSX, engine="openpyxl") as writer:
        for vid in sorted(all_preds.keys()):
            d      = all_preds[vid]
            y_true = d["y_true"]
            y_us   = d["y_us"]
            y_cdt  = d["y_cdt"]
            y_ltd  = d["y_ltd"]

            df = pd.DataFrame({
                "block_index"         : np.arange(len(y_true)),
                "ground_truth"        : np.round(y_true, 6),
                "pred_user_specific"  : np.round(y_us,   6),
                "pred_cdt_full"       : np.round(y_cdt,  6),
                "pred_cdt_limited"    : np.round(y_ltd,  6),
                "error_us"            : np.round(y_us  - y_true, 6),
                "error_cdt"           : np.round(y_cdt - y_true, 6),
                "error_ltd"           : np.round(y_ltd - y_true, 6),
                "sq_error_us"         : np.round((y_us  - y_true)**2, 8),
                "sq_error_cdt"        : np.round((y_cdt - y_true)**2, 8),
                "sq_error_ltd"        : np.round((y_ltd - y_true)**2, 8),
            })

            sheet = f"Vehicle_{vid}"
            df.to_excel(writer, sheet_name=sheet, index=False)
            _autosize(writer.sheets[sheet])

    print(f"  Predictions  →  {config.PREDICTIONS_XLSX}  "
          f"({len(all_preds)} sheets)")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics summary Excel  (MSE + bootstrap CI only)
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics_excel(results_df: pd.DataFrame) -> None:
    """
    Write the MSE summary Excel with one row per vehicle + a MEAN row.

    Parameters
    ----------
    results_df : DataFrame returned by main.run_experiment()
    """
    rows = []
    for _, r in results_df.iterrows():
        rows.append({
            "vehicle_id"        : int(r["vehicle_id"]),
            "train_blocks"      : int(r["train_blocks"]),
            "test_blocks"       : int(r["test_blocks"]),
            "best_model"        : r["best_model"],
            "ranking"           : r["ranking"],
            # User-Specific
            "US_MSE_mean"       : round(r["mse_us_mean"],    6),
            "US_MSE_CI_low"     : round(r["mse_us_ci_low"],  6),
            "US_MSE_CI_high"    : round(r["mse_us_ci_high"], 6),
            # CDT Full
            "CDT_MSE_mean"      : round(r["mse_cdt_mean"],   6),
            "CDT_MSE_CI_low"    : round(r["mse_cdt_ci_low"], 6),
            "CDT_MSE_CI_high"   : round(r["mse_cdt_ci_high"],6),
            # CDT Limited
            "LTD_MSE_mean"      : round(r["mse_ltd_mean"],   6),
            "LTD_MSE_CI_low"    : round(r["mse_ltd_ci_low"], 6),
            "LTD_MSE_CI_high"   : round(r["mse_ltd_ci_high"],6),
            # Significance
            "US_CDT_CI_overlap" : r["us_cdt_ci_overlap"],
        })

    df = pd.DataFrame(rows)

    num_cols = [
        "US_MSE_mean",  "US_MSE_CI_low",  "US_MSE_CI_high",
        "CDT_MSE_mean", "CDT_MSE_CI_low", "CDT_MSE_CI_high",
        "LTD_MSE_mean", "LTD_MSE_CI_low", "LTD_MSE_CI_high",
    ]
    summary = {c: round(float(df[c].mean()), 6) for c in num_cols}
    summary.update({
        "vehicle_id": "MEAN", "train_blocks": "", "test_blocks": "",
        "best_model": "", "ranking": "", "US_CDT_CI_overlap": "",
    })
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

    os.makedirs(os.path.dirname(config.METRICS_XLSX) or ".", exist_ok=True)
    with pd.ExcelWriter(config.METRICS_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="MSE_Summary", index=False)
        _autosize(writer.sheets["MSE_Summary"])

    print(f"  Metrics      →  {config.METRICS_XLSX}")


# ─────────────────────────────────────────────────────────────────────────────
# Raw CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_results_csv(results_df: pd.DataFrame) -> None:
    """Save the per-vehicle results DataFrame to CSV."""
    os.makedirs(os.path.dirname(config.RESULTS_CSV) or ".", exist_ok=True)
    results_df.to_csv(config.RESULTS_CSV, index=False)
    print(f"  CSV          →  {config.RESULTS_CSV}")
