# =============================================================================
# src/train.py
# Training loop with Adam, ReduceLROnPlateau, gradient clipping,
# early stopping, and best-weight checkpoint saving.
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config


def train_model(
        model        : nn.Module,
        train_loader : DataLoader,
        val_loader   : DataLoader,
        checkpoint   : str,
        epochs       : int   = config.EPOCHS,
        lr           : float = config.LR,
        patience     : int   = config.EARLY_STOP_PATIENCE,
        verbose      : bool  = False,
) -> tuple:
    """
    Train a PatchTSTRegressor with MSE loss, early stopping, and checkpointing.

    Protocol
    --------
    • Loss      : MSELoss (matches dissertation evaluation metric)
    • Optimiser : Adam  (lr=1e-3, adjustable via config)
    • Scheduler : ReduceLROnPlateau (factor=0.5, patience=5)
    • Grad clip : ℓ₂ norm clipped at 1.0 to prevent exploding gradients
    • Early stop: training halts when validation MSE fails to improve for
                  `patience` consecutive epochs
    • Checkpoint: best weights (lowest val MSE) saved to `checkpoint` path
                  and reloaded before return

    Parameters
    ----------
    model        : PatchTSTRegressor (or any nn.Module)
    train_loader : DataLoader for the training split
    val_loader   : DataLoader for the validation / test split
    checkpoint   : file path (.pt) where the best weights are saved
    epochs       : maximum training epochs
    lr           : Adam initial learning rate
    patience     : early-stopping patience in epochs
    verbose      : if True, print loss every 5 epochs + early-stop message

    Returns
    -------
    train_losses : list[float]  — per-epoch training MSE
    val_losses   : list[float]  — per-epoch validation MSE
    """
    os.makedirs(os.path.dirname(checkpoint) or ".", exist_ok=True)
    model.to(config.DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode    = "min",
        factor  = config.LR_FACTOR,
        patience= config.LR_PATIENCE,
    )

    train_losses : list[float] = []
    val_losses   : list[float] = []
    best_val     = float("inf")
    wait         = 0
    best_state   = None

    for epoch in range(1, epochs + 1):

        # ── Training pass ─────────────────────────────────────────────────────
        model.train()
        running = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(config.DEVICE), yb.to(config.DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
            running += loss.item() * len(Xb)
        t_loss = running / len(train_loader.dataset)

        # ── Validation pass ───────────────────────────────────────────────────
        model.eval()
        running = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(config.DEVICE), yb.to(config.DEVICE)
                running += criterion(model(Xb), yb).item() * len(Xb)
        v_loss = running / len(val_loader.dataset)

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        scheduler.step(v_loss)

        # ── Checkpoint & early stopping ───────────────────────────────────────
        if v_loss < best_val:
            best_val   = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state, checkpoint)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"    Early stop at epoch {epoch}  "
                          f"(best val MSE = {best_val:.6f})")
                break

        if verbose and epoch % 5 == 0:
            print(f"    Epoch {epoch:4d}/{epochs}  "
                  f"train={t_loss:.6f}  val={v_loss:.6f}  "
                  f"best={best_val:.6f}")

    # Restore best weights before returning
    if best_state is not None:
        model.load_state_dict(best_state)

    return train_losses, val_losses
