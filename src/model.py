# =============================================================================
# src/model.py
# PatchTST-based scalar regression model for one-step acceleration prediction.
#
# Architecture
# ────────────
# Backbone : HuggingFace PatchTSTForPrediction  (Nie et al., ICLR 2023)
# Head     : nn.Linear(num_input_channels → 1)
#
# Why PatchTST over vanilla LSTM?
# ──────────────────────────────
# • Patching: groups consecutive timesteps into subseries-level tokens,
#   giving the Transformer richer semantic units than raw scalars.
#   With INPUT_STEPS=10, PATCH_LENGTH=2 → 5 patch tokens per channel.
# • Channel-independence: each of the 4 input features shares Transformer
#   weights but is processed as an independent stream — the same design
#   as the LSTM baseline, enabling a fair architectural comparison.
# • Self-attention attends across all 5 patch tokens simultaneously,
#   capturing temporal dependencies that LSTM processes sequentially.
#
# Input / Output
# ──────────────
# Input  : (batch, INPUT_STEPS=10, NUM_INPUT_CHANNELS=4)
# Output : (batch,)  predicted acceleration at t+1  (m/s²)
#
# Reference
# ─────────
# Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023).
# A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.
# International Conference on Learning Representations (ICLR 2023).
# https://arxiv.org/abs/2211.14730
# HuggingFace: https://huggingface.co/ibm-granite/granite-timeseries-patchtst
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTForPrediction

import config


class PatchTSTRegressor(nn.Module):
    """
    PatchTST backbone with a scalar regression head.

    The HuggingFace PatchTSTForPrediction outputs a tensor of shape
    (batch, prediction_length=1, num_input_channels=4).
    The regression head maps that to a (batch,) scalar representing
    the predicted follower vehicle acceleration at the next timestep.

    Parameters
    ----------
    d_model    : Transformer embedding dimension    [default: config.D_MODEL]
    nhead      : number of attention heads          [default: config.NHEAD]
    num_layers : number of encoder layers           [default: config.NUM_LAYERS]
    ffn_dim    : feed-forward hidden dimension      [default: config.FFN_DIM]
    dropout    : attention + FF dropout rate        [default: config.DROPOUT]
    """

    def __init__(
        self,
        d_model    : int   = config.D_MODEL,
        nhead      : int   = config.NHEAD,
        num_layers : int   = config.NUM_LAYERS,
        ffn_dim    : int   = config.FFN_DIM,
        dropout    : float = config.DROPOUT,
    ):
        super().__init__()

        patchtst_cfg = PatchTSTConfig(
            # Input / window
            num_input_channels  = config.NUM_INPUT_CHANNELS,  # 4
            context_length      = config.INPUT_STEPS,         # 10
            # Patching  (10 steps ÷ 2 = 5 non-overlapping tokens)
            patch_length        = config.PATCH_LENGTH,        # 2
            patch_stride        = config.PATCH_STRIDE,        # 2
            # Transformer
            d_model             = d_model,
            num_attention_heads = nhead,
            num_hidden_layers   = num_layers,
            ffn_dim             = ffn_dim,
            attention_dropout   = dropout,
            ff_dropout          = dropout,
            # Output  (single prediction step; head maps 4 → 1)
            prediction_length   = 1,
            loss                = "mse",
            distribution_output = "normal",
            scaling             = "std",
        )

        self.backbone = PatchTSTForPrediction(patchtst_cfg)
        # (batch, 1, 4) → (batch, 1) → (batch,)
        self.head = nn.Linear(config.NUM_INPUT_CHANNELS, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : FloatTensor  shape (batch, INPUT_STEPS, NUM_INPUT_CHANNELS)
                         i.e.  (batch, 10, 4)

        Returns
        -------
        FloatTensor  shape (batch,)
            Predicted follower vehicle acceleration at the next timestep.
        """
        out  = self.backbone(past_values=x)
        # out.prediction_outputs : (batch, prediction_length=1, num_channels=4)
        pred = out.prediction_outputs[:, 0, :]   # (batch, 4)
        return self.head(pred).squeeze(-1)        # (batch,)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
