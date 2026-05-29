#!/usr/bin/env python3
"""
predictive_model.py — LSTM-based Predictive Failure Detection (Step 19)
=======================================================================

Trains an LSTM neural network on historical interface-error time-series data
to predict error counts **N steps ahead** (``PREDICTION_HORIZON``).  When the
predicted error count exceeds a configurable threshold an alert is printed,
enabling proactive remediation before a failure cascade.

Architecture
------------
``LSTMPredictor``
    LSTM(input_size, hidden_size, num_layers) → Linear(hidden_size, 1)

Data pipeline
-------------
1. Load ``error_data.csv`` (timestamp + per-interface error columns).
2. Aggregate to a single ``total_errors`` series (sum across interfaces).
3. Min-max normalise, then slice into overlapping sliding windows of length
   ``LSTM_SEQ_LENGTH`` with targets ``PREDICTION_HORIZON`` steps ahead.
4. 80 / 20 chronological train / test split (no shuffle — preserving order).

Usage
-----
    # Train with defaults
    python predictive_model.py

    # Override epochs, enable plot
    python predictive_model.py --epochs 100 --plot

    # Predict on the latest window and alert if threshold exceeded
    python predictive_model.py --predict
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── project configuration ──────────────────────────────────────────────
from config import (
    LSTM_EPOCHS,
    LSTM_HIDDEN_SIZE,
    LSTM_LEARNING_RATE,
    LSTM_NUM_LAYERS,
    LSTM_SEQ_LENGTH,
    LOG_DIR,
    MODEL_DIR,
    PREDICTION_HORIZON,
)

# ── logging setup ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── alert threshold (errors / interval) for --predict mode ─────────────
ALERT_THRESHOLD = 50.0


# ╭───────────────────────────────────────────────────────────────────────╮
# │  Model                                                                │
# ╰───────────────────────────────────────────────────────────────────────╯
class LSTMPredictor(nn.Module):
    """Multi-layer LSTM followed by a fully-connected regression head.

    Parameters
    ----------
    input_size : int
        Number of features per time-step (1 for univariate error count,
        more if traffic columns are included).
    hidden_size : int
        Dimensionality of each LSTM hidden state.
    num_layers : int
        Number of stacked LSTM layers.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, seq_len, input_size)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 1)`` — predicted error count.
        """
        # h0, c0 default to zeros
        lstm_out, _ = self.lstm(x)
        # Take the output of the last time-step
        last_hidden = lstm_out[:, -1, :]
        prediction = self.fc(last_hidden)
        return prediction


# ╭───────────────────────────────────────────────────────────────────────╮
# │  Synthetic data generation                                            │
# ╰───────────────────────────────────────────────────────────────────────╯
def generate_synthetic_data(dest_path: Path) -> pd.DataFrame:
    """Create 30 days of hourly interface-error counts with realistic patterns.

    The series contains:
    * a slow upward trend (ageing / degradation),
    * diurnal seasonality (traffic-driven),
    * Gaussian noise,
    * occasional random spikes (simulated micro-outages).

    Parameters
    ----------
    dest_path : Path
        Where to write the resulting CSV.

    Returns
    -------
    pd.DataFrame
        The generated DataFrame (also saved to *dest_path*).
    """
    warnings.warn(
        "Real data not found — generating synthetic time-series for demonstration.",
        UserWarning,
        stacklevel=2,
    )
    logger.warning("Using SYNTHETIC data — results are illustrative only.")

    rng = np.random.default_rng(seed=42)
    hours = 30 * 24  # 30 days of hourly samples

    timestamps = pd.date_range(
        start="2026-01-01", periods=hours, freq="h"
    )

    # Trend component — slow linear growth
    trend = np.linspace(0, 15, hours)

    # Diurnal seasonality (peak errors during business hours)
    hour_of_day = np.array([t.hour for t in timestamps])
    seasonality = 5.0 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)

    # Noise
    noise = rng.normal(loc=0, scale=3, size=hours)

    # Random spikes (≈ 2 % of the time)
    spikes = np.zeros(hours)
    spike_mask = rng.random(hours) < 0.02
    spikes[spike_mask] = rng.uniform(30, 80, size=spike_mask.sum())

    total_errors = np.clip(trend + seasonality + noise + spikes, 0, None)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "total_errors": np.round(total_errors, 2),
    })

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest_path, index=False)
    logger.info("Synthetic data saved to %s (%d rows).", dest_path, len(df))
    return df


# ╭───────────────────────────────────────────────────────────────────────╮
# │  Data loading & preprocessing                                         │
# ╰───────────────────────────────────────────────────────────────────────╯
def load_data(data_dir: str) -> pd.DataFrame:
    """Load ``error_data.csv`` or fall back to synthetic generation.

    Parameters
    ----------
    data_dir : str
        Directory expected to contain ``error_data.csv``.

    Returns
    -------
    pd.DataFrame
        DataFrame with at least a ``total_errors`` column.
    """
    csv_path = Path(data_dir) / "error_data.csv"

    if csv_path.is_file():
        logger.info("Loading data from %s", csv_path)
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        # If individual interface columns exist, aggregate them
        error_cols = [c for c in df.columns if c not in ("timestamp",)]
        if "total_errors" not in df.columns:
            df["total_errors"] = df[error_cols].sum(axis=1)
        return df

    logger.warning("%s not found.", csv_path)
    return generate_synthetic_data(csv_path)


def create_sequences(
    data: np.ndarray,
    seq_length: int = LSTM_SEQ_LENGTH,
    horizon: int = PREDICTION_HORIZON,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sliding-window input/target pairs.

    Parameters
    ----------
    data : np.ndarray
        1-D normalised time-series.
    seq_length : int
        Number of look-back time-steps per sample.
    horizon : int
        How many steps ahead the target is.

    Returns
    -------
    X : np.ndarray, shape ``(N, seq_length, 1)``
    y : np.ndarray, shape ``(N, 1)``
    """
    X, y = [], []
    for i in range(len(data) - seq_length - horizon + 1):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length + horizon - 1])
    return (
        np.array(X).reshape(-1, seq_length, 1),
        np.array(y).reshape(-1, 1),
    )


# ╭───────────────────────────────────────────────────────────────────────╮
# │  Training & evaluation                                                │
# ╰───────────────────────────────────────────────────────────────────────╯
def train_model(
    model: LSTMPredictor,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> list[float]:
    """Train the LSTM model with MSE loss and Adam optimiser.

    Parameters
    ----------
    model : LSTMPredictor
        The model to train (moved to *device* in-place).
    train_loader : DataLoader
        Batched training data.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate for Adam.
    device : torch.device
        CPU or CUDA device.

    Returns
    -------
    list[float]
        Per-epoch average training loss.
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimiser.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * X_batch.size(0)

        avg_loss = epoch_loss / len(train_loader.dataset)
        history.append(avg_loss)

        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            logger.info("Epoch %3d/%d — MSE loss: %.6f", epoch, epochs, avg_loss)

    return history


def evaluate_model(
    model: LSTMPredictor,
    test_loader: DataLoader,
    device: torch.device,
    data_min: float,
    data_max: float,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Compute RMSE and MAE on the test set (original scale).

    Parameters
    ----------
    model : LSTMPredictor
        Trained model.
    test_loader : DataLoader
        Batched test data.
    device : torch.device
        CPU or CUDA device.
    data_min, data_max : float
        Min / max used during normalisation (for inverse transform).

    Returns
    -------
    rmse : float
    mae : float
    actuals : np.ndarray
    predictions : np.ndarray
    """
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y_batch.numpy())

    preds_norm = np.concatenate(all_preds).flatten()
    targets_norm = np.concatenate(all_targets).flatten()

    # Inverse min-max scaling
    scale = data_max - data_min if data_max != data_min else 1.0
    preds_orig = preds_norm * scale + data_min
    targets_orig = targets_norm * scale + data_min

    rmse = float(np.sqrt(np.mean((preds_orig - targets_orig) ** 2)))
    mae = float(np.mean(np.abs(preds_orig - targets_orig)))

    return rmse, mae, targets_orig, preds_orig


# ╭───────────────────────────────────────────────────────────────────────╮
# │  Plotting                                                             │
# ╰───────────────────────────────────────────────────────────────────────╯
def save_prediction_plot(
    actuals: np.ndarray,
    predictions: np.ndarray,
    out_path: Path,
) -> None:
    """Save an *actual vs predicted* comparison plot.

    Parameters
    ----------
    actuals : np.ndarray
        Ground-truth error counts.
    predictions : np.ndarray
        Model predictions (same length as *actuals*).
    out_path : Path
        Destination PNG path.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(actuals, label="Actual", alpha=0.8, linewidth=0.9)
    ax.plot(predictions, label="Predicted", alpha=0.8, linewidth=0.9)
    ax.set_xlabel("Test sample index")
    ax.set_ylabel("Error count")
    ax.set_title(
        f"LSTM Predictive Model — {PREDICTION_HORIZON}-step-ahead forecast"
    )
    ax.legend()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Prediction plot saved to %s", out_path)


# ╭───────────────────────────────────────────────────────────────────────╮
# │  Prediction / alert mode                                              │
# ╰───────────────────────────────────────────────────────────────────────╯
def run_prediction(data_dir: str, device: torch.device) -> None:
    """Load the latest data window, run the saved model, and print an alert
    if the predicted error count exceeds ``ALERT_THRESHOLD``.

    Parameters
    ----------
    data_dir : str
        Directory containing ``error_data.csv``.
    device : torch.device
        CPU or CUDA device.
    """
    model_path = Path(MODEL_DIR) / "lstm_predictor.pth"
    if not model_path.is_file():
        logger.error("No saved model found at %s. Train first.", model_path)
        sys.exit(1)

    df = load_data(data_dir)
    series = df["total_errors"].values.astype(np.float32)
    data_min, data_max = series.min(), series.max()
    scale = data_max - data_min if data_max != data_min else 1.0
    normalised = (series - data_min) / scale

    if len(normalised) < LSTM_SEQ_LENGTH:
        logger.error(
            "Not enough data (%d points) for a sequence of length %d.",
            len(normalised),
            LSTM_SEQ_LENGTH,
        )
        sys.exit(1)

    # Take the most recent window
    window = normalised[-LSTM_SEQ_LENGTH:]
    x_tensor = torch.tensor(window, dtype=torch.float32).reshape(1, LSTM_SEQ_LENGTH, 1)

    model = LSTMPredictor(input_size=1)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    with torch.no_grad():
        pred_norm = model(x_tensor.to(device)).cpu().item()

    predicted_errors = pred_norm * scale + data_min
    logger.info(
        "Predicted error count (%d steps ahead): %.2f",
        PREDICTION_HORIZON,
        predicted_errors,
    )

    if predicted_errors > ALERT_THRESHOLD:
        logger.warning(
            "⚠  ALERT: Predicted errors (%.2f) exceed threshold (%.2f). "
            "Proactive remediation recommended!",
            predicted_errors,
            ALERT_THRESHOLD,
        )
    else:
        logger.info(
            "Predicted errors (%.2f) within acceptable range (threshold: %.2f).",
            predicted_errors,
            ALERT_THRESHOLD,
        )


# ╭───────────────────────────────────────────────────────────────────────╮
# │  Main pipeline                                                        │
# ╰───────────────────────────────────────────────────────────────────────╯
def main(args: argparse.Namespace) -> None:
    """End-to-end training pipeline: load → preprocess → train → evaluate → save.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ── prediction-only mode ───────────────────────────────────────────
    if args.predict:
        run_prediction(args.data_dir, device)
        return

    # ── load & preprocess ──────────────────────────────────────────────
    df = load_data(args.data_dir)
    series = df["total_errors"].values.astype(np.float32)

    # Min-max normalisation
    data_min, data_max = series.min(), series.max()
    scale = data_max - data_min if data_max != data_min else 1.0
    normalised = (series - data_min) / scale

    logger.info(
        "Data: %d samples | min=%.2f  max=%.2f | seq_len=%d  horizon=%d",
        len(series),
        data_min,
        data_max,
        LSTM_SEQ_LENGTH,
        PREDICTION_HORIZON,
    )

    # ── sliding-window sequences ───────────────────────────────────────
    X, y = create_sequences(normalised)
    logger.info("Created %d sequences.", len(X))

    if len(X) == 0:
        logger.error("Not enough data to create sequences. Aborting.")
        sys.exit(1)

    # ── chronological train / test split (80 / 20) ─────────────────────
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info("Train: %d | Test: %d", len(X_train), len(X_test))

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # ── model initialisation ───────────────────────────────────────────
    model = LSTMPredictor(input_size=1)
    logger.info("Model:\n%s", model)

    # ── training ───────────────────────────────────────────────────────
    epochs = args.epochs if args.epochs is not None else LSTM_EPOCHS
    history = train_model(model, train_loader, epochs, LSTM_LEARNING_RATE, device)

    # ── evaluation ─────────────────────────────────────────────────────
    rmse, mae, actuals, predictions = evaluate_model(
        model, test_loader, device, data_min, data_max
    )
    logger.info("Test RMSE: %.4f", rmse)
    logger.info("Test MAE:  %.4f", mae)

    # ── save model ─────────────────────────────────────────────────────
    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "lstm_predictor.pth"
    torch.save(model.state_dict(), model_path)
    logger.info("Model saved to %s", model_path)

    # ── optional plot ──────────────────────────────────────────────────
    if args.plot:
        plot_path = Path(LOG_DIR) / "prediction_plot.png"
        save_prediction_plot(actuals, predictions, plot_path)


# ╭───────────────────────────────────────────────────────────────────────╮
# │  CLI entry point                                                      │
# ╰───────────────────────────────────────────────────────────────────────╯
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Step 19 — LSTM-based predictive failure detection.  "
            "Trains on historical interface-error time-series to forecast "
            "error counts N steps ahead."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/",
        help="Directory containing error_data.csv (default: data/)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Training epochs (default: {LSTM_EPOCHS} from config.py)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save an actual-vs-predicted plot to logs/prediction_plot.png",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help=(
            "Skip training; load saved model and predict on latest data. "
            "Prints an alert if the forecast exceeds the threshold."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Example usage:
    #   python predictive_model.py                          # train with defaults
    #   python predictive_model.py --epochs 100 --plot      # custom epochs + plot
    #   python predictive_model.py --predict                # inference / alert mode
    main(parse_args())
