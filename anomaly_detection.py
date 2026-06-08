"""
Anomaly Detection — Step 18
============================
Train an Isolation Forest model on exported network telemetry
(traffic + error counters) to flag anomalous data points.

Usage
-----
    # Basic run (uses data/ directory, default contamination from config.py)
    python anomaly_detection.py

    # Custom data directory, override contamination, generate plot
    python anomaly_detection.py --data-dir data/ --contamination 0.03 --plot
"""

import argparse
import os
import sys
import warnings
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config import ANOMALY_CONTAMINATION, MODEL_DIR, LOG_DIR


# ─────────────────────────────────────────────
#  Synthetic-data generator (fallback)
# ─────────────────────────────────────────────

def _generate_synthetic_data(data_dir: str) -> None:
    """Generate 30 days of hourly synthetic traffic / error CSVs.

    Anomalies are injected at ~5 % of data points by adding large
    spikes in traffic and error counts.

    Parameters
    ----------
    data_dir : str
        Directory where ``traffic_data.csv`` and ``error_data.csv``
        will be written.
    """
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    hours = 30 * 24  # 720 data points
    timestamps = [
        (datetime(2026, 4, 1) + timedelta(hours=h)).isoformat()
        for h in range(hours)
    ]

    # ── traffic_data.csv ──
    in_octets = rng.normal(loc=5000, scale=800, size=hours).clip(min=0)
    out_octets = rng.normal(loc=4500, scale=700, size=hours).clip(min=0)

    # Inject anomalies (~5 %)
    anomaly_mask = rng.random(hours) < 0.05
    in_octets[anomaly_mask] *= rng.uniform(5, 15, size=anomaly_mask.sum())
    out_octets[anomaly_mask] *= rng.uniform(5, 12, size=anomaly_mask.sum())

    traffic_df = pd.DataFrame({
        "timestamp": timestamps,
        "in_octets": in_octets.round(2),
        "out_octets": out_octets.round(2),
    })
    traffic_path = os.path.join(data_dir, "traffic_data.csv")
    traffic_df.to_csv(traffic_path, index=False)

    # ── error_data.csv ──
    in_errors = rng.poisson(lam=2, size=hours).astype(float)
    out_errors = rng.poisson(lam=1, size=hours).astype(float)

    in_errors[anomaly_mask] += rng.integers(50, 200, size=anomaly_mask.sum())
    out_errors[anomaly_mask] += rng.integers(30, 150, size=anomaly_mask.sum())

    error_df = pd.DataFrame({
        "timestamp": timestamps,
        "in_errors": in_errors,
        "out_errors": out_errors,
    })
    error_path = os.path.join(data_dir, "error_data.csv")
    error_df.to_csv(error_path, index=False)

    print(f"[synthetic] Wrote {traffic_path}  ({len(traffic_df)} rows)")
    print(f"[synthetic] Wrote {error_path}  ({len(error_df)} rows)")


# ─────────────────────────────────────────────
#  Data loading & preprocessing
# ─────────────────────────────────────────────

def load_data(data_dir: str) -> pd.DataFrame:
    """Load and merge traffic + error CSVs into a single DataFrame.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing ``traffic_data.csv`` and
        ``error_data.csv``.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with a ``timestamp`` column and all numeric
        feature columns.

    Raises
    ------
    FileNotFoundError
        If either CSV is still missing after synthetic-data generation.
    """
    traffic_path = os.path.join(data_dir, "traffic_data.csv")
    error_path = os.path.join(data_dir, "error_data.csv")

    # Fall back to synthetic data when real exports are absent.
    if not os.path.isfile(traffic_path) or not os.path.isfile(error_path):
        warnings.warn(
            "CSV files not found — generating synthetic sample data.  "
            "Run export_data.py first to use real InfluxDB telemetry.",
            stacklevel=2,
        )
        _generate_synthetic_data(data_dir)

    # Final check after potential generation.
    for p in (traffic_path, error_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Required file missing: {p}")

    traffic_df = pd.read_csv(traffic_path)
    error_df = pd.read_csv(error_path)

    # Merge on shared timestamp column.
    merged = pd.merge(traffic_df, error_df, on="timestamp", how="outer")
    merged.sort_values("timestamp", inplace=True)
    merged.reset_index(drop=True, inplace=True)

    return merged


def preprocess(df: pd.DataFrame):
    """Handle missing values and normalise numeric features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw merged data (must contain a ``timestamp`` column).

    Returns
    -------
    tuple[np.ndarray, list[str], StandardScaler]
        * ``X_scaled`` — (n_samples, n_features) normalised array
        * ``feature_cols`` — list of column names used as features
        * ``scaler`` — fitted ``StandardScaler`` instance
    """
    feature_cols = [c for c in df.columns if c != "timestamp"]

    # Fill missing numeric values with column medians.
    df[feature_cols] = df[feature_cols].apply(
        lambda col: col.fillna(col.median())
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols].values)

    return X_scaled, feature_cols, scaler


# ─────────────────────────────────────────────
#  Model training
# ─────────────────────────────────────────────

def train_model(
    X: np.ndarray,
    contamination: float = ANOMALY_CONTAMINATION,
) -> IsolationForest:
    """Fit an Isolation Forest on the preprocessed feature matrix.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix of shape ``(n_samples, n_features)``.
    contamination : float
        Expected proportion of anomalies in the dataset.

    Returns
    -------
    IsolationForest
        Fitted model.
    """
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)
    return model


# ─────────────────────────────────────────────
#  Persistence helpers
# ─────────────────────────────────────────────

def save_model(model: IsolationForest, model_dir: str = MODEL_DIR) -> str:
    """Serialise the trained model to disk via joblib.

    Parameters
    ----------
    model : IsolationForest
        Fitted model.
    model_dir : str
        Target directory (created if absent).

    Returns
    -------
    str
        Absolute path to the saved ``.pkl`` file.
    """
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "anomaly_model.pkl")
    joblib.dump(model, path)
    return path


def save_results(df: pd.DataFrame, data_dir: str) -> str:
    """Write the labelled results to ``anomaly_results.csv``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that already contains an ``anomaly_label`` column.
    data_dir : str
        Output directory.

    Returns
    -------
    str
        Path to the saved CSV.
    """
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "anomaly_results.csv")
    df.to_csv(path, index=False)
    return path


# ─────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────

def plot_anomalies(df: pd.DataFrame, log_dir: str = LOG_DIR) -> str:
    """Render a time-series chart with anomalies highlighted in red.

    The plot is saved to ``<log_dir>/anomaly_plot.png``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``timestamp``, ``in_octets``, and ``anomaly_label``
        columns.
    log_dir : str
        Output directory for the PNG file.

    Returns
    -------
    str
        Absolute path to the saved plot.
    """
    # Import lazily so the script still works on headless servers that
    # lack a display backend.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(log_dir, exist_ok=True)

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    values = df["in_octets"] if "in_octets" in df.columns else df.iloc[:, 1]
    labels = df["anomaly_label"]

    normal = labels == 1
    anomaly = labels == -1

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ts[normal], values[normal], ".", color="steelblue",
            markersize=3, label="Normal")
    ax.scatter(ts[anomaly], values[anomaly], color="red",
               s=25, zorder=5, label="Anomaly")
    ax.set_title("Anomaly Detection — Isolation Forest")
    ax.set_xlabel("Time")
    ax.set_ylabel("in_octets")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(log_dir, "anomaly_plot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────
#  Summary printer
# ─────────────────────────────────────────────

def print_summary(labels: np.ndarray) -> None:
    """Print a human-readable summary of detection results.

    Parameters
    ----------
    labels : np.ndarray
        Array of labels (1 = normal, -1 = anomaly).
    """
    total = len(labels)
    anomalies = int((labels == -1).sum())
    pct = anomalies / total * 100 if total else 0.0

    print("\n" + "=" * 50)
    print("  Anomaly Detection Summary")
    print("=" * 50)
    print(f"  Total data points  : {total:,}")
    print(f"  Anomalies detected : {anomalies:,}")
    print(f"  Anomaly percentage : {pct:.2f} %")
    print("=" * 50 + "\n")


# ─────────────────────────────────────────────
#  CLI entry-point
# ─────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    """Build and parse command-line arguments.

    Parameters
    ----------
    argv : list[str] | None
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description=(
            "Step 18 — Train an Isolation Forest anomaly detector on "
            "exported network telemetry data."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/",
        help="Directory containing traffic_data.csv and error_data.csv "
             "(default: data/)",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=None,
        help=(
            f"Override the anomaly contamination rate "
            f"(default from config.py: {ANOMALY_CONTAMINATION})"
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate a time-series anomaly plot (saved to logs/anomaly_plot.png)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Run the full anomaly-detection pipeline."""
    args = parse_args(argv)

    contamination = args.contamination if args.contamination is not None else ANOMALY_CONTAMINATION

    # 1. Load data ────────────────────────────
    print("[*] Loading data …")
    df = load_data(args.data_dir)
    print(f"    Loaded {len(df):,} rows  |  columns: {list(df.columns)}")

    # 2. Preprocess ───────────────────────────
    print("[*] Preprocessing (fill NaN, normalise) …")
    X_scaled, feature_cols, scaler = preprocess(df)
    print(f"    Feature matrix shape: {X_scaled.shape}")

    # 3. Train Isolation Forest ───────────────
    print(f"[*] Training Isolation Forest (contamination={contamination}) …")
    model = train_model(X_scaled, contamination=contamination)

    # 4. Predict labels ───────────────────────
    labels = model.predict(X_scaled)
    df["anomaly_label"] = labels

    # Also store the anomaly score (lower → more anomalous).
    df["anomaly_score"] = model.decision_function(X_scaled)

    # 5. Save model ───────────────────────────
    model_path = save_model(model)
    print(f"[+] Model saved → {model_path}")

    # 6. Save labelled results ────────────────
    results_path = save_results(df, args.data_dir)
    print(f"[+] Results saved → {results_path}")

    # 7. Summary ──────────────────────────────
    print_summary(labels)

    # 8. Optional plot ────────────────────────
    if args.plot:
        try:
            plot_path = plot_anomalies(df)
            print(f"[+] Plot saved → {plot_path}")
        except Exception as exc:
            print(f"[!] Plotting failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
