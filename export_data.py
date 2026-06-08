#!/usr/bin/env python3
"""
export_data.py — Step 17: Export Historical Telemetry to CSV
============================================================
Part of the Network Digital Twin project (Net.pdf, Step 17).

Connects to the InfluxDB telemetry store and exports three CSV datasets
that are consumed by downstream ML pipelines (anomaly detection, LSTM
forecasting, and RL reward shaping):

  • interface_status.csv — up/down status history per interface
  • traffic_data.csv     — inbound octet (traffic-rate) history
  • error_data.csv       — inbound error-counter history

Usage
-----
    python export_data.py                      # defaults: 30 days → data/
    python export_data.py --days 7             # last week only
    python export_data.py --output-dir exports # custom output folder
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from influxdb_client import InfluxDBClient
from influxdb_client.client.exceptions import InfluxDBError

from config import (
    INFLUXDB_BUCKET,
    INFLUXDB_ORG,
    INFLUXDB_TOKEN,
    INFLUXDB_URL,
)


# ── dataset definitions ────────────────────────────────────────────────
# Each entry maps a friendly filename to the InfluxDB (measurement, field)
# pair that should be exported.
DATASETS: dict[str, dict[str, str]] = {
    "interface_status.csv": {
        "measurement": "interface",
        "field": "status",
    },
    "traffic_data.csv": {
        "measurement": "interface",
        "field": "in_octets",
    },
    "error_data.csv": {
        "measurement": "interface",
        "field": "in_errors",
    },
}


def build_flux_query(bucket: str, measurement: str, field: str, days: int) -> str:
    """Return a Flux query string for *field* from *measurement*.

    Parameters
    ----------
    bucket : str
        InfluxDB bucket name.
    measurement : str
        Target measurement (e.g. ``"interface"``).
    field : str
        Target field key (e.g. ``"status"``, ``"in_octets"``).
    days : int
        Number of past days to include.

    Returns
    -------
    str
        A ready-to-execute Flux query.
    """
    return (
        f'from(bucket: "{bucket}")\n'
        f"  |> range(start: -{days}d)\n"
        f'  |> filter(fn: (r) => r["_measurement"] == "{measurement}")\n'
        f'  |> filter(fn: (r) => r["_field"] == "{field}")\n'
        f"  |> pivot(rowKey: [\"_time\"], columnKey: [\"_field\"], valueColumn: \"_value\")\n"
        f'  |> keep(columns: ["_time", "host", "interface", "{field}"])'
    )


def query_to_dataframe(
    client: InfluxDBClient,
    query: str,
    org: str,
) -> pd.DataFrame:
    """Execute a Flux query and return the result as a DataFrame.

    Parameters
    ----------
    client : InfluxDBClient
        An authenticated InfluxDB client instance.
    query : str
        The Flux query to run.
    org : str
        InfluxDB organization name.

    Returns
    -------
    pd.DataFrame
        Query results (may be empty if no data matched).
    """
    query_api = client.query_api()
    tables = query_api.query_data_frame(query=query, org=org)

    # query_data_frame may return a list of DataFrames when multiple tables
    # are present; concatenate them into a single frame.
    if isinstance(tables, list):
        df = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    else:
        df = tables

    # Drop InfluxDB-internal columns that are not useful downstream.
    drop_cols = [c for c in ("result", "table") if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    return df


def export_datasets(
    days: int = 30,
    output_dir: str = "data",
) -> dict[str, int]:
    """Connect to InfluxDB and export each dataset to CSV.

    Parameters
    ----------
    days : int
        Number of historical days to query.
    output_dir : str
        Directory where CSV files will be written.

    Returns
    -------
    dict[str, int]
        Mapping of filename → number of rows exported.

    Raises
    ------
    ConnectionError
        If the InfluxDB server cannot be reached.
    """
    # Ensure the output directory exists.
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Open client and verify connectivity.
    client: Optional[InfluxDBClient] = None
    try:
        client = InfluxDBClient(
            url=INFLUXDB_URL,
            token=INFLUXDB_TOKEN,
            org=INFLUXDB_ORG,
            timeout=10_000,  # 10 s connect/read timeout
        )
        health = client.health()
        if health.status != "pass":
            raise ConnectionError(
                f"InfluxDB health-check failed: {health.message}"
            )
        print(f"✔ Connected to InfluxDB at {INFLUXDB_URL}")
    except Exception as exc:
        if client is not None:
            client.close()
        raise ConnectionError(
            f"Cannot reach InfluxDB at {INFLUXDB_URL}: {exc}"
        ) from exc

    summary: dict[str, int] = {}

    try:
        for filename, spec in DATASETS.items():
            query = build_flux_query(
                bucket=INFLUXDB_BUCKET,
                measurement=spec["measurement"],
                field=spec["field"],
                days=days,
            )
            print(f"  Querying {spec['measurement']}.{spec['field']} "
                  f"(last {days} days) …")
            df = query_to_dataframe(client, query, INFLUXDB_ORG)

            csv_path = out_path / filename
            df.to_csv(csv_path, index=False)
            summary[filename] = len(df)
    except InfluxDBError as exc:
        print(f"✘ InfluxDB query error: {exc}", file=sys.stderr)
        raise
    finally:
        client.close()

    return summary


def print_summary(summary: dict[str, int], output_dir: str) -> None:
    """Print a human-readable export summary to stdout.

    Parameters
    ----------
    summary : dict[str, int]
        Mapping of filename → row count returned by :func:`export_datasets`.
    output_dir : str
        The directory where files were saved (for display purposes).
    """
    print("\n╔══════════════════════════════════════════╗")
    print("║        Export Summary (Step 17)          ║")
    print("╠══════════════════════════════════════════╣")
    for fname, rows in summary.items():
        path_display = os.path.join(output_dir, fname)
        print(f"║  {path_display:<28s} {rows:>6,} rows ║")
    total = sum(summary.values())
    print("╠══════════════════════════════════════════╣")
    print(f"║  {'Total':<28s} {total:>6,} rows ║")
    print("╚══════════════════════════════════════════╝")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list[str] | None
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    argparse.Namespace
        Parsed arguments with ``days`` and ``output_dir`` attributes.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Step 17 — Export historical telemetry data from InfluxDB "
            "to CSV files for ML training."
        ),
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days of history to export (default: 30).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to write CSV files into (default: data/).",
    )
    return parser.parse_args(argv)


# ── entry-point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    print(f"Step 17 · Exporting last {args.days} day(s) of telemetry to "
          f"'{args.output_dir}/' …\n")

    try:
        results = export_datasets(days=args.days, output_dir=args.output_dir)
        print_summary(results, args.output_dir)
    except ConnectionError as exc:
        print(f"\n✘ Connection failed: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\n✘ Unexpected error: {exc}", file=sys.stderr)
        sys.exit(1)
