#!/usr/bin/env python3
"""Render the committed db-benchmark reference results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_INPUT = RESULTS_DIR / "db_benchmark_10m_m5_pro.csv"
DEFAULT_OUTPUT = RESULTS_DIR / "db_benchmark_10m_m5_pro.svg"

CASES = [
    ("1e7-Q1", "Q1"),
    ("1e7-Q2", "Q2"),
    ("1e7-Q4-encoded", "Q4\n(encoded)"),
    ("1e7-Q5", "Q5"),
]
TASKS = [("count", "Count"), ("materialize", "Materialize")]
ENGINES = [
    ("flash_join_bloom", "Flash Join (Bloom)", "#2563EB"),
    ("flash_join_no_bloom", "Flash Join (no Bloom)", "#10B981"),
    ("duckdb", "DuckDB", "#64748B"),
]


def read_results(path: Path) -> dict[tuple[str, str, str], float]:
    results: dict[tuple[str, str, str], float] = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            key = (row["case"], row["task"], row["engine"])
            if key in results:
                raise ValueError(f"duplicate result: {key}")
            results[key] = float(row["milliseconds"])

    expected = {
        (case, task, engine)
        for case, _ in CASES
        for task, _ in TASKS
        for engine, _, _ in ENGINES
    }
    missing = expected - results.keys()
    if missing:
        raise ValueError(f"missing results: {sorted(missing)}")
    return results


def render(input_path: Path, output_path: Path) -> None:
    results = read_results(input_path)
    output_format = output_path.suffix.removeprefix(".").lower() or "svg"
    if output_format not in {"svg", "png", "pdf"}:
        raise ValueError("output extension must be .svg, .png, or .pdf")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "svg.hashsalt": "flash-join-db-benchmark-10m",
        }
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharey=True)
    positions = np.arange(len(CASES))
    width = 0.24

    for axis, (task, task_label) in zip(axes, TASKS):
        for engine_index, (engine, label, color) in enumerate(ENGINES):
            values = [results[(case, task, engine)] for case, _ in CASES]
            axis.bar(
                positions + (engine_index - 1) * width,
                values,
                width,
                label=label,
                color=color,
                edgecolor="white",
                linewidth=0.7,
            )

        axis.set_title(task_label)
        axis.set_xticks(positions, [label for _, label in CASES])
        axis.set_yscale("log")
        axis.set_ylim(0.5, 100)
        axis.grid(axis="y", which="both", color="#CBD5E1", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Median runtime (ms, log scale) — lower is better")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.93),
    )
    figure.suptitle("db-benchmark join distributions — 10M probe rows, 6 threads", y=0.995)
    figure.text(
        0.5,
        0.015,
        "Median of 3 complete runs; each result is the median of 11 measurements",
        ha="center",
        color="#475569",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.88), w_pad=2.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        format=output_format,
        bbox_inches="tight",
        metadata={"Creator": "benchmarks/plot_results.py", "Date": None},
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.input, args.output)
    print(f"Chart: {args.output}")


if __name__ == "__main__":
    main()
