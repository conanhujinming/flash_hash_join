#!/usr/bin/env python3
"""Reproducible Flash Join versus DuckDB benchmark.

Data ingestion is excluded for both engines. Flash Join reports its C++ core
time; DuckDB tables are created before timing and only the SQL join is timed.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd

import flash_join


@dataclass
class JoinCase:
    name: str
    build_keys: np.ndarray
    build_values: np.ndarray
    probe_keys: np.ndarray


def synthetic_case(
    build_size: int, probe_size: int, match_rate: float, seed: int
) -> JoinCase:
    rng = np.random.default_rng(seed)
    build_keys = np.arange(build_size, dtype=np.uint64)
    rng.shuffle(build_keys)
    build_values = rng.integers(0, 1 << 32, build_size, dtype=np.uint64)
    matching_rows = round(probe_size * match_rate)
    probe_keys = np.empty(probe_size, dtype=np.uint64)
    probe_keys[:matching_rows] = rng.integers(
        0, build_size, matching_rows, dtype=np.uint64
    )
    probe_keys[matching_rows:] = rng.integers(
        build_size, build_size * 2, probe_size - matching_rows, dtype=np.uint64
    )
    rng.shuffle(probe_keys)
    return JoinCase("synthetic", build_keys, build_values, probe_keys)


def _sample_all(
    values: np.ndarray, size: int, rng: np.random.Generator
) -> np.ndarray:
    """NumPy equivalent of join-datagen.R's sample_all helper."""
    if values.size > size:
        raise ValueError("values must fit in the requested sample")
    if values.size == size:
        result = values.copy()
    else:
        result = np.concatenate(
            [values, rng.choice(values, size=size - values.size, replace=True)]
        )
    rng.shuffle(result)
    return result


def _split_xlr(
    distinct_keys: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return the x+l and x+r domains from db-benchmark split_xlr."""
    expanded_size = distinct_keys * 11 // 10
    common_size = distinct_keys * 9 // 10
    keys = rng.permutation(expanded_size).astype(np.uint64, copy=False) + 1
    common = keys[:common_size]
    left = keys[common_size:distinct_keys]
    right = keys[distinct_keys:expanded_size]
    return np.concatenate([common, left]), np.concatenate([common, right])


def db_benchmark_cases(size: int, seed: int) -> Iterable[JoinCase]:
    """Port the key distributions from db-benchmark/_data/join-datagen.R.

    The original generator requires R/data.table. This in-memory port keeps its
    90% common + 10% side-only domains and sample_all multiplicities, while only
    materializing columns consumed by this uint64 join benchmark.
    """
    if size < 10_000_000 or size % 1_000_000 != 0:
        raise ValueError("db-benchmark size must be >= 10,000,000 and divisible by 1,000,000")

    rng = np.random.default_rng(seed)
    definitions = [
        ("Q1", size // 1_000_000),
        ("Q2", size // 1_000),
        ("Q5", size),
    ]
    size_label = f"{size // 10 ** (len(str(size)) - 1)}e{len(str(size)) - 1}"
    for query, distinct_keys in definitions:
        lhs_domain, rhs_domain = _split_xlr(distinct_keys, rng)
        probe_keys = _sample_all(lhs_domain, size, rng)
        build_keys = _sample_all(rhs_domain, distinct_keys, rng)
        build_values = np.arange(distinct_keys, dtype=np.uint64)
        yield JoinCase(
            f"{size_label}-{query}", build_keys, build_values, probe_keys
        )
        if query == "Q2":
            # id5 is the string encoding of id2 in the original CSV. Encoding
            # it back to uint64 yields the same equality distribution.
            yield JoinCase(
                f"{size_label}-Q4-encoded", build_keys, build_values, probe_keys
            )


def encode_join_keys(
    build_keys: pd.Series, probe_keys: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    if pd.api.types.is_numeric_dtype(build_keys) and pd.api.types.is_numeric_dtype(
        probe_keys
    ):
        return (
            np.ascontiguousarray(build_keys, dtype=np.uint64),
            np.ascontiguousarray(probe_keys, dtype=np.uint64),
        )

    combined = pd.concat([build_keys, probe_keys], ignore_index=True)
    codes, _ = pd.factorize(combined, sort=False)
    if np.any(codes < 0):
        raise ValueError("NULL join keys are not supported")
    split = build_keys.size
    return (
        np.ascontiguousarray(codes[:split], dtype=np.uint64),
        np.ascontiguousarray(codes[split:], dtype=np.uint64),
    )


def discover_h2o_cases(data_dir: Path) -> list[JoinCase]:
    cases: list[JoinCase] = []
    groups: dict[str, list[str]] = {}
    for filename in glob.glob(str(data_dir / "J1_*.csv")):
        group = Path(filename).name.split("_")[1]
        groups.setdefault(group, []).append(filename)

    for group in sorted(groups):
        base_digit = group[0]
        paths = {
            "x": data_dir / f"J1_{group}_NA_0_0.csv",
            "small": data_dir / f"J1_{group}_{base_digit}e1_0_0.csv",
            "medium": data_dir / f"J1_{group}_{base_digit}e4_0_0.csv",
            "big": data_dir / f"J1_{group}_{base_digit}e7_0_0.csv",
        }
        if not all(path.exists() for path in paths.values()):
            continue

        frames = {name: pd.read_csv(path) for name, path in paths.items()}
        definitions = [
            ("Q1", "small", "id1"),
            ("Q2", "medium", "id2"),
            ("Q4-encoded", "medium", "id5"),
            ("Q5", "big", "id3"),
        ]
        for query, build_name, key in definitions:
            build = frames[build_name]
            probe = frames["x"]
            if key not in build or key not in probe or "v2" not in build:
                continue
            build_keys, probe_keys = encode_join_keys(build[key], probe[key])
            cases.append(
                JoinCase(
                    f"{group}-{query}",
                    build_keys,
                    np.arange(build_keys.size, dtype=np.uint64),
                    probe_keys,
                )
            )
    return cases


def flash_times(
    case: JoinCase, repeats: int, materialize: bool, use_bloom: bool
) -> tuple[int, float]:
    if materialize:
        function = flash_join.join if use_bloom else flash_join.join_no_bloom
    else:
        function = (
            flash_join.join_count
            if use_bloom
            else flash_join.join_count_no_bloom
        )
    function(case.build_keys, case.build_values, case.probe_keys)
    observations = [
        function(case.build_keys, case.build_values, case.probe_keys)
        for _ in range(repeats)
    ]
    counts = {int(result[0]) for result in observations}
    if len(counts) != 1:
        raise RuntimeError(f"Flash Join returned inconsistent counts: {counts}")
    return counts.pop(), median(float(result[1]) for result in observations)


def duckdb_times(
    case: JoinCase, repeats: int, threads: int, materialize: bool
) -> tuple[int, float]:
    connection = duckdb.connect(database=":memory:")
    connection.execute(f"PRAGMA threads={threads}")
    build_frame = pd.DataFrame({"key": case.build_keys, "value": case.build_values})
    probe_frame = pd.DataFrame({"key": case.probe_keys})
    connection.execute("CREATE TABLE build_side AS SELECT * FROM build_frame")
    connection.execute("CREATE TABLE probe_side AS SELECT * FROM probe_frame")

    count_sql = "SELECT count(*) FROM build_side b JOIN probe_side p ON b.key = p.key"
    if not materialize:
        connection.execute(count_sql).fetchone()
        observations = []
        count = 0
        for _ in range(repeats):
            start = perf_counter()
            count = int(connection.execute(count_sql).fetchone()[0])
            observations.append(perf_counter() - start)
    else:
        materialize_sql = (
            "CREATE OR REPLACE TEMP TABLE joined AS "
            "SELECT p.key, b.value FROM build_side b JOIN probe_side p ON b.key = p.key"
        )
        connection.execute(materialize_sql)
        observations = []
        count = 0
        for _ in range(repeats):
            start = perf_counter()
            connection.execute(materialize_sql)
            count = int(connection.execute("SELECT count(*) FROM joined").fetchone()[0])
            observations.append(perf_counter() - start)
    connection.close()
    return count, median(observations)


def benchmark(case: JoinCase, repeats: int, threads: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task, materialize in (("count", False), ("materialize", True)):
        flash_results = [
            (
                "flash_join_bloom" if use_bloom else "flash_join_no_bloom",
                *flash_times(case, repeats, materialize, use_bloom),
            )
            for use_bloom in (True, False)
        ]
        duck_count, duck_seconds = duckdb_times(case, repeats, threads, materialize)
        for engine, count, _ in flash_results:
            if count != duck_count:
                raise RuntimeError(
                    f"{case.name}/{task}: result mismatch: "
                    f"{engine}={count}, DuckDB={duck_count}"
                )
        engine_times = [
            (engine, seconds) for engine, _, seconds in flash_results
        ] + [("duckdb", duck_seconds)]
        for engine, seconds in engine_times:
            rows.append(
                {
                    "case": case.name,
                    "task": task,
                    "engine": engine,
                    "threads": threads,
                    "build_rows": case.build_keys.size,
                    "probe_rows": case.probe_keys.size,
                    "result_rows": duck_count,
                    "seconds": seconds,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--repeats", type=int, default=5)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data-dir", type=Path)
    source.add_argument("--db-benchmark-size", type=int)
    parser.add_argument("--build-size", type=int, default=200_000)
    parser.add_argument("--probe-size", type=int, default=1_000_000)
    parser.add_argument("--match-rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=108)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark.csv"))
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1:
        parser.error("--threads and --repeats must be positive")
    if args.build_size < 1 or args.probe_size < 0:
        parser.error("--build-size must be positive and --probe-size must be non-negative")
    if not 0 <= args.match_rate <= 1:
        parser.error("--match-rate must be between zero and one")
    if args.db_benchmark_size and (
        args.db_benchmark_size < 10_000_000
        or args.db_benchmark_size % 1_000_000 != 0
    ):
        parser.error(
            "--db-benchmark-size must be >= 10000000 and divisible by 1000000"
        )

    flash_join.set_num_threads(args.threads)
    if args.data_dir:
        cases: Iterable[JoinCase] = discover_h2o_cases(args.data_dir)
    elif args.db_benchmark_size:
        cases = db_benchmark_cases(args.db_benchmark_size, args.seed)
    else:
        cases = [
            synthetic_case(
                args.build_size, args.probe_size, args.match_rate, args.seed
            )
        ]
    rows: list[dict[str, object]] = []
    for case in cases:
        rows.extend(benchmark(case, args.repeats, args.threads))
    if not rows:
        parser.error("no complete H2O db-benchmark join suites were found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nRaw results: {args.output}")


if __name__ == "__main__":
    main()
