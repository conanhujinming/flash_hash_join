# Flash Join

Flash Join is a parallel `uint64` hash join for NumPy. Version 0.2 replaces the
old linear-probing table with the **unchained** join table described in
CedarDB's [Simple, Efficient, and Robust Hash Tables for Join Processing](https://cedardb.com/blog/simple_efficient_hash_tables/)
and its linked [DaMoN '24 paper](https://db.in.tum.de/~birler/papers/hashtable.pdf).

The project intentionally exposes a small engine rather than claiming to be a
general database replacement. `benchmarks/benchmark.py` compares the join core
with DuckDB using the same thread count and validates result cardinality before
reporting timings.

## CedarDB design correspondence

| CedarDB technique | Flash Join implementation |
| --- | --- |
| Power-of-two directory sized as `2^ceil(log2(1.125n))` | `UnchainedHashTable::directory_capacity` |
| High hash bits select a directory slot | Shift-only `slot_for` |
| Dense adjacency array for collisions and duplicates | Consecutive `BuildTuple` ranges |
| Neighboring entries delimit each range | Branch-free `directory[-1]` sentinel and packed end offsets |
| Per-slot 16-bit register-blocked Bloom filter | Compile-time `UseBloomFilter` policy; enabled entries use their low 16 bits |
| Four-bit tags from a 4 KiB, 2048-entry table | All 1820 four-of-sixteen tags plus deterministic uniform padding |
| CRC hash specialized for 64-bit keys | Two hardware CRC32C operations and the paper's `0x2545F4914F6CDD1D` multiply |
| Parallel, lock-free build via partition ownership | Build-side hash partitioning followed by per-partition count/prefix/copy |
| Dense writes during tuple collection | Small software write-combine buffers |
| Tight probe loop with CPU-managed memory-level parallelism | No explicit prefetch; Bloom-enabled non-matches stop at the fused check |

Unlike the previous implementation, duplicate build keys are never discarded:
the table has the multimap semantics required by relational joins.

## Repository layout

```text
flash_hash_join/
├── include/flash_join/        # Hashing, parallel helpers, unchained table
├── src/                       # Python extension bindings and join operators
├── tests/                     # Correctness and Bloom-policy equivalence tests
├── benchmarks/                # Synthetic and H2O/DuckDB benchmarks
├── scripts/                   # Dataset generation utilities
├── CMakeLists.txt             # Native target
└── pyproject.toml             # Python build/package metadata
```

The historical benchmark charts in the repository root predate the unchained
implementation. New benchmark runs write raw measurements under `artifacts/`,
which is intentionally ignored by Git.

## Build and test

Python 3.9+, CMake 3.20+, and a C++17 compiler are required. Hardware CRC32C is
used on SSE4.2 x86-64 and CRC-capable AArch64; other targets use a portable
fallback.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,benchmark]'
pytest
```

The extension no longer requires mimalloc to build. The existing submodule is
kept for reproducibility of older revisions but is not part of the v0.2 target.

## Python API

```python
import numpy as np
import flash_join

build_keys = np.array([1, 1, 2], dtype=np.uint64)
build_values = np.array([10, 11, 20], dtype=np.uint64)
probe_keys = np.array([1, 2, 3], dtype=np.uint64)

flash_join.set_num_threads(8)

# Includes both rows for duplicate key 1.
count, core_seconds = flash_join.join_count(
    build_keys, build_values, probe_keys
)

# Materializes internally, matching the DuckDB materialization benchmark.
count, core_seconds = flash_join.join(
    build_keys, build_values, probe_keys
)

# Retrieve materialized columns when the rows themselves are needed.
joined_keys, joined_values, core_seconds = flash_join.join_pairs(
    build_keys, build_values, probe_keys
)

# Compile-time specialization with all Bloom work removed.
count, core_seconds = flash_join.join_count_no_bloom(
    build_keys, build_values, probe_keys
)
```

The API deliberately does not preserve the old combinatorial `adaptive_*`,
`*_bloom`, or `*_radix` aliases. The only variants are the default CedarDB Bloom
layout and the explicit `*_no_bloom` experiment. Build-side partitioning remains
an internal parallel-build stage rather than a separate join algorithm.

## Benchmark

Run the deterministic synthetic comparison:

```bash
python benchmarks/benchmark.py \
  --threads 8 \
  --build-size 1000000 \
  --probe-size 10000000 \
  --match-rate 0.5 \
  --repeats 5
```

`--match-rate` controls join selectivity. The fused Bloom fast path is designed
for selective workloads, so benchmark both the expected production rate and a
high-match stress case rather than relying on one aggregate number. Every run
reports `flash_join_bloom`, `flash_join_no_bloom`, and `duckdb`.

Run the original db-benchmark join key distributions directly in memory:

```bash
python benchmarks/benchmark.py \
  --db-benchmark-size 10000000 \
  --threads 8 \
  --repeats 11
```

This is a NumPy port of `db-benchmark/_data/join-datagen.R`: it retains the 90%
common, 10% side-only domains and `sample_all` multiplicities, while generating
only the columns consumed by this uint64 join. Q4 dictionary-encodes the original
`id5` string representation back to its `id2` equality domain.

Alternatively, if R and data.table are installed, generate and read the original
CSV files:

```bash
git submodule update --init db-benchmark
bash scripts/generate_data.sh
python benchmarks/benchmark.py --data-dir data --threads 8 --repeats 5
```

For both engines, ingestion happens before timing. The benchmark records the
median join time, checks identical row counts, and writes the raw CSV to
`artifacts/benchmark.csv`.

## License

[MIT](LICENSE)
