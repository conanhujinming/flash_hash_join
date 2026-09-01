from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import flash_join


def u64(values):
    return np.asarray(values, dtype=np.uint64)


def test_reports_adaptive_implementation():
    assert flash_join.implementation == "adaptive_perfect_or_cedardb_unchained"


def test_dense_unique_build_uses_identical_join_semantics():
    build_keys = u64([12, 10, 14, 11])
    build_values = u64([120, 100, 140, 110])
    probe_keys = u64([9, 10, 12, 13, 14, 15, 10])

    count, _ = flash_join.join_count(build_keys, build_values, probe_keys)
    keys, values, _ = flash_join.join_pairs(build_keys, build_values, probe_keys)

    assert count == 4
    assert Counter(zip(keys.tolist(), values.tolist())) == Counter(
        [(10, 100), (12, 120), (14, 140), (10, 100)]
    )


def test_count_preserves_multimap_semantics():
    build_keys = u64([1, 1, 2, 4])
    build_values = u64([10, 11, 20, 40])
    probe_keys = u64([0, 1, 2, 1, 3])

    count, seconds = flash_join.join_count(build_keys, build_values, probe_keys)

    assert count == 5
    assert seconds >= 0


def test_materialized_pairs_include_every_duplicate():
    keys, values, seconds = flash_join.join_pairs(
        u64([7, 7, 8]),
        u64([70, 71, 80]),
        u64([7, 8, 7, 9]),
    )

    assert Counter(zip(keys.tolist(), values.tolist())) == Counter(
        [(7, 70), (7, 71), (8, 80), (7, 70), (7, 71)]
    )
    assert seconds >= 0


def test_no_bloom_variant_has_identical_results():
    build_keys = u64([1, 1, 2, 4])
    build_values = u64([10, 11, 20, 40])
    probe_keys = u64([0, 1, 2, 1, 3])

    assert flash_join.join_count_no_bloom(
        build_keys, build_values, probe_keys
    )[0] == flash_join.join_count(build_keys, build_values, probe_keys)[0]

    assert flash_join.join_no_bloom(
        build_keys, build_values, probe_keys
    )[0] == flash_join.join(build_keys, build_values, probe_keys)[0]


@pytest.mark.parametrize(
    ("build_keys", "build_values", "probe_keys", "expected"),
    [
        ([], [], [], 0),
        ([], [], [1, 2, 3], 0),
        ([1, 2, 3], [10, 20, 30], [], 0),
        ([1, 2, 3], [10, 20, 30], [4, 5, 6], 0),
    ],
)
def test_empty_and_non_matching_inputs(build_keys, build_values, probe_keys, expected):
    count, _ = flash_join.join_count(u64(build_keys), u64(build_values), u64(probe_keys))
    assert count == expected


def test_randomized_count_matches_reference():
    rng = np.random.default_rng(20240831)
    build_keys = rng.integers(0, 257, size=120_000, dtype=np.uint64)
    build_values = np.arange(build_keys.size, dtype=np.uint64)
    probe_keys = rng.integers(0, 400, size=180_000, dtype=np.uint64)
    build_counts = Counter(build_keys.tolist())
    expected = sum(build_counts[key] for key in probe_keys.tolist())

    flash_join.set_num_threads(4)
    try:
        count, _ = flash_join.join_count(build_keys, build_values, probe_keys)
        count_no_bloom, _ = flash_join.join_count_no_bloom(
            build_keys, build_values, probe_keys
        )
    finally:
        flash_join.set_num_threads(0)

    assert count == expected
    assert count_no_bloom == expected


def test_rejects_shape_and_length_mismatches():
    with pytest.raises(ValueError, match="same length"):
        flash_join.join_count(u64([1, 2]), u64([10]), u64([1]))

    with pytest.raises(ValueError, match="one-dimensional"):
        flash_join.join_count(
            np.asarray([[1, 2]], dtype=np.uint64),
            np.asarray([[10, 20]], dtype=np.uint64),
            u64([1]),
        )


def test_persistent_worker_pool_supports_concurrent_joins():
    build_keys = np.arange(1024, dtype=np.uint64)
    build_values = build_keys * 10
    probe_keys = np.arange(200_000, dtype=np.uint64) % 2048
    expected = int(np.count_nonzero(probe_keys < 1024))

    flash_join.set_num_threads(4)
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                executor.map(
                    lambda _: flash_join.join_count(
                        build_keys, build_values, probe_keys
                    )[0],
                    range(8),
                )
            )
    finally:
        flash_join.set_num_threads(0)

    assert results == [expected] * 8
