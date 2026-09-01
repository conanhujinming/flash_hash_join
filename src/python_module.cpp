#include "flash_join/parallel.hpp"
#include "flash_join/unchained_hash_table.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace flash_join {
namespace {

using UInt64Array = py::array_t<std::uint64_t, py::array::c_style>;

constexpr std::size_t kProbeGrain = 32U * 1024U;
constexpr std::size_t kCacheLineSize = 64;

std::atomic<std::size_t> configured_threads{0};

struct Inputs {
    const std::uint64_t* build_keys;
    const std::uint64_t* build_values;
    const std::uint64_t* probe_keys;
    std::size_t build_size;
    std::size_t probe_size;
};

struct JoinMetrics {
    std::size_t count;
    double seconds;
};

struct OutputTuple {
    std::uint64_t key;
    std::uint64_t value;
};

struct MaterializedJoin {
    std::vector<OutputTuple> tuples;
    double seconds;
};

struct alignas(kCacheLineSize) PaddedCount {
    std::size_t value = 0;
};

std::size_t default_thread_count() noexcept {
    return std::max<std::size_t>(1, std::thread::hardware_concurrency());
}

std::size_t thread_count() noexcept {
    const auto configured = configured_threads.load(std::memory_order_relaxed);
    return configured == 0 ? default_thread_count() : configured;
}

void set_num_threads(std::size_t threads) {
    configured_threads.store(threads, std::memory_order_relaxed);
}

Inputs validate_inputs(
    const UInt64Array& build_keys,
    const UInt64Array& build_values,
    const UInt64Array& probe_keys) {
    if (build_keys.ndim() != 1 || build_values.ndim() != 1 || probe_keys.ndim() != 1) {
        throw py::value_error("build_keys, build_values, and probe_keys must be one-dimensional");
    }
    if (build_keys.size() != build_values.size()) {
        throw py::value_error("build_keys and build_values must have the same length");
    }
    return Inputs{
        build_keys.data(),
        build_values.data(),
        probe_keys.data(),
        static_cast<std::size_t>(build_keys.size()),
        static_cast<std::size_t>(probe_keys.size()),
    };
}

template <bool UseBloomFilter>
JoinMetrics run_count(const Inputs& input) {
    const auto start = std::chrono::steady_clock::now();
    const auto threads = thread_count();
    const UnchainedHashTable<UseBloomFilter> table(
        input.build_keys, input.build_values, input.build_size, threads);

    const auto probe_threads = parallel::effective_thread_count(
        threads, input.probe_size, kProbeGrain);
    std::vector<PaddedCount> counts(probe_threads);
    parallel::for_chunks(
        input.probe_size,
        probe_threads,
        [&](std::size_t thread, std::size_t begin, std::size_t end) {
            std::size_t local_count = 0;
            for (auto index = begin; index < end; ++index) {
                local_count += table.count(input.probe_keys[index]);
            }
            counts[thread].value = local_count;
        });

    std::size_t total = 0;
    for (const auto& count : counts) {
        total += count.value;
    }
    const auto seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return JoinMetrics{total, seconds};
}

template <bool UseBloomFilter>
MaterializedJoin run_materialize(const Inputs& input) {
    const auto start = std::chrono::steady_clock::now();
    const auto threads = thread_count();
    const UnchainedHashTable<UseBloomFilter> table(
        input.build_keys, input.build_values, input.build_size, threads);

    const auto probe_threads = parallel::effective_thread_count(
        threads, input.probe_size, kProbeGrain);
    std::vector<std::vector<OutputTuple>> local_results(probe_threads);
    parallel::for_chunks(
        input.probe_size,
        probe_threads,
        [&](std::size_t thread, std::size_t begin, std::size_t end) {
            auto& output = local_results[thread];
            output.reserve(end - begin);
            for (auto index = begin; index < end; ++index) {
                const auto key = input.probe_keys[index];
                table.find(key, [&](const BuildTuple& build) {
                    output.push_back(OutputTuple{key, build.value});
                });
            }
        });

    std::vector<std::size_t> offsets(probe_threads + 1, 0);
    for (std::size_t thread = 0; thread < probe_threads; ++thread) {
        offsets[thread + 1] = offsets[thread] + local_results[thread].size();
    }

    std::vector<OutputTuple> output(offsets.back());
    parallel::for_tasks(probe_threads, probe_threads, [&](std::size_t thread) {
        std::copy(
            local_results[thread].begin(),
            local_results[thread].end(),
            output.begin() + static_cast<std::ptrdiff_t>(offsets[thread]));
    });
    const auto seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return MaterializedJoin{std::move(output), seconds};
}

template <bool UseBloomFilter>
py::tuple join_count(
    const UInt64Array& build_keys,
    const UInt64Array& build_values,
    const UInt64Array& probe_keys) {
    const auto input = validate_inputs(build_keys, build_values, probe_keys);
    JoinMetrics result{};
    {
        py::gil_scoped_release release;
        result = run_count<UseBloomFilter>(input);
    }
    return py::make_tuple(py::int_(result.count), result.seconds);
}

template <bool UseBloomFilter>
py::tuple join_materialize(
    const UInt64Array& build_keys,
    const UInt64Array& build_values,
    const UInt64Array& probe_keys) {
    const auto input = validate_inputs(build_keys, build_values, probe_keys);
    MaterializedJoin result;
    {
        py::gil_scoped_release release;
        result = run_materialize<UseBloomFilter>(input);
    }
    return py::make_tuple(py::int_(result.tuples.size()), result.seconds);
}

template <bool UseBloomFilter>
py::tuple join_pairs(
    const UInt64Array& build_keys,
    const UInt64Array& build_values,
    const UInt64Array& probe_keys) {
    const auto input = validate_inputs(build_keys, build_values, probe_keys);
    MaterializedJoin result;
    {
        py::gil_scoped_release release;
        result = run_materialize<UseBloomFilter>(input);
    }

    py::array_t<std::uint64_t> keys(result.tuples.size());
    py::array_t<std::uint64_t> values(result.tuples.size());
    auto* output_keys = keys.mutable_data();
    auto* output_values = values.mutable_data();
    for (std::size_t index = 0; index < result.tuples.size(); ++index) {
        output_keys[index] = result.tuples[index].key;
        output_values[index] = result.tuples[index].value;
    }
    return py::make_tuple(std::move(keys), std::move(values), result.seconds);
}

}  // namespace
}  // namespace flash_join

PYBIND11_MODULE(flash_join, module) {
    using namespace flash_join;

    module.doc() = "Parallel CedarDB-style unchained hash joins for uint64 NumPy arrays.";
    module.attr("implementation") = "cedardb_unchained";

    module.def(
        "join_count",
        &join_count<true>,
        "Count all join matches, including duplicate build-side keys.",
        py::arg("build_keys"), py::arg("build_values"), py::arg("probe_keys"));
    module.def(
        "join",
        &join_materialize<true>,
        "Materialize the join internally and return (row_count, core_seconds).",
        py::arg("build_keys"), py::arg("build_values"), py::arg("probe_keys"));
    module.def(
        "join_pairs",
        &join_pairs<true>,
        "Return (joined_keys, joined_values, core_seconds).",
        py::arg("build_keys"), py::arg("build_values"), py::arg("probe_keys"));
    module.def(
        "join_count_no_bloom",
        &join_count<false>,
        "Count all join matches without the per-slot Bloom filter.",
        py::arg("build_keys"), py::arg("build_values"), py::arg("probe_keys"));
    module.def(
        "join_no_bloom",
        &join_materialize<false>,
        "Materialize internally without the per-slot Bloom filter.",
        py::arg("build_keys"), py::arg("build_values"), py::arg("probe_keys"));
    module.def(
        "set_num_threads",
        &set_num_threads,
        "Set worker count; zero restores hardware-concurrency auto-detection.",
        py::arg("threads"));
    module.def("get_num_threads", &thread_count, "Return the effective worker count.");
}
