# Flash Hash Join

Flash Hash Join is a high-performance, C++ implementation of the hash join algorithm. On our benchmarks, it runs up to **2x faster** than DuckDB's hash join implementation.

This project is open-source and released under the MIT license.

## Inspiration

This implementation was inspired by the techniques described in the blog post "[Simple, Efficient Hash Tables](https://cedardb.com/blog/simple_efficient_hash_tables/)" from CedarDB. The article provides excellent insights into building fast hash tables, which form the core of a hash join algorithm.

## Implementation Details

Our goal was to build the fastest hash join possible by focusing on a few key principles:

*   **Linear Probing:** We use linear probing for collision resolution, which is cache-friendly and avoids the pointer-chasing overhead of separate chaining. This is one of the core ideas from the inspirational blog post that proved highly effective.

*   **Optimized Memory Layout:** The hash table is designed for data locality, ensuring that the CPU's cache is used as efficiently as possible during both the build and probe phases.

*   **Exclusion of Bloom Filters:** The original blog post suggested using bloom filters to pre-filter tuples during the probing phase. However, through extensive benchmarking, we discovered that for our specific workloads and datasets, the overhead of building and checking the bloom filter actually led to a decrease in performance. Therefore, we decided to omit this component to achieve maximum speed.

## Performance

The primary motivation for this project was raw performance. The included benchmark (`benchmark.py`) compares this implementation against DuckDB.

In our tests, Flash Hash Join consistently outperforms the competition, in some cases achieving a **2x speedup**. We encourage you to run the benchmarks yourself to see the results.

## Getting Started

### Prerequisites
- C++ compiler (g++ or clang)
- CMake
- Python 3

### Building
```bash
# Clone the repository
git clone --recurse-submodules https://github.com/conanhujinming/flash_hash_join.git
cd flash_hash_join

# Build the project
mkdir build
cd build
cmake ..
make
```

## Running the Benchmark

To replicate the performance tests, you can run the provided Python benchmark script:

```bash
# From the project root directory
pip install -r requirements.txt
python setup.py build_ext --inplace
python benchmark.py
```

## License

This project is licensed under the [MIT License](LICENSE).