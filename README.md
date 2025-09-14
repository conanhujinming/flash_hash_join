# Flash Join

Flash Join is a high-performance, parallel join algorithm library implemented in C++. It is designed to deliver extreme speed for data analysis tasks. By leveraging the features of modern CPU architectures, our implementation demonstrates superior performance, outperforming the industry-leading in-memory database DuckDB by more than **2x** in some scenarios.

This project now includes multiple join algorithm implementations, allowing for optimal performance across different data distributions and query patterns.

This project is open-source and released under the MIT license.

## Algorithm Implementations

In our pursuit of maximum performance, we have implemented and benchmarked several join algorithms. The `benchmark.py` script provides a fair comparison of all the following implementations:

*   **`flash_join` (Classic Hash Join)**
    *   **Core Idea**: Uses cache-friendly **Linear Probing** for collision resolution, avoiding the pointer-chasing overhead of separate chaining. The hash table is designed for data locality to maximize CPU cache efficiency during both the build and probe phases.
    *   **Inspiration**: This approach is heavily inspired by the techniques described in the CedarDB blog post "[Simple, Efficient Hash Tables](https://cedardb.com/blog/simple_efficient_hash_tables/)".

*   **`flash_join_radix` (Radix Join)**
    *   **Core Idea**: A non-hashing join algorithm that works by partitioning keys based on their binary representation. Through multiple partitioning passes, it physically clusters tuples with the same key, allowing for a highly efficient final join phase.
    *   **Best for**: Scenarios where join keys are integers.

*   **`flash_join_bloom` (Hash Join with Bloom Filter)**
    *   **Core Idea**: Builds a Bloom filter alongside the hash table. During the probe phase, the Bloom filter is used to quickly prune probe keys that are guaranteed not to exist in the build table. This reduces the number of accesses to the main hash table and minimizes cache misses.
    *   **Best for**: Cases where a large percentage of keys in the probe side do not have a match in the build side.

*   **`flash_join_radix_bloom` (Radix Join with Bloom Filter)**
    *   **Core Idea**: Combines the advantages of both approaches by using a Bloom filter to pre-filter the probe table before running the radix join algorithm.

## Performance

The core objective of this project is raw performance. The included benchmark script (`benchmark.py`) compares our implementations against DuckDB on two common join tasks:

1.  **Join Count**: Calculates the total number of resulting rows from the join without materializing them. This primarily measures the performance of the core join algorithm (build and probe phases).
2.  **Join Materialize**: Creates and returns the complete set of joined rows. This is a more realistic end-to-end test that reflects real-world application workloads.

The benchmark automatically generates performance comparison charts.
## Getting Started

### Prerequisites
- A C++ compiler (g++ or clang)
- CMake (>= 3.10)
- Python (>= 3.7)
- R environment
  - Requires the `data.table` and `arrow` packages.

### Setup and Execution

#### 1. Clone the Repository
```bash
# Clone the repository and all its submodules
git clone --recurse-submodules https://github.com/conanhujinming/flash_hash_join.git
cd flash_hash_join
```

#### 2. Generate Benchmark Data
The benchmark requires a specific dataset. Please follow these steps to generate it.

```bash
# First, ensure your R environment has the necessary packages.
# You can run the following command in an R console:
# install.packages(c("data.table", "arrow"))

# Next, run the data generation script.
# This will create a series of CSV files in the ./data directory.
bash ./generate-data.sh
```
**Note**: This data generation process may take several minutes to complete.

#### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Build the C++ Extension
This command compiles the C++ source code into a Python extension module (`.so` on Linux/macOS or `.pyd` on Windows) that can be imported directly into your Python scripts.
```bash
python setup.py build_ext --inplace
```

#### 5. Run the Benchmark
You are now ready to run the performance tests.

```bash
# Run the benchmark using all available CPU cores
python benchmark.py

# You can also specify the number of threads to use
python benchmark.py --threads 8
```
After the script finishes, it will generate two result images in the project's root directory:
- `benchmark_join_count.png`
- `benchmark_join_materialize.png`

## License

This project is licensed under the [MIT License](LICENSE).