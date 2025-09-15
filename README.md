好的，当然！这是一个根据你的新代码更新后的README。

我将重点突出新增的`adaptive_join`功能，并将其定位为推荐给大多数用户的首选接口。同时，我将原有的特定算法实现重新组织为“核心策略”，这样既能保留其技术深度，又能使用户的入门路径更清晰。

---

# Flash Join

Flash Join is a high-performance, parallel hash join library implemented in C++ with a Python interface. It is designed to deliver extreme speed for data analysis tasks. By leveraging modern CPU architectures and an intelligent adaptive execution engine, Flash Join demonstrates superior performance, often outperforming the industry-leading in-memory database DuckDB.

This project is open-source and released under the MIT license.

## New: Adaptive Join - The Best of Both Worlds

The latest version of Flash Join introduces a powerful **adaptive join strategy**, which is now the recommended API for most use cases.

-   **Core Idea**: Different join algorithms excel under different conditions. A classic hash join has low overhead and is perfect for smaller datasets, while a radix join's partitioning strategy is more cache-efficient and scalable for larger datasets. Manually choosing between them can be complex.
-   **How it Works**: The `adaptive_join` function automatically analyzes the input data (specifically, the size of the build table) and selects the optimal underlying algorithm at runtime.
    -   For **small build tables** (e.g., < 1 million rows), it uses a highly optimized **Scalar Hash Join**.
    -   For **large build tables**, it automatically switches to a cache-friendly **Radix Join**.
-   **Benefit**: You get the best possible performance without needing to be an expert in join algorithms. It's intelligent, hands-off performance optimization.

### Quick Example

Using the new adaptive API is simple:

```python
import flash_join
import numpy as np

# Assume build_keys, build_values, and probe_keys are NumPy arrays
# ...

# Let Flash Join choose the best strategy automatically
num_matches, time_taken = flash_join.adaptive_join(
    build_keys, 
    build_values, 
    probe_keys
)

print(f"Found {num_matches} matches in {time_taken:.4f} seconds.")
```

## Underlying Core Strategies

Flash Join's adaptive engine chooses between two powerful, battle-tested join strategies. For expert users or benchmarking purposes, these can still be called directly.

*   **Scalar Hash Join (`flash_join`)**
    *   **Core Idea**: Builds a single, global hash table using cache-friendly **Linear Probing** for collision resolution. This avoids the pointer-chasing overhead of separate chaining and is designed for maximum data locality, which is extremely efficient when the entire hash table fits comfortably in the CPU caches.
    *   **Inspiration**: This approach is heavily inspired by the techniques described in the CedarDB blog post "[Simple, Efficient Hash Tables](https://cedardb.com/blog/simple_efficient_hash_tables/)".
    *   **Best for**: Smaller build tables where the overhead of partitioning would outweigh the benefits.

*   **Radix Join (`flash_join_radix`)**
    *   **Core Idea**: A "no-hash-table" join algorithm that works by partitioning both the build and probe tables into many small, cache-aligned chunks based on the radix (binary representation) of the join keys. Each pair of corresponding partitions is then joined independently.
    *   **Best for**: Large build tables. This divide-and-conquer approach ensures that each sub-problem is small enough to fit within the CPU's L1/L2 caches, drastically reducing cache misses and enabling massive parallelism.

## Available Optimizations

*   **Bloom Filters (`_bloom` suffix)**
    *   **Core Idea**: Builds a compact Bloom filter alongside the hash table. During the probe phase, this filter is checked first to rapidly prune probe keys that are guaranteed not to have a match. This significantly reduces memory accesses to the main hash table.
    *   **Availability**: This optimization can be applied to both the adaptive and the explicit APIs (e.g., `adaptive_join_bloom`, `hash_join_bloom`).
    *   **Best for**: Joins where a large percentage of probe keys do not have a match in the build table (low join selectivity).

## Performance

The core objective of this project is raw performance. The included benchmark script (`benchmark.py`) compares our underlying implementations against DuckDB on two common join tasks:

1.  **Join Count**: Calculates the total number of resulting rows from the join without materializing them. This primarily measures the performance of the core build and probe phases.
2.  **Join Materialize**: Creates and returns the complete set of joined rows. This is a more realistic end-to-end test that reflects real-world application workloads.

The benchmark automatically generates performance comparison charts.

## Getting Started

### Prerequisites
- A C++ compiler (g++ or clang with C++17 support)
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
This command compiles the C++ source code into a Python extension module that can be imported directly into your Python scripts.
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