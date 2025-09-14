import sys
import os
import argparse
import time
import numpy as np
import pandas as pd

# --- Module Import and Initialization ---
# Import flash_join FIRST to avoid potential memory allocator conflicts with other libraries.
try:
    import flash_join
    print("Successfully imported flash_join module.")
    # Initialize the library if it has a global setup function.
    flash_join.initialize()
except ImportError as e:
    print(f"CRITICAL: Could not import 'flash_join' module: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL: Error during flash_join initialization: {e}", file=sys.stderr)
    sys.exit(1)

# Import other libraries after flash_join
try:
    import duckdb
    print("Successfully imported duckdb module.")
except ImportError as e:
    print(f"CRITICAL: Could not import 'duckdb' module: {e}", file=sys.stderr)
    sys.exit(1)

print("All modules imported and initialized successfully!")
print("-" * 60)


# --- Helper Function for Benchmarking ---
def run_benchmark(label, task_name, threads, func):
    """
    A helper function to run a benchmark, time it, and print results in two formats.
    - func: A no-argument lambda function that executes the actual benchmark and returns a result.
    """
    print(f"\n🚀 Starting benchmark for {label} ({task_name})...")
    start_time = time.perf_counter()
    result = func()
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    # User-friendly print
    print(f"   - Finished in: {duration:.4f} seconds")
    if isinstance(result, (int, np.integer)):
        print(f"   - Result (count): {result:,}")
        result_str = str(result)
    elif result is not None:
        # For full join results, we print the length to avoid flooding the console
        print(f"   - Result (length): {len(result):,}")
        result_str = str(len(result))
    else:
        print(f"   - Result: None")
        result_str = "N/A"
        
    # Standardized output (CSV format for easy parsing by benchmark tools)
    # This format is compatible with the db-benchmark framework.
    print(f"RESULT,Library={label},Task={task_name},Threads={threads},Time={duration:.4f},Result={result_str}")
    return duration


# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Run Join benchmarks for flash_join and DuckDB using h2oai/db-benchmark data.")
    parser.add_argument("--path", type=str, required=True, help="Path to the data directory (e.g., './_data/join/N_1e8_K_10_P_5e8')")
    parser.add_argument("--threads", type=int, default=os.cpu_count(), help="Number of threads to use for DuckDB.")
    args = parser.parse_args()

    # --- 1. Load Data ---
    print(f"Loading data from: {args.path}")
    build_file = os.path.join(args.path, 'build.parquet')
    probe_file = os.path.join(args.path, 'probe.parquet')

    if not os.path.exists(build_file) or not os.path.exists(probe_file):
        print(f"Error: Parquet files not found in '{args.path}'.", file=sys.stderr)
        print("Please generate the data first using `./run.sh datagen join` after configuring `run.conf`.", file=sys.stderr)
        sys.exit(1)

    # Load using pandas for easy conversion to NumPy arrays
    build_df = pd.read_parquet(build_file)
    probe_df = pd.read_parquet(probe_file)
    
    print(f"Build table size: {len(build_df):,} records")
    print(f"Probe table size: {len(probe_df):,} records")

    # --- CRITICAL: Data Type Conversion ---
    # As noted in your original script, flash_join expects uint32.
    # The benchmark framework generates Int64 by default, so this conversion is essential.
    print("Converting data types to uint32 for flash_join compatibility...")
    build_keys = build_df['key'].astype(np.uint32).to_numpy()
    build_values = build_df['value'].astype(np.uint32).to_numpy()
    probe_keys = probe_df['key'].astype(np.uint32).to_numpy()
    print("Data loading and preparation complete.")
    print("=" * 60)

    # --- 2. Run flash_join Benchmarks ---
    # Note: The 'threads' parameter is passed for reporting consistency.
    # The actual parallelism of flash_join is determined by its internal implementation.
    run_benchmark("flash_join", "join", args.threads, 
                  lambda: flash_join.hash_join(build_keys, build_values, probe_keys))
    
    run_benchmark("flash_join", "join_count", args.threads, 
                  lambda: flash_join.hash_join_count(build_keys, build_values, probe_keys))

    run_benchmark("flash_join_radix", "join", args.threads, 
                  lambda: flash_join.hash_join_radix(build_keys, build_values, probe_keys))
                  
    run_benchmark("flash_join_radix", "join_count", args.threads, 
                  lambda: flash_join.hash_join_count_radix(build_keys, build_values, probe_keys))

    run_benchmark("flash_join_scalar", "join", args.threads, 
                  lambda: flash_join.hash_join_scalar(build_keys, build_values, probe_keys))

    run_benchmark("flash_join_scalar", "join_count", args.threads, 
                  lambda: flash_join.hash_join_count_scalar(build_keys, build_values, probe_keys))


    # --- 3. Run DuckDB Benchmarks ---
    print("=" * 60)
    print(f"Configuring DuckDB to use {args.threads} threads...")
    con = duckdb.connect(database=':memory:')
    con.execute(f"PRAGMA THREADS={args.threads}")
    con.execute(f"PRAGMA-SET-VERIFY-EXTERNAL=true") # Recommended when reading from external files

    # DuckDB - Join Count
    # This is the standard db-benchmark way: read directly from Parquet files.
    query_count = f"""
    SELECT COUNT(*) 
    FROM read_parquet('{probe_file}') AS p 
    JOIN read_parquet('{build_file}') AS b ON p.key = b.key;
    """
    run_benchmark("duckdb", "join_count", args.threads, 
                  lambda: con.execute(query_count).fetchone()[0])

    # DuckDB - Join and Materialize Result
    # This simulates a scenario where all matching rows need to be returned.
    query_materialize = f"""
    SELECT b.value
    FROM read_parquet('{probe_file}') AS p 
    JOIN read_parquet('{build_file}') AS b ON p.key = b.key;
    """
    # We get the length of the result, not the full result set itself, to avoid memory issues and slow printing.
    run_benchmark("duckdb", "join", args.threads, 
                  lambda: len(con.execute(query_materialize).fetchall()))
                  
    con.close()
    
    print("\n" + "="*60)
    print("Benchmark run finished.")
    print("="*60)


if __name__ == "__main__":
    main()