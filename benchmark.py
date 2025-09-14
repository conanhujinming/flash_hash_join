import sys
import os
import argparse
import time
import numpy as np
import pandas as pd
import re
import glob
from collections import defaultdict
import pandas.api.types as ptypes

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

# Import and check for plotting libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("Plotting libraries (matplotlib, seaborn) loaded successfully.")
except ImportError:
    print("\nWARNING: Plotting libraries not found. Results will not be visualized.")
    print("Please run 'pip install matplotlib seaborn' to enable plotting.")
    plt = None
    sns = None

print("-" * 80)


# --- Helper Function for Benchmarking ---
def run_benchmark(label, task_name, threads, func):
    """
    A helper function to run a benchmark, time it, and print results in a standard format.
    - func: A no-argument lambda function that executes the actual benchmark and returns a result.
    """
    print(f"\n  🚀 Starting benchmark for {label} ({task_name})...")
    start_time = time.perf_counter()
    result = func()
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    # User-friendly print
    print(f"     - Finished in: {duration:.4f} seconds")
    if isinstance(result, (int, np.integer)):
        print(f"     - Result (count): {result:,}")
        result_str = str(result)
    elif result is not None:
        print(f"     - Result (length): {len(result):,}")
        result_str = str(len(result))
    else:
        print(f"     - Result: None")
        result_str = "N/A"
        
    # Standardized output for easy parsing
    print(f"    RESULT,Library={label},Task={task_name},Threads={threads},Time={duration:.4f},Result={result_str}")
    return duration

# --- Plotting Function ---
def plot_results(results_df, task_name, output_filename="benchmark_results.png"):
    """
    Generates and saves a bar chart of the benchmark results for a specific task.
    """
    if plt is None or sns is None:
        return # Skip plotting if libraries are not available

    # Filter data for the specific task (e.g., 'join_count')
    task_df = results_df[results_df['task'] == task_name].copy()
    
    if task_df.empty:
        print(f"\nNo data to plot for task: {task_name}")
        return

    # Sort implementations for consistent color ordering in the plot
    # A custom order can make the plot more readable
    impl_order = sorted(task_df['implementation'].unique())
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 9))
    
    sns.barplot(
        data=task_df,
        x='case',
        y='time',
        hue='implementation',
        hue_order=impl_order,
        ax=ax
    )
    
    ax.set_title(f'Benchmark Performance: {task_name.replace("_", " ").title()}', fontsize=20, pad=20)
    ax.set_xlabel('Benchmark Case (Dataset-Query)', fontsize=14)
    ax.set_ylabel('Time (seconds)', fontsize=14)
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(title='Implementation', fontsize=12)
    
    # Add text labels on top of each bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=9, rotation=90, padding=5)

    ax.margins(y=0.2) # Add some space at the top for the labels
    fig.tight_layout()
    
    try:
        fig.savefig(output_filename)
        print(f"\n📊 Plot saved to '{output_filename}'")
    except Exception as e:
        print(f"\nError saving plot: {e}")
        
    plt.close(fig)

# --- Main Logic ---

def discover_benchmark_suites(data_dir):
    """
    Automatically discovers benchmark suites from the data directory.
    """
    print(f"Scanning for benchmark suites in: {data_dir}")
    all_csv_files = glob.glob(os.path.join(data_dir, "J1_*.csv"))
    
    groups = defaultdict(list)
    for f in all_csv_files:
        basename = os.path.basename(f)
        match = re.match(r"J1_(\de\d+)_", basename)
        if match:
            groups[match.group(1)].append(f)
    
    suites = []
    for group_key, file_list in groups.items():
        base_digit = group_key[0]
        paths = {
            'x': os.path.join(data_dir, f"J1_{group_key}_{group_key}_0_0.csv"),
            'small': os.path.join(data_dir, f"J1_{group_key}_{base_digit}e1_0_0.csv"),
            'medium': os.path.join(data_dir, f"J1_{group_key}_{base_digit}e4_0_0.csv"),
            'big': os.path.join(data_dir, f"J1_{group_key}_{base_digit}e7_0_0.csv"),
        }
        
        if all(os.path.exists(p) for p in paths.values()):
            suite = paths.copy()
            suite['group_name'] = group_key
            suites.append(suite)
            print(f"  - Found complete suite for data size '{group_key}'")
        else:
            print(f"  - WARNING: Incomplete file set for data size '{group_key}'. Skipping.")
            
    return suites

def main():
    parser = argparse.ArgumentParser(description="Run and visualize a fair join benchmark suite.")
    parser.add_argument("--data-dir", type=str, default='./data', help="Path to the directory containing the join benchmark CSV files.")
    parser.add_argument("--threads", type=int, default=os.cpu_count(), help="Number of threads to use for all libraries.")
    args = parser.parse_args()

    results_for_plotting = [] # List to store all benchmark results

    benchmark_suites = discover_benchmark_suites(args.data_dir)
    if not benchmark_suites:
        print("Error: No complete benchmark suites found.", file=sys.stderr)
        sys.exit(1)

    for suite in benchmark_suites:
        print("=" * 80)
        print(f"Loading data for suite '{suite['group_name']}'")
        
        tables_full = {name: pd.read_csv(path) for name, path in suite.items() if name != 'group_name'}

        benchmark_cases = [
            {"id": "Q1", "desc": "INNER JOIN with 'small' table ON id1", "left": "x", "right": "small", "key": "id1"},
            {"id": "Q2", "desc": "INNER JOIN with 'medium' table ON id2", "left": "x", "right": "medium", "key": "id2"},
            {"id": "Q4", "desc": "INNER JOIN with 'medium' table ON id5 (factor key)", "left": "x", "right": "medium", "key": "id5"},
            {"id": "Q5", "desc": "INNER JOIN with 'big' table ON id3", "left": "x", "right": "big", "key": "id3"},
        ]

        for case in benchmark_cases:
            case_id = f"{suite['group_name']}-{case['id']}"
            print("-" * 80)
            print(f"▶️  Running Benchmark Case {case_id}: {case['desc']}")
            print("-" * 80)

            build_df_full = tables_full[case['right']]
            probe_df_full = tables_full[case['left']]
            join_key, value_col = case['key'], 'v2' 

            if join_key not in build_df_full.columns or join_key not in probe_df_full.columns or value_col not in build_df_full.columns:
                print(f"  - WARNING: Required columns not found. Skipping case.")
                continue

            is_key_numeric = ptypes.is_numeric_dtype(build_df_full[join_key]) and ptypes.is_numeric_dtype(probe_df_full[join_key])
            is_value_numeric = ptypes.is_numeric_dtype(build_df_full[value_col])
            
            if not (is_key_numeric and is_value_numeric):
                print(f"  - WARNING: Key or value column is not purely numeric. Skipping case.")
                continue
            
            build_df = build_df_full[[join_key, value_col]].rename(columns={join_key: 'key', value_col: 'value'})
            probe_df = probe_df_full[[join_key]].rename(columns={join_key: 'key'})
            
            build_df['key'], build_df['value'], probe_df['key'] = \
                build_df['key'].astype(np.uint64), build_df['value'].astype(np.uint64), probe_df['key'].astype(np.uint64)
            
            build_keys, build_values, probe_keys = \
                build_df['key'].to_numpy(), build_df['value'].to_numpy(), probe_df['key'].to_numpy()
            
            # --- flash_join Benchmarks ---
            impl_map = {
                'flash_join': (flash_join.hash_join_count, flash_join.hash_join),
                'flash_join_radix': (flash_join.hash_join_count_radix, flash_join.hash_join_radix),
                'flash_join_scalar': (flash_join.hash_join_count_scalar, flash_join.hash_join_scalar),
            }
            for label, (count_func, mat_func) in impl_map.items():
                duration_count = run_benchmark(label, "join_count", args.threads, lambda: count_func(build_keys, build_values, probe_keys))
                results_for_plotting.append({'case': case_id, 'implementation': label, 'task': 'join_count', 'time': duration_count})
                
                duration_mat = run_benchmark(label, "join_materialize", args.threads, lambda: mat_func(build_keys, build_values, probe_keys))
                results_for_plotting.append({'case': case_id, 'implementation': label, 'task': 'join_materialize', 'time': duration_mat})

            # --- DuckDB Benchmarks ---
            # ... (DuckDB logic remains the same) ...
            con = duckdb.connect(database=':memory:')
            con.execute(f"PRAGMA THREADS={args.threads}")

            start_ingest = time.perf_counter()
            con.execute("CREATE TABLE build_native AS SELECT * FROM build_df;")
            con.execute("CREATE TABLE probe_native AS SELECT * FROM probe_df;")
            duration_ingest = time.perf_counter() - start_ingest
            
            duration_join_count = run_benchmark("duckdb", "join_count", args.threads, 
                                                lambda: con.execute("SELECT count(*) FROM build_native b JOIN probe_native p ON b.key = p.key;").fetchone()[0])
            
            def duckdb_materialize_and_count():
                con.execute("CREATE OR REPLACE TEMPORARY TABLE temp AS SELECT p.key, b.value FROM build_native b JOIN probe_native p ON b.key = p.key;")
                return con.execute("SELECT count(*) FROM temp").fetchone()[0]
            duration_join_materialize = run_benchmark("duckdb", "join_materialize", args.threads, duckdb_materialize_and_count)
            con.close()

            # Add DuckDB results to plotting data
            results_for_plotting.append({'case': case_id, 'implementation': 'duckdb (Join Only)', 'task': 'join_count', 'time': duration_join_count})
            results_for_plotting.append({'case': case_id, 'implementation': 'duckdb (Ingest + Join)', 'task': 'join_count', 'time': duration_ingest + duration_join_count})
            results_for_plotting.append({'case': case_id, 'implementation': 'duckdb (Join Only)', 'task': 'join_materialize', 'time': duration_join_materialize})
            results_for_plotting.append({'case': case_id, 'implementation': 'duckdb (Ingest + Join)', 'task': 'join_materialize', 'time': duration_ingest + duration_join_materialize})


    print("\n" + "="*80)
    print("All benchmark cases finished.")
    print("="*80)

    # --- Final Step: Generate Plots ---
    if results_for_plotting:
        results_df = pd.DataFrame(results_for_plotting)
        plot_results(results_df, 'join_count', 'benchmark_join_count.png')
        plot_results(results_df, 'join_materialize', 'benchmark_join_materialize.png')

if __name__ == "__main__":
    main()