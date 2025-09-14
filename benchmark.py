import sys

print("Importing flash_join and duckdb modules...")

# --- Import flash_join FIRST to avoid memory allocator conflicts ---
try:
    import flash_join
    print("Successfully imported flash_join module")
    flash_join.initialize()
    print("Initialization complete.")
except ImportError as e:
    print(f"Error: Could not import 'flash_join' module: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error during flash_join initialization: {e}")
    sys.exit(1)

# --- Import other modules AFTER flash_join ---
import numpy as np
import time
import pandas as pd
import os

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as e:
    print(f"Warning: Could not import pyarrow: {e}")
    pa = None
    pq = None

try:
    import duckdb
    print("Successfully imported duckdb module")
except ImportError as e:
    print(f"Error: Could not import 'duckdb' module: {e}")
    sys.exit(1)

print("Initialization complete.")
print("All modules imported successfully!")

# --- (generate_data function remains the same) ---
def generate_data(build_size, probe_size):
    print("="*40)
    print("Generating test data...")
    key_cardinality = build_size // 10
    # Fix data type - use uint32 instead of uint64 to match C++ extension
    build_keys = np.random.randint(0, key_cardinality, size=build_size)
    build_values = np.arange(build_size)
    probe_keys = np.random.randint(0, key_cardinality * 2, size=probe_size)
    print(f"Build table size: {len(build_keys):,} records")
    print(f"Probe table size: {len(probe_keys):,} records")
    print("Data generation complete.")
    print("="*40)
    return build_keys, build_values, probe_keys

if __name__ == "__main__":
    # --- Benchmark Parameters ---
    BUILD_SIZE = 10_000_000  # 100 Million
    PROBE_SIZE = 50_000_000  # 500 Million

    build_keys, build_values, probe_keys = generate_data(BUILD_SIZE, PROBE_SIZE)
    
    # Prepare DataFrames once
    print("Preparing Pandas DataFrames...")
    build_df = pd.DataFrame({'key': build_keys, 'value': build_values})
    probe_df = pd.DataFrame({'key': probe_keys})
    print("Preparation complete.")

    # --- 1. Benchmark our C++ (flash_join) ---
    print("\n🚀 Starting benchmark for C++ (flash_join) [End-to-End]...")

    print("Testing flash_join.hash_join...")
    start_time_cpp = time.perf_counter()
    try:
        cpp_result = flash_join.hash_join(build_keys, build_values, probe_keys)
        end_time_cpp = time.perf_counter()
        duration_cpp = end_time_cpp - start_time_cpp
        print(f"C++ (flash_join) finished in: {duration_cpp:.4f} seconds")
        print(f"Result length: {len(cpp_result) if cpp_result is not None else 'None'}")
    except Exception as e:
        print(f"Error in flash_join.hash_join: {e}")
        sys.exit(1)

    print("Testing flash_join.hash_join_count...")
    start_time_cpp = time.perf_counter()
    try:
        cpp_count = flash_join.hash_join_count(build_keys, build_values, probe_keys)
        end_time_cpp = time.perf_counter()
        duration_cpp = end_time_cpp - start_time_cpp
        print(f"C++ (flash_join_count) finished in: {duration_cpp:.4f} seconds")
        print(f"Result count: {cpp_count}")
    except Exception as e:
        print(f"Error in flash_join.hash_join_count: {e}")
        sys.exit(1)

    start_time_cpp = time.perf_counter()
    cpp_count = flash_join.hash_join_scalar(build_keys, build_values, probe_keys)
    end_time_cpp = time.perf_counter()
    duration_cpp = end_time_cpp - start_time_cpp
    print(f"C++ (flash_join_scalar) finished in: {duration_cpp:.4f} seconds")

    print("\n🚀 Starting benchmark for C++ (flash_join_scalar) [End-to-End]...")
    start_time_cpp = time.perf_counter()
    cpp_count = flash_join.hash_join_count_scalar(build_keys, build_values, probe_keys)
    end_time_cpp = time.perf_counter()
    duration_cpp_count = end_time_cpp - start_time_cpp
    print(f"C++ (flash_join_count_scalar) finished in: {duration_cpp_count:.4f} seconds")

    build_file = 'build.parquet'
    probe_file = 'probe.parquet'
    pq.write_table(pa.Table.from_pandas(build_df), build_file)
    pq.write_table(pa.Table.from_pandas(probe_df), probe_file)

    # --- 2. Benchmark Pandas (pd.merge) ---
    # print("\n🐢 Starting benchmark for Pandas (pd.merge)...")
    # start_time_pandas = time.perf_counter()
    # pandas_result = pd.merge(probe_df, build_df, on='key', how='inner', sort=False)
    # end_time_pandas = time.perf_counter()
    # duration_pandas = end_time_pandas - start_time_pandas
    # print(f"Pandas (pd.merge) finished in: {duration_pandas:.4f} seconds")

    # --- 3. Benchmark DuckDB ---
    print("\n🏆 Starting benchmark for DuckDB [Ingest + Join]...")
    
    con = duckdb.connect(database=':memory:')
    cpu_count = os.cpu_count()
    if cpu_count: 
        con.execute(f"PRAGMA THREADS={cpu_count}")
        print(f"(DuckDB is configured to use {cpu_count} threads)")

    # --- DuckDB Ingest Phase ---
    print("  - Ingesting data into DuckDB native tables...")
    start_ingest = time.perf_counter()
    # This creates native, columnar tables inside DuckDB's memory space
    con.execute("CREATE TABLE build_native AS SELECT * FROM build_df;")
    con.execute("CREATE TABLE probe_native AS SELECT * FROM probe_df;")
    end_ingest = time.perf_counter()
    duration_ingest = end_ingest - start_ingest
    print(f"  - DuckDB ingest finished in: {duration_ingest:.4f} seconds")
    # This is where you will see the ~200% CPU bottleneck

    # --- DuckDB Join Phase ---
    print("  - Joining native tables with query SELECT count(*) FROM probe_native AS p JOIN build_native AS b ON p.key = b.key")
    start_join = time.perf_counter()
    # This query now runs entirely on DuckDB's internal, parallel-friendly format
    query = "SELECT count(*) FROM probe_native AS p JOIN build_native AS b ON p.key = b.key"
    duckdb_result = con.execute(query).arrow()
    end_join = time.perf_counter()
    duration_join_count = end_join - start_join
    print(f"  - DuckDB native join count finished in: {duration_join_count:.4f} seconds")
    
    duration_duckdb_total_count = duration_ingest + duration_join_count
    print(f"DuckDB (Total Ingest + Join count) finished in: {duration_duckdb_total_count:.4f} seconds, count: {duckdb_result}")


    start_join = time.perf_counter()
    # This query now runs entirely on DuckDB's internal, parallel-friendly format
    query = '''
        CREATE OR REPLACE TEMPORARY TABLE temp_join_result AS
        SELECT p.key, b.value FROM probe_native AS p JOIN build_native AS b ON p.key = b.key'''
    duckdb_result = con.execute(query).arrow()
    end_join = time.perf_counter()
    duration_join = end_join - start_join
    print(f"  - DuckDB native join finished in: {duration_join_count:.4f} seconds")
    
    duration_duckdb_total = duration_ingest + duration_join
    print(f"DuckDB (Total Ingest + Join) finished in: {duration_duckdb_total:.4f} seconds, count: {len(duckdb_result)}")
    
    # --- Results Summary ---
    print("\n" + "="*40)
    print("📊 Final Performance Summary (End-to-End)")
    print(f"Our C++ (Build + Probe) count:  {duration_cpp_count:.4f} seconds")
    print(f"Our C++ (Build + Probe):  {duration_cpp:.4f} seconds")
    print(f"DuckDB (Ingest + Join count):   {duration_duckdb_total_count:.4f} seconds")
    print(f"  - DuckDB Ingest phase:    ({duration_ingest:.4f}s)")
    print(f"  - DuckDB Join count phase:      ({duration_join_count:.4f}s)")
    print(f"DuckDB (Ingest + Join):   {duration_duckdb_total:.4f} seconds")
    print(f"  - DuckDB Join phase:      ({duration_join:.4f}s)")
    # print(f"Pandas (pd.merge):        {duration_pandas:.4f} seconds")
    print("-" * 40)
    print("="*40)