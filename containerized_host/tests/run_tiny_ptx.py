#!/usr/bin/env python3

import subprocess
import time
import os
import sys
from datetime import datetime

def main():
    # Record start time
    start_time = time.time()
    print(f"Python wrapper started at: {datetime.now().strftime('%H:%M:%S.%f')}")
    
    # Sleep for 1 second
    print("Sleeping for 1 second...")
    time.sleep(1)
    
    # Check if tiny_ptx exists, if not, compile it
    tiny_ptx_path = "./tiny_ptx"
    if not os.path.exists(tiny_ptx_path):
        print("Compiling tiny_ptx.c...")
        compile_result = subprocess.run(
            ["nvcc", "tiny_ptx.c", "-o", "tiny_ptx", "-lcuda"],
            capture_output=True,
            text=True
        )
        if compile_result.returncode != 0:
            print(f"Compilation failed: {compile_result.stderr}")
            sys.exit(1)
        print("Compilation successful")
    
    # Execute the tiny_ptx program
    print("\nExecuting tiny_ptx program...")
    exec_start = time.time()
    proc = subprocess.run([tiny_ptx_path], capture_output=True, text=True)
    exec_end = time.time()
    
    # Print the output from tiny_ptx
    print("\n--- tiny_ptx output ---")
    print(proc.stdout)
    if proc.stderr:
        print("--- Error output ---")
        print(proc.stderr)
    
    # Calculate and log overhead times
    exec_time = exec_end - exec_start
    total_time = time.time() - start_time
    python_overhead = total_time - exec_time - 1.0  # Subtract the 1s sleep
    
    print("\n--- Python Wrapper Timing ---")
    print(f"Sleep time: 1.000 seconds")
    print(f"tiny_ptx execution time: {exec_time:.3f} seconds")
    print(f"Python overhead: {python_overhead:.3f} seconds")
    print(f"Total time: {total_time:.3f} seconds")

if __name__ == "__main__":
    main()