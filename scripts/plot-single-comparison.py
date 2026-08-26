#!/usr/bin/env python3

import sqlite3
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import shutil
import subprocess
import os
import sys
import time
import hashlib
import threading

POLYGEIST_PATH=os.getenv("POLYGEIST_PATH", "")

# ANSI color codes
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

class Spinner:
    """A simple rotating spinner for showing process activity"""
    def __init__(self, message="Processing"):
        self.spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.message = message
        self.spinning = False
        self.thread = None
    
    def spin(self):
        """The spinning animation loop"""
        i = 0
        while self.spinning:
            char = self.spinner_chars[i % len(self.spinner_chars)]
            print(f"\r{Colors.BLUE}{char} {self.message}...{Colors.RESET}", end='', flush=True)
            time.sleep(0.1)
            i += 1
    
    def start(self):
        """Start the spinner in a separate thread"""
        self.spinning = True
        self.thread = threading.Thread(target=self.spin)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop the spinner and clear the line"""
        self.spinning = False
        if self.thread:
            self.thread.join()
        print(f"\r{' ' * (len(self.message) + 10)}\r", end='', flush=True)

def run_command(cmd, description, step_num=None, total_steps=None):
    """Run a command and handle errors with colorized output"""
    if step_num and total_steps:
        print(f"\n{Colors.BOLD}{Colors.BLUE}📋 Step {step_num}/{total_steps}: {description}{Colors.RESET}")
    else:
        print(f"\n{Colors.BOLD}{Colors.BLUE}🔄 {description}{Colors.RESET}")
    
    print(f"{Colors.CYAN}Command: {' '.join(cmd)}{Colors.RESET}")
    
    # Create and start spinner
    spinner = Spinner("Running")
    spinner.start()
    
    try:
        # Start the process
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait for process to complete
        stdout, stderr = process.communicate()
        
        # Stop spinner
        spinner.stop()
        
        if process.returncode == 0:
            print(f"{Colors.GREEN}{Colors.BOLD}✅ {description} completed successfully!{Colors.RESET}")
            return subprocess.CompletedProcess(cmd, 0, stdout, stderr)
        else:
            raise subprocess.CalledProcessError(process.returncode, cmd, stdout, stderr)
            
    except subprocess.CalledProcessError as e:
        spinner.stop()
        print(f"\n{Colors.RED}{Colors.BOLD}❌ Error running {description}{Colors.RESET}")
        print(f"{Colors.RED}Return code: {e.returncode}{Colors.RESET}")
        if e.stdout:
            print(f"{Colors.YELLOW}stdout: {e.stdout}{Colors.RESET}")
        if e.stderr:
            print(f"{Colors.RED}stderr: {e.stderr}{Colors.RESET}")
        sys.exit(1)
    except KeyboardInterrupt:
        spinner.stop()
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  Process interrupted by user{Colors.RESET}")
        sys.exit(1)

def extract_simulation_data(db_path, block_size_bytes):
    """Extract miss ratios from simulation database"""
    print(f"{Colors.CYAN}📊 Extracting simulation data from {os.path.basename(db_path)}...{Colors.RESET}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all records and compute miss ratios
    cursor.execute("SELECT d1_cache_size, d1_miss_count, total_access FROM records ORDER BY d1_cache_size")
    results = cursor.fetchall()
    
    cache_sizes = []
    miss_ratios = []
    
    for cache_size, miss_count, total_access in results:
        if total_access > 0:  # Avoid division by zero
            miss_ratio = miss_count / total_access
            # Convert cache size from bytes to number of blocks
            cache_sizes.append(cache_size // block_size_bytes)
            miss_ratios.append(miss_ratio)
    
    conn.close()
    return cache_sizes, miss_ratios

def extract_prediction_data(json_path):
    """Extract miss ratios from prediction JSON file"""
    print(f"{Colors.CYAN}📈 Extracting prediction data from {os.path.basename(json_path)}...{Colors.RESET}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    miss_ratio_curve = data.get("miss_ratio_curve", {})
    
    # Get turning points (already in number of blocks)
    turning_points = miss_ratio_curve.get("turning_points", [])
    miss_ratios = miss_ratio_curve.get("miss_ratio", [])
    miss_ratios = [min(max(mr, 0.0), 1.0) for mr in miss_ratios]  # Clamp to [0, 1]
    
    # If no turning points, generate based on miss_ratios length
    if not turning_points and miss_ratios:
        turning_points = [2**i for i in range(len(miss_ratios))]
    
    return turning_points, miss_ratios

def convert_c_to_mlir(c_file, output_mlir):
    """Convert a C file to MLIR using polygeist"""
    cgeist_path = os.path.join(POLYGEIST_PATH, "cgeist") if POLYGEIST_PATH else "cgeist"
    polygeist_opt_path = os.path.join(POLYGEIST_PATH, "polygeist-opt") if POLYGEIST_PATH else "polygeist-opt"
    
    # Check the polygeist binaries exist (as a file path, or resolvable on PATH
    # when POLYGEIST_PATH is unset and we're using bare names).
    def _resolvable(p):
        return os.path.exists(p) or shutil.which(p) is not None
    if not _resolvable(cgeist_path):
        print(f"{Colors.RED}❌ Error: cgeist not found ({cgeist_path}){Colors.RESET}")
        sys.exit(1)
    if not _resolvable(polygeist_opt_path):
        print(f"{Colors.RED}❌ Error: polygeist-opt not found ({polygeist_opt_path}){Colors.RESET}")
        sys.exit(1)
    
    print(f"{Colors.CYAN}🔄 Converting C file to MLIR: {os.path.basename(c_file)} -> {os.path.basename(output_mlir)}{Colors.RESET}")
    
    # Create the pipeline command: cgeist input.c -S -raise-scf-to-affine | polygeist-opt -strip-dlti-attributes
    # Current Polygeist has no -strip-dlti-attributes pass; canonicalize, then
    # strip the module attribute dict textually (its dlti.dl_spec breaks the
    # analyzer's newer MLIR parser).
    cmd = f'"{cgeist_path}" "{c_file}" -S -raise-scf-to-affine | "{polygeist_opt_path}" --canonicalize | sed -E "1s/^module attributes \\{{.*\\}} \\{{$/module {{/" > "{output_mlir}"'
    
    # Create and start spinner
    spinner = Spinner("Converting C to MLIR")
    spinner.start()
    
    try:
        # Use shell=True to handle the pipe
        process = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        spinner.stop()
        print(f"{Colors.GREEN}{Colors.BOLD}✅ C to MLIR conversion completed successfully!{Colors.RESET}")
        return output_mlir
    except subprocess.CalledProcessError as e:
        spinner.stop()
        print(f"\n{Colors.RED}{Colors.BOLD}❌ Error converting C to MLIR{Colors.RESET}")
        print(f"{Colors.RED}Return code: {e.returncode}{Colors.RESET}")
        if e.stdout:
            print(f"{Colors.YELLOW}stdout: {e.stdout}{Colors.RESET}")
        if e.stderr:
            print(f"{Colors.RED}stderr: {e.stderr}{Colors.RESET}")
        sys.exit(1)
    except KeyboardInterrupt:
        spinner.stop()
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  Process interrupted by user{Colors.RESET}")
        sys.exit(1)

def main():
    """Generate cache miss ratio analysis with automatic command execution"""
    # Print colorful header
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}" + "="*60 + f"{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}🚀 CACHE MISS RATIO ANALYSIS TOOL 🚀{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}" + "="*60 + f"{Colors.RESET}\n")
    
    parser = argparse.ArgumentParser(description='Cache miss ratio analysis tool')
    parser.add_argument('-i', '--input', required=True, help='Input C or MLIR file')
    parser.add_argument('-b', '--block-size', type=int, default=8, help='Cache block size in number of elements (default: 8)')
    parser.add_argument('-e', '--element-size', type=int, default=8, help='Element size in bytes (default: 8)')
    parser.add_argument('-a', '--associativity', type=int, default=4, help='Cache associativity (default: 4)')
    parser.add_argument('-c', '--cache-size', type=int, default=512, help='Cache size in bytes (default: 512)')
    parser.add_argument('-d', '--data-dir', default='/tmp', help='Directory for intermediate files (default: /tmp)')
    parser.add_argument('-o', '--output-dir', help='Output directory for plots (if not provided, plots will not be saved)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"{Colors.RED}❌ Error: Input file does not exist: {args.input}{Colors.RESET}")
        sys.exit(1)
    
    # Handle C file conversion to MLIR
    input_file = args.input
    if args.input.endswith('.c'):
        # Generate MLIR filename in data directory
        input_basename = os.path.basename(args.input)
        mlir_filename = input_basename.replace('.c', '.mlir')
        mlir_path = os.path.join(args.data_dir if hasattr(args, 'data_dir') else '/tmp', mlir_filename)
        
        # Convert C to MLIR if MLIR file doesn't exist
        if os.path.exists(mlir_path):
            print(f"\n{Colors.BOLD}{Colors.GREEN}⏭️  Skipping C to MLIR conversion - file exists: {os.path.basename(mlir_path)}{Colors.RESET}")
        else:
            os.makedirs(os.path.dirname(mlir_path), exist_ok=True)
            convert_c_to_mlir(args.input, mlir_path)
        
        input_file = mlir_path
        print(f"{Colors.CYAN}🎯 Using converted MLIR file: {input_file}{Colors.RESET}")
    
    # Calculate block size in bytes
    block_size_bytes = args.block_size * args.element_size
    
    # Generate file prefix based on input file and arguments
    # Create a hash of the arguments to keep filename manageable
    arg_string = f"{args.input}_{args.block_size}_{args.element_size}_{args.associativity}_{args.cache_size}"
    arg_hash = hashlib.md5(arg_string.encode()).hexdigest()[:8]
    input_basename = os.path.basename(args.input).replace('.', '_')
    file_prefix = f"{input_basename}_{arg_hash}"
    
    # Print configuration with emojis and colors
    print(f"{Colors.BOLD}{Colors.YELLOW}📋 Configuration:{Colors.RESET}")
    print(f"  🎯 Input file: {Colors.CYAN}{args.input}{Colors.RESET}")
    print(f"  📏 Block size: {Colors.GREEN}{args.block_size} elements × {args.element_size} bytes = {block_size_bytes} bytes{Colors.RESET}")
    print(f"  🔢 Associativity: {Colors.GREEN}{args.associativity}-way{Colors.RESET}")
    print(f"  💾 Cache size: {Colors.GREEN}{args.cache_size} blocks{Colors.RESET}")
    print(f"  📁 Data directory: {Colors.CYAN}{args.data_dir}{Colors.RESET}")
    print(f"  🏷️  File prefix: {Colors.YELLOW}{file_prefix}{Colors.RESET}")
    
    # Ensure data directory exists
    os.makedirs(args.data_dir, exist_ok=True)
    print(f"\n{Colors.YELLOW}📁 Using data directory: {Colors.CYAN}{args.data_dir}{Colors.RESET}")
    
    # File paths with prefixed names to avoid conflicts
    full_json = os.path.join(args.data_dir, f"{file_prefix}_full.json")
    assoc_json = os.path.join(args.data_dir, f"{file_prefix}_assoc.json")
    full_db = os.path.join(args.data_dir, f"{file_prefix}_full.db")
    assoc_db = os.path.join(args.data_dir, f"{file_prefix}_assoc.db")
        
    # Step 1: Predict fully associative cache miss ratio
    if os.path.exists(full_json):
        print(f"\n{Colors.BOLD}{Colors.GREEN}⏭️  Step 1/4: Skipping fully associative analysis - file exists: {os.path.basename(full_json)}{Colors.RESET}")
    else:
        cmd1 = [
            os.getenv("ANALYZER", "analyzer"),
            "--json",
            "-i", input_file,
            "-o", full_json,
            "barvinok",
            "--infinite-repeat",
            "--barvinok-arg=--approximation-method=scale",
            f"--block-size={args.block_size}"
        ]
        run_command(cmd1, "🔬 Analyzing fully associative cache", 1, 4)
        
    # Step 2: Calculate associative cache miss ratio by Smith's method
    if os.path.exists(assoc_json):
        print(f"\n{Colors.BOLD}{Colors.GREEN}⏭️  Step 2/4: Skipping associative conversion - file exists: {os.path.basename(assoc_json)}{Colors.RESET}")
    else:
        cmd2 = [
            os.getenv("ASSOC_CONV", "assoc-conv"),
            "-i", full_json,
            "-a", str(args.associativity),
            "-o", assoc_json
        ]
        run_command(cmd2, f"🔄 Converting to {args.associativity}-way associative cache", 2, 4)
        
    # Calculate cache parameters for simulation
    # b' = block_size_bytes, c' = cache_size * b'
    b_prime = block_size_bytes
    c_prime = args.cache_size * b_prime
    
    # Step 3: Run simulator for fully associative cache
    if os.path.exists(full_db):
        print(f"\n{Colors.BOLD}{Colors.GREEN}⏭️  Step 3/4: Skipping fully associative simulation - file exists: {os.path.basename(full_db)}{Colors.RESET}")
    else:
        cmd3 = [
            os.getenv("CACHEGRIND_RUNNER", "cachegrind-runner"),
            "-i", input_file,
            "-C", str(c_prime),
            f"-B{b_prime}",
            "-c32768",
            "-b64",
            "-a16",
            "--database", full_db,
            "--batched"
        ]
        run_command(cmd3, "🖥️ Running simulation for fully associative cache", 3, 4)
    
    # Step 4: Run simulator for associative cache
    if os.path.exists(assoc_db):
        print(f"\n{Colors.BOLD}{Colors.GREEN}⏭️  Step 4/4: Skipping associative simulation - file exists: {os.path.basename(assoc_db)}{Colors.RESET}")
    else:
        cmd4 = [
            os.getenv("CACHEGRIND_RUNNER", "cachegrind-runner"),
            "-i", input_file,
            "-C", str(c_prime),
            f"-B{b_prime}",
            "-A", str(args.associativity),
            "-c32768",
            "-b64",
            "-a4",
            "--database", assoc_db,
            "--batched"
        ]
        run_command(cmd4, f"🖥️ Running simulation for {args.associativity}-way associative cache", 4, 4)
        
    # Step 5: Generate plots
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}📊 Generating plots...{Colors.RESET}")
    plt.figure(figsize=(12, 8))
    
    # Simulation data sources
    simulation = [
        (f"{args.associativity}-way", assoc_db),
        ("fully", full_db),
    ]
    
    # Prediction data sources
    prediction = [
        (f"{args.associativity}-way", assoc_json),
        ("fully", full_json),
    ]
        
    # Plot simulation data
    for name, db_file in simulation:
        try:
            cache_sizes, miss_ratios = extract_simulation_data(db_file, block_size_bytes)
            plt.plot(cache_sizes, miss_ratios, 'o-', label=f'Simulation - {name}', linewidth=2, markersize=6, alpha=0.7)
            print(f"{Colors.GREEN}✅ Loaded simulation data for {name}: {len(cache_sizes)} points{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}❌ Error loading simulation data from {db_file}: {e}{Colors.RESET}")
    
    # Plot prediction data
    for name, json_file in prediction:
        try:
            cache_sizes, miss_ratios = extract_prediction_data(json_file)
            plt.plot(cache_sizes, miss_ratios, 's--', drawstyle='steps-post', label=f'Prediction - {name}', linewidth=2, markersize=4, alpha=0.7)
            print(f"{Colors.GREEN}✅ Loaded prediction data for {name}: {len(cache_sizes)} points{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}❌ Error loading prediction data from {json_file}: {e}{Colors.RESET}")
    
    # Set plot properties
    plt.xlabel('Cache Size (number of blocks)', fontsize=14)
    plt.ylabel('Miss Ratio', fontsize=14)
    plt.title('Cache Miss Ratio Comparison: Simulation vs Prediction', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Use log scale for x-axis
    plt.xscale('log')
    
    # Set y-axis limits
    plt.ylim(0, 1)
    
    plt.tight_layout()
    
    # Save the plot only if output directory is provided
    if args.output_dir:
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        png_path = os.path.join(args.output_dir, 'cache_miss_ratio_comparison.png')
        svg_path = os.path.join(args.output_dir, 'cache_miss_ratio_comparison.svg')
        
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(svg_path, bbox_inches='tight')
        
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 SUCCESS! Plot saved as:{Colors.RESET}")
        print(f"  📊 {Colors.CYAN}{png_path}{Colors.RESET}")
        print(f"  📊 {Colors.CYAN}{svg_path}{Colors.RESET}")
    else:
        print(f"\n{Colors.YELLOW}ℹ️  No output directory specified, plots will not be saved{Colors.RESET}")
    
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}🎊 Analysis completed successfully! 🎊{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}" + "="*60 + f"{Colors.RESET}\n")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
