#!/usr/bin/env bash

# Script to run predictions for all deriche loop nests
# Generates predictions in separate directories: deriche1/, deriche2/, etc.

POLY_PATH=/home/schrodingerzy/Documents/Polygeist/build/bin
export SYMBOLICA_HIDE_BANNER=1
export RUST_LOG=info

# Get script directory and change to it
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Base directory for results
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

# Loop through each deriche file
for i in {1..6}; do
    PROGRAM_FILE="deriche${i}.c"
    PROGRAM_NAME="deriche${i}"
    
    echo "========================================"
    echo "Processing ${PROGRAM_NAME}..."
    echo "========================================"
    
    if [ ! -f "$PROGRAM_FILE" ]; then
        echo "Warning: ${PROGRAM_FILE} not found, skipping..."
        continue
    fi
    
    # Create MLIR
    echo "Generating MLIR for ${PROGRAM_NAME}..."
    $POLY_PATH/cgeist "$PROGRAM_FILE" -S -raise-scf-to-affine | \
        $POLY_PATH/polygeist-opt --strip-dlti-attributes > /tmp/"${PROGRAM_NAME}.mlir"
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to generate MLIR for ${PROGRAM_NAME}"
        continue
    fi
    
    # Create output directories
    FA_DIR="${RESULTS_DIR}/deriche${i}-fa"
    WA_DIR="${RESULTS_DIR}/deriche${i}-12way"
    mkdir -p "$FA_DIR"
    mkdir -p "$WA_DIR"
    
    # Run fully-associative prediction
    echo "Running FA prediction for ${PROGRAM_NAME}..."
    cargo run --release --quiet --bin analyzer -- \
        -i /tmp/"${PROGRAM_NAME}.mlir" \
        --json \
        -o "${FA_DIR}/${PROGRAM_NAME}.json" \
        barvinok \
        --barvinok-arg=--approximation-method=scale \
        --infinite-repeat \
        --block-size=8
    
    if [ $? -ne 0 ]; then
        echo "Error: FA prediction failed for ${PROGRAM_NAME}"
        continue
    fi
    
    # Run 12-way associative prediction
    echo "Running 12-way prediction for ${PROGRAM_NAME}..."
    cargo run --release --quiet --bin assoc-conv -- \
        -o "${WA_DIR}/${PROGRAM_NAME}.json" \
        -i "${FA_DIR}/${PROGRAM_NAME}.json" \
        -a 12
    
    if [ $? -ne 0 ]; then
        echo "Error: 12-way prediction failed for ${PROGRAM_NAME}"
        continue
    fi
    
    echo "✓ Completed ${PROGRAM_NAME}"
    echo ""
done

echo "========================================"
echo "All predictions complete!"
echo "========================================"
echo ""
echo "Results saved in:"
for i in {1..6}; do
    echo "  - ${RESULTS_DIR}/deriche${i}-fa/"
    echo "  - ${RESULTS_DIR}/deriche${i}-12way/"
done
