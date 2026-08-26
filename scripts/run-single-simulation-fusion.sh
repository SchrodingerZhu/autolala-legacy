#!/usr/bin/env bash

POLY_PATH="${POLY_PATH:-}"
export SYMBOLICA_HIDE_BANNER=1
export RUST_LOG=warn
PROGRAM_NAME=$(basename $1 .c)
${POLY_PATH:+$POLY_PATH/}cgeist $1 -S -raise-scf-to-affine | \
    ${POLY_PATH:+$POLY_PATH/}polygeist-opt --canonicalize | sed -E "1s/^module attributes \{.*\} \{$/module {/" | /home/schrodingerzy/Documents/llvm-project/mlir-opt/bin/mlir-opt -affine-loop-fusion -affine-loop-normalize -canonicalize | tee | cat > /tmp/"${PROGRAM_NAME}.mlir"

cargo run --release --bin cachegrind-runner --quiet -- -i /tmp/"${PROGRAM_NAME}.mlir" "${@:2}"


