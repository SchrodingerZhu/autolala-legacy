#!/usr/bin/env bash

POLY_PATH="${POLY_PATH:-}"
export SYMBOLICA_HIDE_BANNER=1
export RUST_LOG=info
PROGRAM_NAME=$(basename $1 .c)
${POLY_PATH:+$POLY_PATH/}cgeist $1 -S -raise-scf-to-affine | \
    ${POLY_PATH:+$POLY_PATH/}polygeist-opt --canonicalize | sed -E "1s/^module attributes \{.*\} \{$/module {/" > /tmp/"${PROGRAM_NAME}.mlir"

mkdir -p results/fully-associative
mkdir -p results/12-way
cargo run --release --quiet --bin analyzer -- -i /tmp/"${PROGRAM_NAME}.mlir" --json -o results/fully-associative/${PROGRAM_NAME}.json barvinok --barvinok-arg=--approximation-method=scale --infinite-repeat --block-size=8
cargo run --release --quiet --bin assoc-conv -- -o results/12-way/${PROGRAM_NAME}.json -i results/fully-associative/${PROGRAM_NAME}.json -a 12

