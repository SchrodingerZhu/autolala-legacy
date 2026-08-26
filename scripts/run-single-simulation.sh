#!/usr/bin/env bash

POLY_PATH="${POLY_PATH:-}"
export SYMBOLICA_HIDE_BANNER=1
export RUST_LOG=trace
PROGRAM_NAME=$(basename $1 .c)
${POLY_PATH:+$POLY_PATH/}cgeist $1 -S -raise-scf-to-affine | \
    ${POLY_PATH:+$POLY_PATH/}polygeist-opt --canonicalize | sed -E "1s/^module attributes \{.*\} \{$/module {/" > /tmp/"${PROGRAM_NAME}.mlir"

cargo run --release --bin cachegrind-runner --quiet -- -i /tmp/"${PROGRAM_NAME}.mlir" "${@:2}"


