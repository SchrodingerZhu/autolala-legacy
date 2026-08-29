
mkdir -p results/12-way-skew-1.5
for i in results/fully-associative/*.json; do
    PROGRAM_NAME=$(basename $i .json)
    echo "Processing $PROGRAM_NAME"
    cargo run --release --quiet --bin assoc-conv -- -o results/12-way-skew-1.5/${PROGRAM_NAME}.json -i $i -a 12
done
