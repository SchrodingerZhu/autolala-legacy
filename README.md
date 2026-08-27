# AutoLALA

AUTOmatic Loop Asympotic Locality Analysis

## How to Compile

(assume ubuntu)
```bash
# install LLVM
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 20

# install build tools
sudo apt install build-essential cmake autoconf libtool

# set environment variables
export MLIR_SYS_200_PREFIX=/usr/lib/llvm-20
export TABLEGEN_200_PREFIX=/usr/lib/llvm-20

# build and test
cargo build --release
cargo test --release
```

## Recommended development setup (for VSCode)

- Install DirEnv 
  - [Installation](https://direnv.net/docs/installation.html)
  - [Setup](https://direnv.net/docs/hook.html)

- Create `.envrc` file at the root of this project
  ```bash
  # .envrc
  export MLIR_SYS_200_PREFIX=/usr/lib/llvm-20
  export TABLEGEN_200_PREFIX=/usr/lib/llvm-20
  ```

- Install `rust-analyzer` extension for VSCode
  - [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=matklad.rust-analyzer)

  Add the following settings to `.vscode/settings` under the root of this project
  ```json
  {
    "rust-analyzer.cargo.extraEnv": {
        "MLIR_SYS_200_PREFIX" : "/usr/lib/llvm-20",
        "TABLEGEN_200_PREFIX" : "/usr/lib/llvm-20",
    },
    "rust-analyzer.check.extraEnv": {
        "MLIR_SYS_200_PREFIX" : "/usr/lib/llvm-20",
        "TABLEGEN_200_PREFIX" : "/usr/lib/llvm-20",
    },
    "rust-analyzer.server.extraEnv": {
        "MLIR_SYS_200_PREFIX" : "/usr/lib/llvm-20",
        "TABLEGEN_200_PREFIX" : "/usr/lib/llvm-20",
    },
    "rust-analyzer.runnables.extraEnv": {
        "MLIR_SYS_200_PREFIX" : "/usr/lib/llvm-20",
        "TABLEGEN_200_PREFIX" : "/usr/lib/llvm-20",
    },
    "editor.formatOnSave": true,
    "files.insertFinalNewline": true,
  }
  ```

## Parallel Loop Locality

Two pieces, meant to be used together: a sampler that measures ground truth, and
an analytical model checked against it.

### `parallel-sampler` (`psample`) — reproducible ground truth

Measures reuse for a loop nest parallelized by `schedule(static, chunk)`, from
the same affine MLIR the analyzer consumes. Thread interleaving is *simulated*
by a seeded PRNG rather than run on real threads, so a run is a pure function of
its configuration and repeats byte-for-byte across machines. Every report
carries a manifest and a digest for exactly that reason.

```bash
cargo run --release -p parallel-sampler -- \
  --input parallel-sampler/misc/matmul_128.mlir \
  --parallel-loop-depth 0 --threads 8 --chunk 1 --seed 1
```

It reports reuse-interval and reuse-distance histograms, two miss-ratio curves,
and — the part that makes it a diagnostic rather than just a measurement — the
joint law of private versus concurrent reuse interval, split by whether the
reuse crossed a thread boundary and attributed to the access site and array it
came from.

`--scheduler` selects the interleaving model. `uniform` is the analytical
model's own assumption, so a disagreement under it is a fault in the model's
algebra rather than in its premises; `round-robin`, `burst:<n>` and
`skewed:<rates>` exist to probe what happens when the assumption fails.

### `analyzer --parallel-loop-depth` — the model

Adds the PACT'24 concurrent-reuse-interval model (DOI 10.1145/3656019.3676948)
to the Barvinok path. The thread that owns each iteration is materialized as an
extra polyhedral dimension, which lets the same timestamp space yield both the
sequential reuse interval and the thread-private one, and makes data sharing an
exact set operation rather than a syntactic guess about array subscripts.

```bash
cargo run --release -p analyzer -- --input parallel-sampler/misc/matmul_128.mlir \
  barvinok --parallel-loop-depth 0 --threads 8 --chunk 1
```

### Validating one against the other

```bash
scripts/validate-parallel.py           # writes results/parallel/validation.json
```

Errors are reported in three parts — the CRI model, the reuse-interval-to-
reuse-distance conversion, and the two combined — because a prediction can go
wrong in either place independently, and a fourth number guards the reuse
extraction that feeds them.

### Build note

`mlir-sys` links `-lMLIR-C`. Distributions that ship a shared LLVM but only
static MLIR C-API archives (Arch's `mlir` package, for one) have no such
library, and *every* binary in this workspace fails to link. Build one from the
archives and put it on the search path:

```bash
mkdir -p ~/.local/lib/mlir-c-shim && cd ~/.local/lib/mlir-c-shim
{ echo "create libMLIR-C.a"; for a in /usr/lib/libMLIRCAPI*.a; do echo "addlib $a"; done; \
  echo save; echo end; } | ar -M
export RUSTFLAGS="$RUSTFLAGS -L native=$HOME/.local/lib/mlir-c-shim"
```
