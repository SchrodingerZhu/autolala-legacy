{
  description = "AutoLALA — automatic loop asymptotic locality analysis";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, rust-overlay }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ];
      forEachSystem = f: nixpkgs.lib.genAttrs systems (system:
        f (import nixpkgs {
          inherit system;
          overlays = [ rust-overlay.overlays.default ];
        }));
    in
    {
      devShells = forEachSystem (pkgs:
        let
          lib = pkgs.lib;
          llvmPkgs = pkgs.llvmPackages_22;

          # mlir-sys/tblgen expect a single prefix containing llvm-config,
          # the LLVM/MLIR libraries, and headers.
          mlirPrefix = pkgs.symlinkJoin {
            name = "mlir-llvm-22";
            paths = [
              llvmPkgs.mlir
              llvmPkgs.mlir.dev
              llvmPkgs.libllvm
              llvmPkgs.libllvm.lib
              llvmPkgs.libllvm.dev
            ];
            # llvm-config reports the store paths of the split llvm outputs,
            # which lack the MLIR headers/libs; rewrite them to this joined
            # prefix so mlir-sys/raffine/tblgen find everything in one place.
            postBuild = ''
              rm "$out/bin/llvm-config"
              cat > "$out/bin/llvm-config" <<EOF
              #!${pkgs.runtimeShell}
              set -o pipefail
              ${llvmPkgs.libllvm.dev}/bin/llvm-config "\$@" | sed -e "s|${llvmPkgs.libllvm.dev}|$out|g" -e "s|${llvmPkgs.libllvm.lib}|$out|g"
              EOF
              sed -i 's/^ *//' "$out/bin/llvm-config"
              chmod +x "$out/bin/llvm-config"
            '';
          };

          # barvinok-sys reads these prefixes on macOS instead of the
          # compiler's default search paths.
          gmpPrefix = pkgs.symlinkJoin {
            name = "gmp-joined";
            paths = [ pkgs.gmp pkgs.gmp.dev ];
          };

          toolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain;
        in
        {
          default = pkgs.mkShell {
            # glibc's _FORTIFY_SOURCE emits a -Werror'd #warning in -O0
            # builds (tblgen's C++ shim, debug-profile build scripts).
            hardeningDisable = [ "fortify" ];

            nativeBuildInputs = with pkgs; [
              toolchain
              lld_22
              pkg-config
              # cachegrind-runner shells out to clang++ and valgrind
              llvmPkgs.clang
              valgrind
              # autotools stack for the vendored barvinok/isl build
              autoconf
              automake
              libtool
              m4
              gnumake
              bison
              flex
            ];

            buildInputs = with pkgs; [
              gmp
              mpfr
              libmpc
              ntl
              fontconfig
              freetype
              zlib
              zstd
              ncurses
              libxml2
              libffi
            ];

            env = {
              MLIR_SYS_220_PREFIX = "${mlirPrefix}";
              TABLEGEN_220_PREFIX = "${mlirPrefix}";
              GMP_PREFIX = "${gmpPrefix}";
              NTL_PREFIX = "${pkgs.ntl}";
              LIBCLANG_PATH = "${llvmPkgs.libclang.lib}/lib";
              BINDGEN_EXTRA_CLANG_ARGS = builtins.concatStringsSep " " (
                [
                  "-isystem ${llvmPkgs.libclang.lib}/lib/clang/${lib.versions.major llvmPkgs.libclang.version}/include"
                ]
                ++ lib.optionals pkgs.stdenv.hostPlatform.isLinux [
                  "-isystem ${pkgs.stdenv.cc.libc.dev}/include"
                ]
                ++ [
                  "-I${pkgs.gmp.dev}/include"
                  "-I${pkgs.ntl}/include"
                ]
              );
            };

            # If the host routes rustc through sccache with distributed
            # compilation, force rustc jobs local: proc macros (melior-macro)
            # link nix-store MLIR libs that remote build servers can't load.
            # A private daemon with a dist-free config keeps the remote
            # cache (env-configured) working. No-op on hosts without sccache.
            shellHook = ''
              case "''${RUSTC_WRAPPER:-}" in
                *sccache*)
                  export SCCACHE_CONF=${pkgs.writeText "sccache-no-dist.toml" ""}
                  export SCCACHE_SERVER_PORT=''${SCCACHE_SERVER_PORT:-4227}
                  ;;
              esac
            '';
          };
        });
    };
}
