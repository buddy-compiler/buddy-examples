{
  description = "Buddy Codesign development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        python = import ./scripts/nix/python.nix { inherit pkgs; };
        riscv = import ./scripts/nix/riscv.nix { inherit pkgs; };
        cuda = import ./scripts/nix/cuda.nix { inherit pkgs; };
        clibs = import ./scripts/nix/clibs.nix { inherit pkgs; };
        libraries = with pkgs; [
          zlib
          zlib.dev
          lz4.lib
          lz4.dev
          readline
          readline.dev
          libjpeg
          libjpeg.dev
          libpng
          libpng.dev
          elfutils.out
          elfutils.dev
          gmp
          gmp.dev
          clibs.libdwarf
          flatbuffers
          numactl
        ];
        environment = pkgs.buildEnv {
          name = "buddy-codesign-environment";
          paths = with pkgs; [
            python.python3Packages
            riscv.riscv-embedded-gcc
            riscv.riscv-linux-gcc
            cmake
            ninja
            clang-tools
            lld
            gcc
            gnumake
            pkg-config
            git
            rsync
            curl
            wget
            unzip
            zip
            patch
            which
            file
            findutils
            autoconf
            automake
            libtool
            bison
            flex
            texinfo
            dtc
            cpio
            bc
            verilator
            spike
            ccache
            sbt
            jdk17
            coursier
            pre-commit
            gersemi
            cuda.cudatoolkit
            cuda.gcc13
          ] ++ libraries;
        };
      in {
        packages.default = environment;
        devShells.default = pkgs.mkShell {
          packages = [ environment ];
          buildInputs = libraries;
          hardeningDisable = [ "libcxxhardeningfast" ];
          shellHook = ''
            codesign_root=$(git rev-parse --show-toplevel)
            export PATH="$codesign_root/thirdparty/buddy-mlir/build/bin:$codesign_root/platforms/thirdparty/chipyard/sims/firesim/deploy:$codesign_root/platforms/thirdparty/chipyard/software/firemarshal:$PATH"
            export PYTHONPATH="$codesign_root:$codesign_root/thirdparty/buddy-mlir/llvm/build/tools/mlir/python_packages/mlir_core:$codesign_root/thirdparty/buddy-mlir/build/python_packages''${PYTHONPATH:+:$PYTHONPATH}"
            export RISCV="${environment}"
            export CC="${pkgs.clang}/bin/clang"
            export CXX="${pkgs.clang}/bin/clang++"
            export JAVA_HOME="${pkgs.jdk17}"
            export CUDA_HOME="${cuda.cudatoolkit}"
            export CPATH="''${CUDA_HOME}/include''${CPATH:+:$CPATH}"
            # Use the host driver at runtime, not the CUDA Toolkit's link stubs.
            cuda_driver=$(${pkgs.glibc.bin}/bin/ldconfig -p -C /etc/ld.so.cache | ${pkgs.gawk}/bin/awk '$1 == "libcuda.so.1" && /libc6,x86-64/ { print $NF; exit }')
            if [ ! -r "$cuda_driver" ]; then
              echo "ERROR: NVIDIA libcuda.so.1 is missing from the host library cache." >&2
              exit 1
            fi
            export LD_PRELOAD="$cuda_driver''${LD_PRELOAD:+:$LD_PRELOAD}"
            unset cuda_driver
            export OBJCACHE=ccache
            unset codesign_root
          '';
        };
      });
}
