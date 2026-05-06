{
  description = "Buddy Examples - Development environment with buddy-mlir";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    # Import buddy-mlir from local thirdparty directory (as git submodule)
    buddy-mlir = {
      url = "git+file:./thirdparty/buddy-mlir?ref=nix-config-update-latest";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, buddy-mlir }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Get buddy-mlir packages
        buddyPkgs = buddy-mlir.legacyPackages.${system};

      in
      {
        # Expose buddy-mlir as the default package for `nix build`
        packages.default = buddyPkgs.buddy-mlir;

        # Also expose individual packages
        packages = {
          buddy-mlir = buddyPkgs.buddy-mlir;
          buddy-llvm = buddyPkgs.buddy-llvm;
        };

        devShells.default = pkgs.mkShell {
          buildInputs = [
            # Buddy MLIR and LLVM
            buddyPkgs.buddy-mlir
            buddyPkgs.buddy-llvm

            # Common development tools
            pkgs.cmake
            pkgs.ninja
            pkgs.git

            # Python environment
            (pkgs.python3.withPackages (ps: [
              ps.numpy
              ps.pybind11
              ps.pyyaml
            ]))

          ];

          shellHook = ''
            echo "🚀 Buddy Examples Development Environment"
            echo "  - buddy-mlir: ${buddyPkgs.buddy-mlir}"
            echo "  - buddy-llvm: ${buddyPkgs.buddy-llvm}"
            echo ""
            echo "Available commands:"
            echo "  - buddy-opt, buddy-translate, etc."
            echo ""
          '';
        };

        formatter = pkgs.nixpkgs-fmt;
      }
    );
}
