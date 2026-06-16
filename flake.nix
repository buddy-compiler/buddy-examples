{
  description = "Buddy Examples - Development environment with buddy-mlir";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    buddy-mlir = {
      url = "path:./thirdparty/buddy-mlir";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    bebop = {
      url = "path:./thirdparty/bebop";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, buddy-mlir, bebop }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlay = import ./scripts/nix/overlay.nix { inherit buddy-mlir bebop; };
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ overlay ];
        };

      in
      {
        packages.default = pkgs.buddy-compiler;
        packages = {
          buddy-compiler = pkgs.buddy-compiler;
          buddy-llvm = pkgs.buddy-llvm;
          bebop = pkgs.bebop;
        };

        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.buddy-compiler
            pkgs.buddy-llvm
            pkgs.bebop
            pkgs.cmake
            pkgs.ninja
            pkgs.git
            (pkgs.python3.withPackages (ps: [
              ps.numpy
              ps.pybind11
              ps.pyyaml
            ]))

          ];

          shellHook = ''
          '';
        };

        formatter = pkgs.nixpkgs-fmt;
      }
    ) // {
      overlays.default = import ./scripts/nix/overlay.nix { inherit buddy-mlir bebop; };
    };
}
