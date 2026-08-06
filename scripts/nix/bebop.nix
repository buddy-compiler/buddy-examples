{ writeShellApplication, lib, cargo, rustc, src }:
let
  cargoFile = "${src}/bebop/Cargo.toml";
in
if builtins.pathExists cargoFile then
  writeShellApplication {
    name = "bebop";
    runtimeInputs = [ cargo rustc ];
    text = ''
      set -euo pipefail
      exec cargo run --release --manifest-path ${lib.escapeShellArg cargoFile} --bin bebop -- "$@"
    '';
  }
else
  throw "Missing bebop manifest: ${cargoFile}"
