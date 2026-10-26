{ pkgs }:

let
  # Build newlib-nano with size optimization flags
  newlib-nano = pkgs.pkgsCross.riscv64-embedded.newlib.overrideAttrs (oldAttrs: {
    pname = "newlib-nano";
    configureFlags = oldAttrs.configureFlags or [] ++ [
      "--enable-newlib-nano-malloc"
      "--enable-newlib-nano-formatted-io"
      "--enable-newlib-reent-small"
      "--disable-newlib-fvwrite-in-streamio"
      "--disable-newlib-fseek-optimization"
      "--disable-newlib-wide-orient"
      "--disable-newlib-unbuf-stream-opt"
      "--enable-lite-exit"
      "--enable-newlib-global-atexit"
    ];
    CFLAGS_FOR_TARGET = "-Os -ffunction-sections -fdata-sections -mcmodel=medany";
  });

  # Create a custom cross system with newlib-nano
  riscv64EmbeddedWithNano = pkgs.pkgsCross.riscv64-embedded.stdenv.targetPlatform // {
    libc = "newlib-nano";
  };

  # Build the toolchain with the custom platform
  pkgsCrossWithNano = import pkgs.path {
    system = pkgs.stdenv.hostPlatform.system;
    crossSystem = riscv64EmbeddedWithNano;
    overlays = [
      (self: super: {
        newlib = newlib-nano;
      })
    ];
  };
in

{
  # RISC-V embedded toolchain (bare metal), with riscv64-unknown-elf-* symlinks
  riscv-embedded-gcc = pkgs.symlinkJoin {
    name = "riscv64-unknown-elf-gcc";
    paths = [ pkgsCrossWithNano.buildPackages.gcc ];
    postBuild = ''
      cd $out/bin
      for f in riscv64-none-elf-*; do
        [ -e "$f" ] || continue
        newname=''${f/riscv64-none-elf/riscv64-unknown-elf}
        ln -sf "$f" "$newname"
      done
    '';
  };

  # Add the static libc search path only for explicit static links.
  riscv-linux-gcc = let
    cc = pkgs.pkgsCross.riscv64.stdenv.cc;
    libcStatic = pkgs.pkgsCross.riscv64.stdenv.cc.libc.static;
  in pkgs.runCommand "riscv64-linux-gnu-toolchain" {} ''
    mkdir -p $out/bin
    for f in ${cc}/bin/riscv64-unknown-linux-gnu-*; do
      [ -e "$f" ] || continue
      name=$(basename "$f")
      case "$name" in
        riscv64-unknown-linux-gnu-gcc|riscv64-unknown-linux-gnu-g++|riscv64-unknown-linux-gnu-c++)
          echo '#!${pkgs.stdenv.shell}' > $out/bin/$name
          echo 'for arg in "$@"; do' >> $out/bin/$name
          echo '  if [ "$arg" = -static ]; then' >> $out/bin/$name
          echo '    exec "'"$f"'" -L${libcStatic}/lib "$@"' >> $out/bin/$name
          echo '  fi' >> $out/bin/$name
          echo 'done' >> $out/bin/$name
          echo 'exec "'"$f"'" "$@"' >> $out/bin/$name
          chmod +x $out/bin/$name
          ;;
        *)
          ln -s "$f" $out/bin/$name
          ;;
      esac
    done
  '';
}
