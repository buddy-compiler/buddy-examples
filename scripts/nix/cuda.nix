{ pkgs }:

let
  cudaPkgs = pkgs.cudaPackages_12_8;
in
{
  cudatoolkit = cudaPkgs.cudatoolkit;

  # User-space driver (libcuda and the PTX JIT) for the host's kernel module,
  # whose version it must equal (/proc/driver/nvidia/version).
  driver =
    (pkgs.linuxPackages.nvidiaPackages.mkDriver {
      version = "550.54.14";
      sha256_64bit = "sha256-jEl/8c/HwxD7h1FJvDD6pP0m0iN7LLps0uiweAFXz+M=";
      useSettings = false;
      usePersistenced = false;
      useFabricmanager = false;
    }).override { libsOnly = true; };
  nvcc = cudaPkgs.cuda_nvcc;
  cudart = cudaPkgs.cuda_cudart;

  # CUDA host compiler: default nix gcc is 15, toolkit wants <=13.
  # Versioned names only — avoids colliding with default gcc on PATH.
  gcc13 = pkgs.runCommand "gcc13-named" { } ''
    mkdir -p $out/bin
    for b in gcc g++ c++ cpp; do
      [ -e ${pkgs.gcc13}/bin/$b ] || continue
      ln -s ${pkgs.gcc13}/bin/$b $out/bin/$b-13
    done
  '';
}
