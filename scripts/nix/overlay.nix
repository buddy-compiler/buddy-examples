{ buddy-mlir, bebop }:
final: prev:
let
  buddyPkgs = buddy-mlir.legacyPackages.${final.system};
in {
  buddy-compiler = final.callPackage ./compiler.nix {
    inherit buddyPkgs;
  };
  buddy-llvm = buddyPkgs.buddy-llvm;
  bebop = final.callPackage ./bebop.nix {
    src = bebop;
  };
}
