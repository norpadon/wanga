{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
          };
        };
        python = pkgs.python310;
      in
      {
        devShells = {
          default = pkgs.mkShell {
            packages = with pkgs; [
              poetry
              python
            ];
            shellHook = ''
              poetry env use ${python}/bin/python
              poetry install --no-root
              export PYTHONPATH=$PYTHONPATH:$PWD/nuggets
            '';
          };
        };
      });
}
