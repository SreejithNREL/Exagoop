#!/usr/bin/env bash
# Generates the material-point file and the ExaGOOP input file from
# ./PreProcess/config_damage.json using the repository-wide preprocessor.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
python3 "$HERE/../../Tools/Preprocess/Generate_MPs_Inputfile_Generic.py" --config "$HERE/PreProcess/config_damage.json"
