#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
python3 "$HERE/PreProcess/gen_blind_crack.py" --config "$HERE/PreProcess/config.json"
