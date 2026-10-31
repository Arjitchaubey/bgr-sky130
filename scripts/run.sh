#!/usr/bin/env bash
# Usage: scripts/run.sh sim/test_nmos.spice
set -e
mkdir -p build results
f="$1"
out="build/$(basename "$f")"
envsubst '$PDK_ROOT' < "$f" > "$out"
ngspice -b "$out" 2>&1 | tee "build/$(basename "$f" .spice).log"
