#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
self="$(basename -- "$0")"

for f in "$dir"/*.sh; do
  base="$(basename -- "$f")"
  [[ "$base" == "fix_aliked_custom_ops.sh" || "$base" == "$self" ]] && continue
  echo ">> Running $base"
  bash "$f"
done
