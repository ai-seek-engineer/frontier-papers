#!/usr/bin/env bash
set -euo pipefail

research_dir="${1:-content/research}"

if [[ ! -d "$research_dir" ]]; then
  echo "Research content directory not found: $research_dir" >&2
  exit 1
fi

find "$research_dir" -type f -name '*.md' -print0 |
  while IFS= read -r -d '' file; do
    perl -0pi -CSDA -e 's/[[:blank:]]*\x{E200}(?:file)?cite(?:\x{E202}[^\x{E200}\x{E201}]+)+\x{E201}//g' "$file"
  done
