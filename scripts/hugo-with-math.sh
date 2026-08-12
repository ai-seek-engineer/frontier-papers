#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_content_dir="${repo_dir}/content"
staging_parent="$(mktemp -d "${TMPDIR:-/tmp}/frontier-papers-content.XXXXXX")"
staging_dir="${staging_parent}/content"

cleanup() {
  rm -rf -- "$staging_parent"
}
trap cleanup EXIT

cd "$repo_dir"

python3 "${repo_dir}/scripts/prepare-hugo-content.py" \
  "$source_content_dir" \
  "$staging_dir"

bash "${repo_dir}/scripts/clean-research-citations.sh" "${staging_dir}/research"
hugo --contentDir "$staging_dir" "$@"
