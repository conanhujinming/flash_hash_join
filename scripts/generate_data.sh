#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
generator="${repository_root}/db-benchmark/_data/join-datagen.R"
data_dir="${repository_root}/data"

if [[ ! -f "${generator}" ]]; then
    echo "db-benchmark is missing; run: git submodule update --init db-benchmark" >&2
    exit 1
fi

mkdir -p "${data_dir}"
cd "${data_dir}"
for size in 1e7 2e7 4e7; do
    Rscript "${generator}" "${size}" 0 0 0
done
