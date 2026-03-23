#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${ROOT_DIR}/artifacts"
DO_RUN=1
DO_PLOT=1
ONLY_DATASET=""

usage() {
  cat <<'EOF'
Run all PFNC paper pipelines (experiments + plots).

Usage:
  ./scripts/run_all_pipelines.sh [options]

Options:
  --out-root <path>     Root output directory (default: ./artifacts)
  --only <dataset>      Run only one dataset: iris | diabetes | adult | wine | student
  --skip-run            Skip experiment CSV generation
  --skip-plot           Skip figure generation
  -h, --help            Show this help

Notes:
  - This script always uses uv run for Python commands.
  - Outputs are saved under:
      <out-root>/results/<dataset>
      <out-root>/figures/<dataset>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-root)
      OUT_ROOT="$2"
      shift 2
      ;;
    --only)
      ONLY_DATASET="$2"
      shift 2
      ;;
    --skip-run)
      DO_RUN=0
      shift
      ;;
    --skip-plot)
      DO_PLOT=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

run_dataset() {
  local dataset="$1"
  local k_start="$2"
  local k_end="$3"
  local repeats="$4"

  local result_dir="${OUT_ROOT}/results/${dataset}"
  local figure_dir="${OUT_ROOT}/figures/${dataset}"

  echo "========================================================"
  echo "Dataset: ${dataset}"
  echo "k range: ${k_start}-${k_end}"
  echo "Result dir: ${result_dir}"
  echo "Figure dir: ${figure_dir}"

  if [[ "${DO_RUN}" -eq 1 ]]; then
    echo "Running experiments..."
    uv run pfncc run \
      --dataset "${dataset}" \
      --k-start "${k_start}" \
      --k-end "${k_end}" \
      --out-dir "${result_dir}"
  fi

  if [[ "${DO_PLOT}" -eq 1 ]]; then
    echo "Generating plots..."
    uv run pfncc plot \
      --dataset "${dataset}" \
      --k-start "${k_start}" \
      --k-end "${k_end}" \
      --result-dir "${result_dir}" \
      --out-dir "${figure_dir}" \
      --repeats "${repeats}"
  fi
}

cd "${ROOT_DIR}"
mkdir -p "${OUT_ROOT}"

if [[ -n "${ONLY_DATASET}" ]]; then
  case "${ONLY_DATASET}" in
    iris)
      run_dataset iris 2 25 1
      ;;
    diabetes)
      run_dataset diabetes 5 25 40
      ;;
    adult)
      run_dataset adult 5 25 40
      ;;
    wine)
      run_dataset wine 2 15 1
      ;;
    student)
      run_dataset student 2 15 1
      ;;
    *)
      echo "Invalid --only value: ${ONLY_DATASET}"
      usage
      exit 2
      ;;
  esac
else
  run_dataset iris 2 25 1
  run_dataset diabetes 5 25 40
  run_dataset adult 5 25 40
  run_dataset wine 2 15 1
  run_dataset student 2 15 1
fi

echo "Done. Outputs saved under: ${OUT_ROOT}"
