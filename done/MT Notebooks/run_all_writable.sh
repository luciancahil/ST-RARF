#!/usr/bin/env bash
set +e

base="${SLURM_TMPDIR:-/tmp}/jupyter_${SLURM_JOB_ID:-$$}"
export JUPYTER_CONFIG_DIR="$base/config"
export JUPYTER_DATA_DIR="$base/data"
export JUPYTER_RUNTIME_DIR="$base/runtime"
mkdir -p "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR"

failed=()
NOTEBOOKS=(
    "G-Angew2021-MT (Copy 2).ipynb"
    "G-Angew2021-MT (Copy 3).ipynb"
    "G-Angew2021-MT.ipynb"
)


for nb in *.ipynb; do
  echo "Running: $nb"
  jupyter nbconvert --to notebook --execute "$nb" --inplace
  if [ $? -ne 0 ]; then
    echo "❌ Failed: $nb"
    failed+=("$nb")
  else
    echo "✅ Success: $nb"
  fi
done

if [ ${#failed[@]} -ne 0 ]; then
  echo "Failed notebooks:"
  printf ' - %s\n' "${failed[@]}"
  exit 1
fi
