#!/usr/bin/env bash

# Do NOT exit on error
set +e
base="${SLURM_TMPDIR:-/tmp}/jupyter_${SLURM_JOB_ID:-$$}"
export JUPYTER_CONFIG_DIR="$base/config"
export JUPYTER_DATA_DIR="$base/data"
export JUPYTER_RUNTIME_DIR="$base/runtime"
mkdir -p "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR"


failed=()



# 🔧 Hard-coded list of notebooks to run (relative or absolute paths)
NOTEBOOKS=(
    "F-Angew2024-MT (Copy 6).ipynb"
    "F-Angew2024-MT (Copy 7).ipynb"
    "F-Angew2024-MT (Copy 8).ipynb"
    "F-Angew2024-MT (Copy 9).ipynb"
    "F-Angew2024-MT.ipynb"
)
for nb in "${NOTEBOOKS[@]}"; do
    echo "========================================"
    echo "Running notebook: $nb"
    echo "========================================"

    jupyter nbconvert --to notebook --execute "$nb" --inplace

    if [ $? -ne 0 ]; then
        echo "❌ Failed: $nb"
        failed+=("$nb")
    else
        echo "✅ Success: $nb"
    fi
done

echo "========================================"
echo "Execution finished"
echo "========================================"

if [ ${#failed[@]} -ne 0 ]; then
    echo "The following notebooks failed:"
    for nb in "${failed[@]}"; do
        echo "  - $nb"
    done
    exit 1
else
    echo "All notebooks ran successfully 🎉"
    exit 0
fi
