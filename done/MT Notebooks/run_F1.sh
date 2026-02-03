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
    "E-JACS2023-MT.ipynb"
    "F-Angew2024-MT (Copy 10).ipynb"
    "F-Angew2024-MT (Copy 2).ipynb"
    "F-Angew2024-MT (Copy 3).ipynb"
    "F-Angew2024-MT (Copy 4).ipynb"
    "F-Angew2024-MT (Copy 5).ipynb"
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
