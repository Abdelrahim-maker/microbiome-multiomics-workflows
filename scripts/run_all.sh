#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$TASK_ROOT"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/carbon-circle-mpl
TASK_PYTHON="${CARBON_CIRCLE_PYTHON:-$TASK_ROOT/protein_cluster_target_analysis_263MAGs/joint_rpca_env/bin/python}"
if [[ ! -x "$TASK_PYTHON" ]]; then TASK_PYTHON=python; fi
for stage in prepare analyze sensitivity rna mapping_audit validate report; do
  TASK_STAGE_PYTHON="${CARBON_CIRCLE_IO_PYTHON:-python}"
  if [[ "$stage" == analyze || "$stage" == report ]]; then TASK_STAGE_PYTHON="$TASK_PYTHON"; fi
  "$TASK_STAGE_PYTHON" "fnal results/scripts/$stage.py" > "fnal results/$stage.log" 2>&1
done
