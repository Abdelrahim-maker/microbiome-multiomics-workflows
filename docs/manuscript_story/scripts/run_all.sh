#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$TASK_ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/story-mpl
TASK_PYTHON="${CARBON_CIRCLE_PYTHON:-$TASK_ROOT/protein_cluster_target_analysis_263MAGs/joint_rpca_env/bin/python}"
for stage in prepare_story analyze_story redundancy_bootstrap fertilizer legacy_oss carbon_followup finalize_story validate_story; do
  "$TASK_PYTHON" "fnal results/manuscript_story/scripts/$stage.py" > "fnal results/manuscript_story/$stage.log" 2>&1
done
