#!/usr/bin/env bash
set -Eeuo pipefail

unset EVO_ENABLE_THINKING EVO_REASONING_EFFORT EVO_MODEL_FAST EVO_MODEL_STRONG EVO_MODEL_AUX EVO_AUX_MODEL_NAME
unset EVO_MAX_PARALLEL_WORKERS EVO_MAX_TASKS_PER_BATCH EVO_MAX_RUN_TOKENS EVO_MAX_TASK_OUTPUT_TOKENS

export FORCE_COLOR=1

export EVO_SUPERVISOR_ENABLE_THINKING=1
export EVO_SUPERVISOR_REASONING_EFFORT=xhigh
export EVO_WORKER_ENABLE_THINKING=1
export EVO_WORKER_REASONING_EFFORT=medium
export PYTHONUNBUFFERED=1

CAMPAIGN_DIR="${CAMPAIGN_DIR:-results/evo-campaign-qwen38-single-worker-c13-v2}"
TASK_PARALLELISM="${TASK_PARALLELISM:-1}"
DATASET="campaign13-python-behavior/prepared/dataset.jsonl"

mkdir -p "$CAMPAIGN_DIR"

for ROUND in 1 2 3; do
  echo "========== Starting Round ${ROUND} =========="

  uv run evoci benchmark run \
    --manifest "campaign13-python-behavior/manifests/round-${ROUND}.jsonl" \
    --dataset "$DATASET" \
    --variant evo \
    --campaign-dir "$CAMPAIGN_DIR" \
    --round "$ROUND" \
    --freeze-learning-within-round \
    --docker-official-images \
    --parallelism "$TASK_PARALLELISM" \
    2>&1 | tee "$CAMPAIGN_DIR/round-${ROUND}.log"

  status=${PIPESTATUS[0]}
  if [[ "$status" -ne 0 ]]; then
    echo "========== Round ${ROUND} failed with status ${status} =========="
    exit "$status"
  fi

  echo "========== Round ${ROUND} completed =========="
done

echo "========== All three rounds completed =========="
