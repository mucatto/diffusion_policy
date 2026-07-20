#!/usr/bin/env bash
set -Eeuo pipefail

# Required: choose an idle physical GPU only after a fresh nvidia-smi check.
: "${DP_GPU:?Set DP_GPU to one confirmed-idle GPU index before running.}"

DP_CODE_ROOT="${DP_CODE_ROOT:-$(pwd)}"
DP_STORAGE_ROOT="${DP_STORAGE_ROOT:-/groups/2/sk/diffusion_policy}"
DP_IMAGE="${DP_IMAGE:-zty/diffusion-policy-pusht:lowdim-v1}"
DP_RUN_NAME="${DP_RUN_NAME:-pusht-lowdim-$(date +%Y%m%d-%H%M%S)}"
DP_CHECKPOINT="${DP_CHECKPOINT:-${DP_STORAGE_ROOT}/checkpoints/pusht_lowdim.ckpt}"
DP_OUTPUT_DIR="${DP_OUTPUT_DIR:-${DP_STORAGE_ROOT}/runs/${DP_RUN_NAME}}"
DP_LOG_PATH="${DP_LOG_PATH:-${DP_STORAGE_ROOT}/runs/${DP_RUN_NAME}.log}"
# Conservative defaults for a shared host. They stay below the workspace
# ceiling of 8 CPU threads / about 12 GiB memory.
DP_N_ENVS="${DP_N_ENVS:-4}"
DP_MAX_RUNTIME_SECONDS="${DP_MAX_RUNTIME_SECONDS:-85800}"  # 23 h 50 min by default.

if ! [[ "${DP_MAX_RUNTIME_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "DP_MAX_RUNTIME_SECONDS must be a positive integer: ${DP_MAX_RUNTIME_SECONDS}" >&2
  exit 2
fi

if [[ ! -f "${DP_CHECKPOINT}" ]]; then
  echo "Checkpoint does not exist: ${DP_CHECKPOINT}" >&2
  exit 2
fi

if [[ -e "${DP_OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output: ${DP_OUTPUT_DIR}" >&2
  exit 2
fi

extra_args=(--n-envs "${DP_N_ENVS}")
if [[ -n "${DP_N_TRAIN:-}" ]]; then
  extra_args+=(--n-train "${DP_N_TRAIN}")
fi
if [[ -n "${DP_N_TEST:-}" ]]; then
  extra_args+=(--n-test "${DP_N_TEST}")
fi
if [[ -n "${DP_SEED:-}" ]]; then
  extra_args+=(--seed "${DP_SEED}")
fi
if [[ "${DP_TRACE_SHAPES:-0}" == "1" ]]; then
  extra_args+=(--trace-shapes)
fi
if [[ -n "${DP_NUM_INFERENCE_STEPS:-}" ]]; then
  extra_args+=(--num-inference-steps "${DP_NUM_INFERENCE_STEPS}")
fi

mkdir -p "$(dirname "${DP_LOG_PATH}")"

docker run --rm --init \
  --name "zty-dp-${DP_RUN_NAME}" \
  --entrypoint /usr/bin/timeout \
  --gpus "device=${DP_GPU}" \
  --cpus 6 \
  --memory 10g \
  --shm-size 1g \
  --pids-limit 256 \
  --network none \
  --user "$(id -u):$(id -g)" \
  --workdir /workspace/code \
  --env PYTHONPATH=/workspace/code \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env PYGAME_HIDE_SUPPORT_PROMPT=1 \
  --env SDL_VIDEODRIVER=dummy \
  --env WANDB_MODE=offline \
  --env WANDB_DIR=/workspace/storage/cache/wandb \
  --env XDG_CACHE_HOME=/workspace/storage/cache \
  --env OMP_NUM_THREADS=6 \
  --env MKL_NUM_THREADS=6 \
  --env OPENBLAS_NUM_THREADS=6 \
  --env NUMEXPR_NUM_THREADS=6 \
  --mount "type=bind,src=${DP_CODE_ROOT},dst=/workspace/code,readonly" \
  --mount "type=bind,src=${DP_STORAGE_ROOT},dst=/workspace/storage" \
  "${DP_IMAGE}" \
  --foreground --signal=TERM --kill-after=600 "${DP_MAX_RUNTIME_SECONDS}s" \
  python eval.py \
    --checkpoint /workspace/storage/checkpoints/"$(basename "${DP_CHECKPOINT}")" \
    --output_dir /workspace/storage/runs/"$(basename "${DP_OUTPUT_DIR}")" \
    --device cuda:0 \
    "${extra_args[@]}" \
  2>&1 | tee "${DP_LOG_PATH}"
