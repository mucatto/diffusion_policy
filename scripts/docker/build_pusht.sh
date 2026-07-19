#!/usr/bin/env bash
set -Eeuo pipefail

# Build a derived image through a resource-limited bootstrap container instead
# of an unconstrained `docker build` invocation.
DP_CODE_ROOT="${DP_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DP_STORAGE_ROOT="${DP_STORAGE_ROOT:-/groups/2/sk/diffusion_policy}"
DP_BASE_IMAGE="${DP_BASE_IMAGE:-nvidia/cuda:12.4.0-base-ubuntu22.04}"
DP_IMAGE="${DP_IMAGE:-zty/diffusion-policy-pusht:torch1.12-cu116}"
DP_BOOTSTRAP_NAME="${DP_BOOTSTRAP_NAME:-zty-dp-pusht-bootstrap-$(date +%Y%m%d-%H%M%S)}"
DP_LOG_PATH="${DP_LOG_PATH:-${DP_STORAGE_ROOT}/runs/${DP_BOOTSTRAP_NAME}.log}"
DP_BOOTSTRAP_TIMEOUT_SECONDS=7200

if [[ ! -f "${DP_CODE_ROOT}/docker/install_pusht_env.sh" ]]; then
  echo "Missing installer script under: ${DP_CODE_ROOT}" >&2
  exit 2
fi
if ! docker image inspect "${DP_BASE_IMAGE}" >/dev/null 2>&1; then
  echo "Base image is not cached locally: ${DP_BASE_IMAGE}" >&2
  exit 2
fi
if docker image inspect "${DP_IMAGE}" >/dev/null 2>&1; then
  echo "Refusing to overwrite existing image: ${DP_IMAGE}" >&2
  exit 2
fi
if docker container inspect "${DP_BOOTSTRAP_NAME}" >/dev/null 2>&1; then
  echo "Refusing to reuse existing container: ${DP_BOOTSTRAP_NAME}" >&2
  exit 2
fi

mkdir -p "$(dirname "${DP_LOG_PATH}")"

set +e
docker run --name "${DP_BOOTSTRAP_NAME}" --init \
  --cpus 6 \
  --memory 10g \
  --shm-size 1g \
  --pids-limit 256 \
  --network bridge \
  --env NVIDIA_VISIBLE_DEVICES=void \
  --mount "type=bind,src=${DP_CODE_ROOT},dst=/workspace/code,readonly" \
  --workdir /workspace/code \
  "${DP_BASE_IMAGE}" \
  timeout --foreground --signal=TERM --kill-after=600 "${DP_BOOTSTRAP_TIMEOUT_SECONDS}s" \
  bash /workspace/code/docker/install_pusht_env.sh \
  2>&1 | tee "${DP_LOG_PATH}"
bootstrap_status=${PIPESTATUS[0]}
set -e

if [[ ${bootstrap_status} -ne 0 ]]; then
  echo "Bootstrap failed; stopped container retained for inspection: ${DP_BOOTSTRAP_NAME}" >&2
  exit "${bootstrap_status}"
fi

docker commit \
  --change 'ENV PATH=/opt/dp-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' \
  --change 'ENV PYTHONUNBUFFERED=1' \
  --change 'ENV PYTHONDONTWRITEBYTECODE=1' \
  --change 'ENV PYGAME_HIDE_SUPPORT_PROMPT=1' \
  --change 'ENV SDL_VIDEODRIVER=dummy' \
  --change 'WORKDIR /workspace/code' \
  --change 'CMD ["/bin/bash"]' \
  "${DP_BOOTSTRAP_NAME}" "${DP_IMAGE}"

docker image inspect "${DP_IMAGE}"
docker rm "${DP_BOOTSTRAP_NAME}"
