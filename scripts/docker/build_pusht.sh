#!/usr/bin/env bash
set -Eeuo pipefail

# This script only builds the reviewed Push-T image. It never pulls a newer
# base image automatically, and it never starts an evaluation or training run.
DP_IMAGE="${DP_IMAGE:-zty/diffusion-policy-pusht:torch1.12-cu116}"

docker build \
  --pull=false \
  --file docker/Dockerfile.pusht \
  --tag "${DP_IMAGE}" \
  .
