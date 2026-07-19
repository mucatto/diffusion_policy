#!/usr/bin/env bash
set -Eeuo pipefail

# This script runs inside either a Docker build step or the resource-limited
# bootstrap container. It never runs on the Ubuntu host.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates \
  ffmpeg \
  git \
  libgl1 \
  libglib2.0-0 \
  libgomp1 \
  python3 \
  python3-pip \
  python3-venv
rm -rf /var/lib/apt/lists/*

python3 -m venv /opt/dp-venv
/opt/dp-venv/bin/python -m pip install --no-cache-dir \
  'pip<23' \
  'setuptools<66' \
  'wheel<0.39'
/opt/dp-venv/bin/python -m pip install --no-cache-dir \
  -r "${SCRIPT_DIR}/requirements-pusht.txt"
