#!/usr/bin/env bash
set -Eeuo pipefail

# This script runs inside either a Docker build step or the resource-limited
# bootstrap container. It never runs on the Ubuntu host.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export DEBIAN_FRONTEND=noninteractive

# These settings apply only to this container layer. They never alter the
# shared Ubuntu host, Docker daemon, or another user's container.
UBUNTU_MIRROR="${DP_UBUNTU_MIRROR:-https://mirrors.tuna.tsinghua.edu.cn/ubuntu/}"
PIP_INDEX_URL="${DP_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_INDEX_URL

sed -i \
  -e "s|http://archive.ubuntu.com/ubuntu/|${UBUNTU_MIRROR}|g" \
  -e "s|http://security.ubuntu.com/ubuntu/|${UBUNTU_MIRROR}|g" \
  /etc/apt/sources.list

# The CUDA runtime is already part of the base image. This Push-T setup does
# not install CUDA packages through apt, so skip the unrelated NVIDIA apt
# index during the bootstrap download.
if [[ -f /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list ]]; then
  mv /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list \
    /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list.disabled
fi

apt-get update
apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  ffmpeg \
  git \
  libavcodec-dev \
  libavdevice-dev \
  libavfilter-dev \
  libavformat-dev \
  libavutil-dev \
  libgl1 \
  libglib2.0-0 \
  libgomp1 \
  libswresample-dev \
  libswscale-dev \
  pkg-config \
  python3 \
  python3-dev \
  python3-pip \
  python3-venv
rm -rf /var/lib/apt/lists/*

python3 -m venv /opt/dp-venv
/opt/dp-venv/bin/python -m pip install --no-cache-dir \
  'pip<23' \
  'setuptools<66' \
  'wheel<0.39'
/opt/dp-venv/bin/python -m pip install --no-cache-dir \
  'Cython==0.29.36'
/opt/dp-venv/bin/python -m pip install --no-cache-dir \
  --no-build-isolation \
  -r "${SCRIPT_DIR}/requirements-pusht.txt"
