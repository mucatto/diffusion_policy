# Docker layout for the Push-T reproduction

## Persistent host paths

| Host path | Container path | Purpose |
| --- | --- | --- |
| `/home/zty_group/sk/diffusion_policy` | `/workspace/code` (read-only) | Git checkout, Docker files, scripts, and documentation |
| `/groups/2/sk/diffusion_policy` | `/workspace/storage` (read-write) | Data, checkpoints, logs, videos, and cache |

The container is ephemeral: evaluation uses `--rm`, so the container itself is
removed when its foreground process exits. The two bind mounts persist.

## Storage policy

```text
/groups/2/sk/diffusion_policy/
- data/         # downloaded datasets
- checkpoints/  # public and locally trained checkpoints
- runs/         # logs, metrics, videos, and output directories
- cache/        # package and application caches
- tmp/          # recreatable temporary artifacts
```

Do not put datasets, checkpoints, videos, `runs/`, or cache directories into
Git or Docker image layers.

## Resource policy

- One GPU, selected only after a fresh `nvidia-smi` check.
- `--cpus 6`, `--memory 10g`, `--shm-size 1g`, and `--pids-limit 256`.
- Push-T evaluation defaults to four parallel environments. OpenMP, MKL,
  OpenBLAS, and NumExpr are each capped at six threads inside the container.
- Evaluation runs in the foreground with `--rm`; it cannot leave a stopped
  container behind.
- `timeout` inside the container terminates the evaluation after 23 hours and
  50 minutes, then escalates to `SIGKILL` after ten more minutes if needed.
  Longer jobs require an explicitly reviewed script change and group reporting
  when applicable.
- `--network none` is used for evaluation. Image builds and explicit downloads
  are reviewed separately because they require network access.

## Image policy

`docker/Dockerfile.pusht` remains the reproducible recipe. The default image
creation entry point is `scripts/docker/build_pusht.sh`, which starts a
resource-limited bootstrap container from the already cached
`nvidia/cuda:12.4.0-base-ubuntu22.04` image. Inside that container,
`docker/install_pusht_env.sh` installs the minimal Push-T environment, then
`docker commit` saves the stopped container as
`zty/diffusion-policy-pusht:torch1.12-cu116`.

The bootstrap container has no GPU access, six CPU cores, 10 GiB memory, a
one GiB shared-memory limit, and a two-hour timeout. It uses the network only
for explicit package installation. On a successful commit, the bootstrap
container is removed. On failure, it remains stopped for inspection and its
log is stored under `/groups/2/sk/diffusion_policy/runs/`.

The evaluation wrapper defaults to a 23-hour-50-minute timeout, but accepts a
positive `DP_MAX_RUNTIME_SECONDS` override. Short smoke evaluations must set a
reviewed lower cap (for example, 3600 seconds) rather than relying on the
long default.

During this container-only installation, Ubuntu package URLs are redirected to
the Tsinghua Ubuntu mirror and ordinary Python packages use the Tsinghua PyPI
mirror. The pinned CUDA 11.6 PyTorch wheels remain sourced from the official
PyTorch index declared in `requirements-pusht.txt`. The base image's unused
NVIDIA CUDA apt source is disabled only inside the image layer, so
`apt-get update` does not make an unrelated NVIDIA repository request. None
of these settings modify the Ubuntu host, Docker daemon, or other containers.

The shared CUDA base layer is reused; only the minimal Push-T Python
environment adds new image layers under `/var/lib/docker`.

## Image inventory and lifecycle

| Tag | Status | Purpose |
| --- | --- | --- |
| `nvidia/cuda:12.4.0-base-ubuntu22.04` | shared base | Cached NVIDIA CUDA base; never prune or modify for this project. |
| `zty/diffusion-policy-pusht:torch1.12-cu116` | retained baseline | Initial Push-T image; retained because its unconstrained Hugging Face Hub dependency was incompatible with Diffusers 0.11.1. |
| `zty/diffusion-policy-pusht:torch1.12-cu116-hf0121` | repair candidate | Evaluation image with `huggingface-hub==0.12.1`; it passed the import and dependency checks and awaits a successful official evaluation. |

The repaired image has Docker labels beginning with `research.` that record
the project, task, purpose, base image, repair, and status. Tags do not imply
that each image consumes its displayed size independently: Docker shares the
unchanged base layers. The repair's writable container layer was approximately
9 MiB before commit.

Do not use global cleanup commands such as `docker system prune`. If an image
is later obsolete, first review its tag, labels, size, parent relationship, and
whether any current wrapper references it; deletion still requires explicit
user approval.

This first image intentionally supports only Push-T low-dim evaluation and a
short low-dim training run. MuJoCo, robosuite, PyTorch3D, Jetson deployment,
real robots, and image tasks are deferred to separately reviewed environments.
