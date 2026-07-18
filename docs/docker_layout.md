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

`docker/Dockerfile.pusht` derives from the already cached
`nvidia/cuda:12.4.0-base-ubuntu22.04` image. The base layer is shared; only
the minimal Push-T Python environment adds new image layers under
`/var/lib/docker`.

This first image intentionally supports only Push-T low-dim evaluation and a
short low-dim training run. MuJoCo, robosuite, PyTorch3D, Jetson deployment,
real robots, and image tasks are deferred to separately reviewed environments.
