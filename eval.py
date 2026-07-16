"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import random
import os
import pathlib
import click
import hydra
import numpy as np
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.eval_util import apply_eval_overrides


def seed_everything(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def install_shape_trace(policy):
    """Print shapes for the first policy call without logging full tensors."""
    original_predict_action = policy.predict_action
    has_traced = False

    def traced_predict_action(obs_dict):
        nonlocal has_traced
        result = original_predict_action(obs_dict)
        if not has_traced:
            input_shapes = {
                key: list(value.shape)
                for key, value in obs_dict.items()
                if hasattr(value, 'shape')
            }
            output_shapes = {
                key: list(value.shape)
                for key, value in result.items()
                if hasattr(value, 'shape')
            }
            action = result.get('action')
            action_range = None
            if action is not None and action.numel() > 0:
                action_range = {
                    'min': float(action.detach().min().cpu()),
                    'max': float(action.detach().max().cpu())
                }
            batch_size = next(iter(input_shapes.values()))[0]
            trace = {
                'input_shapes': input_shapes,
                'inferred_noise_trajectory_shape': [
                    batch_size,
                    int(policy.horizon),
                    int(policy.action_dim)
                ],
                'num_inference_steps': int(policy.num_inference_steps),
                'output_shapes': output_shapes,
                'action_range': action_range
            }
            click.echo('[shape-trace] ' + json.dumps(trace, sort_keys=True))
            has_traced = True
        return result

    policy.predict_action = traced_predict_action

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--num-inference-steps', type=click.IntRange(min=1), default=None,
    help='Override the checkpoint inference denoising steps.')
@click.option('--n-envs', type=click.IntRange(min=1), default=None,
    help='Override the number of parallel rollout environments.')
@click.option('--n-train', type=click.IntRange(min=0), default=None,
    help='Override the number of training-seed rollouts.')
@click.option('--n-test', type=click.IntRange(min=0), default=None,
    help='Override the number of test-seed rollouts.')
@click.option('--seed', type=int, default=None,
    help='Seed Python, NumPy, PyTorch, and CUDA for this evaluation.')
@click.option('--trace-shapes', is_flag=True,
    help='Print tensor shapes and action range for the first policy call.')
def main(checkpoint, output_dir, device, num_inference_steps, n_envs,
        n_train, n_test, seed, trace_shapes):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    try:
        run_metadata = apply_eval_overrides(
            cfg=cfg,
            policy=policy,
            num_inference_steps=num_inference_steps,
            n_envs=n_envs,
            n_train=n_train,
            n_test=n_test
        )
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc

    device = torch.device(device)
    policy.to(device)
    policy.eval()
    seed_everything(seed)
    if trace_shapes:
        install_shape_trace(policy)
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    json_log['repro/seed'] = seed
    json_log['repro/trace_shapes'] = trace_shapes
    for key, value in run_metadata.items():
        json_log[f'repro/{key}'] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
