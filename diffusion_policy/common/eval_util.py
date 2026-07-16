def _optional_attr(obj, name):
    return getattr(obj, name, None)


def apply_eval_overrides(cfg, policy, num_inference_steps=None, n_envs=None,
        n_train=None, n_test=None):
    """Apply explicit evaluation overrides and return effective metadata.

    All arguments default to ``None`` so the checkpoint configuration remains
    unchanged when no override is requested.
    """
    runner_cfg = cfg.task.env_runner

    if num_inference_steps is not None:
        if num_inference_steps <= 0:
            raise ValueError('num_inference_steps must be greater than zero')
        train_timesteps = int(
            policy.noise_scheduler.config.num_train_timesteps)
        if num_inference_steps > train_timesteps:
            raise ValueError(
                'num_inference_steps cannot exceed scheduler training '
                f'timesteps ({train_timesteps})')
        policy.num_inference_steps = int(num_inference_steps)

    if n_train is not None:
        if _optional_attr(runner_cfg, 'n_train') is None:
            raise ValueError('this environment runner has no n_train setting')
        if n_train < 0:
            raise ValueError('n_train cannot be negative')
        runner_cfg.n_train = int(n_train)

    if n_test is not None:
        if _optional_attr(runner_cfg, 'n_test') is None:
            raise ValueError('this environment runner has no n_test setting')
        if n_test < 0:
            raise ValueError('n_test cannot be negative')
        runner_cfg.n_test = int(n_test)

    effective_n_train = _optional_attr(runner_cfg, 'n_train')
    effective_n_test = _optional_attr(runner_cfg, 'n_test')
    total_rollouts = None
    if effective_n_train is not None and effective_n_test is not None:
        total_rollouts = int(effective_n_train) + int(effective_n_test)
        if (n_train is not None or n_test is not None) and total_rollouts <= 0:
            raise ValueError('n_train and n_test cannot both be zero')

    if n_envs is not None:
        if _optional_attr(runner_cfg, 'n_envs') is None and not hasattr(
                runner_cfg, 'n_envs'):
            raise ValueError('this environment runner has no n_envs setting')
        if n_envs <= 0:
            raise ValueError('n_envs must be greater than zero')
        if total_rollouts is not None and n_envs > total_rollouts:
            raise ValueError(
                f'n_envs ({n_envs}) cannot exceed total rollouts '
                f'({total_rollouts})')
        runner_cfg.n_envs = int(n_envs)

    return {
        'num_inference_steps': int(policy.num_inference_steps),
        'n_envs': _optional_attr(runner_cfg, 'n_envs'),
        'n_train': effective_n_train,
        'n_test': effective_n_test
    }
