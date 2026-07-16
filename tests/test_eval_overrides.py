import unittest
from types import SimpleNamespace

from diffusion_policy.common.eval_util import apply_eval_overrides


def make_config(n_train=6, n_test=50, n_envs=None):
    runner = SimpleNamespace(
        n_train=n_train,
        n_test=n_test,
        n_envs=n_envs
    )
    return SimpleNamespace(
        task=SimpleNamespace(env_runner=runner)
    )


def make_policy(num_inference_steps=100, train_timesteps=100):
    scheduler = SimpleNamespace(
        config=SimpleNamespace(num_train_timesteps=train_timesteps)
    )
    return SimpleNamespace(
        num_inference_steps=num_inference_steps,
        noise_scheduler=scheduler
    )


class ApplyEvalOverridesTest(unittest.TestCase):
    def test_defaults_preserve_checkpoint_configuration(self):
        cfg = make_config()
        policy = make_policy()

        metadata = apply_eval_overrides(cfg, policy)

        self.assertEqual(policy.num_inference_steps, 100)
        self.assertIsNone(cfg.task.env_runner.n_envs)
        self.assertEqual(cfg.task.env_runner.n_train, 6)
        self.assertEqual(cfg.task.env_runner.n_test, 50)
        self.assertEqual(metadata['num_inference_steps'], 100)

    def test_applies_explicit_overrides(self):
        cfg = make_config()
        policy = make_policy()

        metadata = apply_eval_overrides(
            cfg,
            policy,
            num_inference_steps=10,
            n_envs=6,
            n_train=0,
            n_test=8
        )

        self.assertEqual(policy.num_inference_steps, 10)
        self.assertEqual(cfg.task.env_runner.n_envs, 6)
        self.assertEqual(cfg.task.env_runner.n_train, 0)
        self.assertEqual(cfg.task.env_runner.n_test, 8)
        self.assertEqual(metadata['n_envs'], 6)

    def test_rejects_non_positive_inference_steps(self):
        with self.assertRaisesRegex(ValueError, 'greater than zero'):
            apply_eval_overrides(
                make_config(), make_policy(), num_inference_steps=0)

    def test_rejects_inference_steps_above_training_timesteps(self):
        with self.assertRaisesRegex(ValueError, 'cannot exceed'):
            apply_eval_overrides(
                make_config(), make_policy(), num_inference_steps=101)

    def test_rejects_negative_train_rollouts(self):
        with self.assertRaisesRegex(ValueError, 'cannot be negative'):
            apply_eval_overrides(make_config(), make_policy(), n_train=-1)

    def test_rejects_negative_test_rollouts(self):
        with self.assertRaisesRegex(ValueError, 'cannot be negative'):
            apply_eval_overrides(make_config(), make_policy(), n_test=-1)

    def test_rejects_zero_total_rollouts(self):
        with self.assertRaisesRegex(ValueError, 'cannot both be zero'):
            apply_eval_overrides(
                make_config(), make_policy(), n_train=0, n_test=0)

    def test_rejects_more_environments_than_rollouts(self):
        with self.assertRaisesRegex(ValueError, 'cannot exceed'):
            apply_eval_overrides(
                make_config(), make_policy(), n_envs=9, n_train=0, n_test=8)


if __name__ == '__main__':
    unittest.main()
