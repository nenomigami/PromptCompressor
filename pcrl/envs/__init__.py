import os
from typing import Any, Callable, Dict, Optional, Type, Union, List

import gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv
from data.instruction_pool import Sample

def make_vec_env(
    env_id: gym.Env,
    n_envs: int,
    samples: List[Sample],
    sample_k: int = 1,
    seed: Optional[int] = None,
    monitor_dir: Optional[str] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    subset = []
    assert n_envs % sample_k == 0, "n_envs must be divisible by k"
    sample_per_env = len(samples)//(n_envs//sample_k)
    for i in range(n_envs//sample_k):
        subset.append(samples[sample_per_env*i : sample_per_env*(i+1)])
    env_kwargs = {} if env_kwargs is None else env_kwargs
    monitor_kwargs = {} if monitor_kwargs is None else monitor_kwargs
    def make_env(rank):
        def _init():
            env = env_id(
                rank=rank,
                samples=subset[rank//sample_k],
                **env_kwargs
            )
            if seed is not None:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            return env

        return _init

    return SubprocVecEnv([make_env(i) for i in range(n_envs)])