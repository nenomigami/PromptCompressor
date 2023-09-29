from typing import Any, Dict, List, Tuple, Type, Optional

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.type_aliases import TensorDict, MaybeCallback
from stable_baselines3.common.vec_env import VecEnv
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from data.instruction_pool import get_fixed_token_counts
from pcrl.algorithms.common.buffers import MaskableDictRolloutBuffer
from pcrl.algorithms.common.utils import get_action_masks
from pcrl.envs.reward import RewardFunction, extract_values
from pcrl.utils.evaluation_utils import generate
from pcrl.utils.logging_utils import Tracker
from pcrl.utils.metric import BaseMetric
import numpy as np
import torch

def wrap_onpolicy_alg(
    alg_class: Type[OnPolicyAlgorithm],
    alg_kwargs: Dict[str, Any],
    gen_kwargs: Dict[str, Any],
    reward_fn: List[RewardFunction],
    metrics: List[BaseMetric],
    tasks: Dict[str, Any],
    tracker: Tracker,
):
    class OnPolicyAlgText(alg_class):
        def __init__(
            self,
            alg_kwargs: Dict[str, Any],
            gen_kwargs: Dict[str, Any],
            reward_fn: List[RewardFunction],
            metrics: List[BaseMetric],
            tasks: Dict[str, Any],
            tracker: Tracker,
        ):
            from pcrl.utils.training_utils import build_tokenizer
            self.use_scst = alg_kwargs.pop("use_scst")
            alg_kwargs['policy_kwargs']['use_scst'] = self.use_scst
            alg_kwargs['policy_kwargs']['sample_k'] = alg_kwargs.pop("sample_k")
            alg_kwargs["tracker"] = tracker
            super().__init__(**alg_kwargs)
            self.gen_tokenizer = build_tokenizer(gen_kwargs['tokenizer'])
            model_cls = AutoModelForSeq2SeqLM if "t5" in gen_kwargs['model_name'] else AutoModelForCausalLM
            self.gen_model = model_cls.from_pretrained(
                gen_kwargs['model_name'], device_map="auto").eval()
            self.gen_kwargs = gen_kwargs.get("generation_kwargs", {})
            self.tracker = tracker
            self.tasks = tasks

            self.fixed_token_counts = get_fixed_token_counts(self.gen_tokenizer)
            self._reward_fn= reward_fn

        def _init_callback(
            self,
            callback: MaybeCallback,
            eval_freq: int = 10000,
            n_eval_episodes: int = 5,
            log_path: Optional[str] = None,
            use_masking: bool = True,
        ) -> BaseCallback:
            # Convert a list of callbacks into a callback
            if isinstance(callback, list):
                callback = CallbackList(callback)

            # Convert functional callback to object
            if not isinstance(callback, BaseCallback):
                callback = ConvertCallback(callback)

            # Create eval callback in charge of the evaluation
            # from pcrl.algorithms.common.callbacks import MaskableEvalCallback

            # eval_callback = MaskableEvalCallback(
            #     self.gen_model,
            #     self.gen_tokenizer,
            #     self.gen_kwargs,
            #     eval_dataset=self.tasks,
            #     metrics=metrics,
            #     tracker=self.tracker,
            #     best_model_save_path=log_path,
            #     eval_freq=eval_freq,
            #     n_eval_episodes=n_eval_episodes,
            #     log_path=log_path,
            #     use_masking=use_masking,
            # )
            # callback = CallbackList([callback, eval_callback])

            callback.init_callback(self)
            return callback

        def compute_scst_rewards(self, env: VecEnv, obs_tensor: TensorDict, action_masks: np.ndarray):
            with torch.no_grad():
                argmax_actions, _, _ = self.policy(obs_tensor, deterministic=True, action_masks=action_masks)
            argmax_actions = argmax_actions.cpu().numpy()
            results = env.env_method("fake_step", argmax_actions)
            _, _, _, argmax_infos = zip(*results)

            sample_k = self.sample_k
            assert len(argmax_infos) % self.sample_k == 0, "sample_k should be a divisor of n_envs"
            kth_argmax_infos = argmax_infos[::sample_k]
            
            kth_gen_output = generate(
                extract_values(kth_argmax_infos, "compressed_prompt"), 
                self.gen_model,
                self.gen_tokenizer, 
                self.gen_kwargs,
            )
            argmax_rewards, _ = self._reward_fn(kth_argmax_infos, kth_gen_output, self.fixed_token_counts)
            argmax_rewards = np.repeat(argmax_rewards, sample_k, axis=0)
            return argmax_rewards

        def _update_info_rewards(self, infos: List[Dict[str, Any]], rewards: List[float], rewards_dict: Dict[str, List[float]]):
            for info, reward in zip(infos, rewards):
                info["episode"]["r"] = reward
                info["episode"]["r_comp"] = rewards_dict['comp']
                info["episode"]["r_sim"] = rewards_dict['sim']

        def _update_info_token_counts(self, infos: List[Dict[str, Any]], token_counts: List[int]):
            for info, token_count in zip(infos, token_counts):
                info["compressed_token_count"] = token_count

        def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: RolloutBuffer,
            n_rollout_steps: int,
            use_masking: bool = True,
        ) -> bool:
                
            assert isinstance(
                rollout_buffer, MaskableDictRolloutBuffer
            ), "RolloutBuffer(MaskableRolloutBuffer) doesn't support"
            assert self._last_obs is not None, "No previous observation was provided"

            # get tokenizer
            tokenizer = env.unwrapped.get_attr("tokenizer", [0])
            tokenizer = tokenizer[0]

            # Switch to eval mode
            self.policy.set_training_mode(False)

            n_steps = 0
            action_masks = None
            # reset rollout buffer and stats
            rollout_buffer.reset()
            argmax_rewards = np.zeros_like(rollout_buffer.rewards)

            callback.on_rollout_start()

            while not rollout_buffer.full:
                with torch.no_grad():
                    # Convert to pytorch tensor or to TensorDict
                    obs_tensor = obs_as_tensor(self._last_obs, self.device)
                    # This is the only change related to invalid action masking
                    if use_masking:
                        action_masks = get_action_masks(env)

                    actions, values, log_probs = self.policy(obs_tensor, action_masks=action_masks)
                    if self.use_scst is True:
                        argmax_rewards[n_steps] = self.compute_scst_rewards(env, obs_tensor, action_masks)
                        
                actions = actions.cpu().numpy()
                
                new_obs, _, dones, infos = env.step(actions)
            
                #all envs should end at the same time.
                assert np.all(dones[0]==dones)
                if np.all(dones):
                    gen_output = generate(
                        extract_values(infos,"compressed_prompt"),
                        self.gen_model,
                        self.gen_tokenizer, 
                        self.gen_kwargs,
                    )
                    rewards, rw_info = self._reward_fn(infos, gen_output, self.fixed_token_counts)

                    self._update_info_token_counts(infos, gen_output["compressed_token_counts"])
                    self._update_info_rewards(infos, rewards, rw_info)
                self.num_timesteps += env.num_envs

                # Give access to local variables
                callback.update_locals(locals())
                if callback.on_step() is False:
                    return False

                self._update_info_buffer(infos)
                n_steps += 1
                
                rollout_buffer.add(
                    self._last_obs,
                    actions,
                    rewards,
                    self._last_episode_starts,
                    values,
                    log_probs,
                    action_masks=action_masks,
                )
                self._last_obs = new_obs
                self._last_episode_starts = dones

            with torch.no_grad():
                values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
            
            rollout_buffer.compute_returns_and_advantage(
                last_values=values, 
                dones=dones, 
                argmax_rewards=argmax_rewards,
                use_scst=self.use_scst
            )

            callback.on_rollout_end()

            return True

    # instantiate the wrapped alg
    alg = OnPolicyAlgText(alg_kwargs, gen_kwargs, reward_fn, metrics, tasks, tracker)
    return alg