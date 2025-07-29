import math
import os
from typing import Optional

import imageio
import wandb
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune import RunConfig, TuneConfig

from EnvironmentClass import AerialBattle
import numpy as np
import yaml
import torch
import ray
from ray import tune
from ray.rllib.algorithms.sac import SACConfig
from ray.tune.registry import register_env

#TODO: configure the self_play routine for team vs team

# === Configuration Paths ===
Folder = 'Training_Runs'
RunName = 'Train_1_GOODFLIGHT_2'
RunDescription = 'Explore different entropy parameters for fixed gamma=0.999,\n' \
                'train_batch_size=500. explore slight variations on best reward version from previous step.\n' \
                'Added a control effort component to discourage excessive manouvres if not necessary'

ConfigFile = 'Train_Run_config.yaml'
Base_Checkpoint = ''
Base_Policy_restore = ['team_0']  # Policies to restore from checkpoint

# Path where training data and checkpoints will be stored
storage_path = os.path.join("/home/lsp/Desktop/ThesisCode", Folder)

# Set your WandB API key for logging
os.environ["WANDB_API_KEY"] = "1b8b77cc6fc3631890702b9ecbfed2fdc1551347"


# === Custom Callbacks ===
# Save videos, trajectories, and telemetry plots on each checkpoint
class SaveArtifactsOnCheckpoint(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        if algorithm.iteration % alg_config['checkpoint_freq'] == 0:
            trial_name = os.path.basename(algorithm._logdir)
            trial_dir = os.path.join(storage_path, RunName, trial_name)
            checkpoint_dir = os.path.join(trial_dir, f"checkpoint_{result['training_iteration']-1}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            env = algorithm.env_creator({'reward_version': 1})  # Must be num_env_runners = 1

            for i in range(2):  # Save 5 rollouts
                checkpoint_dir_i = os.path.join(checkpoint_dir, f"{i}")
                os.makedirs(checkpoint_dir_i, exist_ok=True)

                obs, _ = env.reset(testing=True)
                terminated = {"__all__": False}
                frames = []

                print(f"Evaluating...")

                while not terminated["__all__"] and len(frames) < alg_config['checkpoint_length']:
                    actions = {}
                    for agent_id, agent_obs in obs.items():
                        policy_id = policy_mapping_fn(agent_id)
                        policy = algorithm.get_policy(policy_id)
                        action, _, _ = policy.compute_single_action(agent_obs, explore=False)
                        actions[agent_id] = action

                    obs, _, terminated, truncated, _ = env.step(actions)
                    terminated["__all__"] = terminated["__all__"] or truncated["__all__"]

                    if hasattr(env, "render"):
                        frames.append(env.render(mode="rgb_array"))

                # Save episode video
                video_path = os.path.join(checkpoint_dir_i, f"episode_video.mp4")
                imageio.mimsave(video_path, frames, fps=10)

                # Save additional visualizations
                if hasattr(env, "render_trajectory"):
                    env.render_trajectory(checkpoint_dir_i)
                if hasattr(env, "plot_telemetry"):
                    env.plot_telemetry(checkpoint_dir_i)
                if hasattr(env, "plot_rewards"):
                    env.plot_rewards(checkpoint_dir_i)

            print(f"Finished Checkpoint at {checkpoint_dir}")


# Log metrics to Weights & Biases (WandB)
class CustomWandbCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.initialized = False

    def on_train_result(self, *, algorithm, result, **kwargs):
        trial_name = os.path.basename(algorithm._logdir)
        super().on_train_result(algorithm=algorithm, result=result, **kwargs)

        # Initialize WandB (only once per worker)
        if not self.initialized:
            wandb.init(
                project="aerial-battle",
                group=f"{RunName}",
                name=f'{RunName}/{trial_name}',
                config=algorithm.config,
                mode="online"
            )
            self.initialized = True

        # Log selected metrics
        metrics = {}
        env_metrics = result.get("env_runners", {})
        metrics["reward_mean"] = env_metrics.get("episode_reward_mean")
        metrics["reward_max"] = env_metrics.get("episode_reward_max")
        metrics["reward_min"] = env_metrics.get("episode_reward_min")
        metrics["episode_len_mean"] = env_metrics.get("episode_len_mean")
        metrics["cont_lane_time_mean"] = env_metrics["custom_metrics"].get("cont_lane_time_mean", 0)
        metrics["cont_lane_time_max"] = env_metrics["custom_metrics"].get("cont_lane_time_max", 0)

        learner_stats = result.get("info", {}).get("learner", {}).get("team_0", {}).get("learner_stats", {})
        for key in ["alpha_value", "actor_loss", "critic_loss", "target_entropy"]:
            if key in learner_stats:
                metrics[key] = learner_stats[key]

        # Filter out NaNs or None
        metrics = {
            k: float(v) for k, v in metrics.items()
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        }

        wandb.log(metrics, step=result['training_iteration'])


# Broker to combine multiple callbacks and restore from a base checkpoint
class CallbacksBroker(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.Artifacts = SaveArtifactsOnCheckpoint()
        self.WandbCallBack = CustomWandbCallback()
    
    def on_algorithm_init(self, *, algorithm, metrics_logger=None, **kwargs):
        if Base_Checkpoint:
            print("\n++++++++++++++++++++++ Loading Checkpoint +++++++++++++++++++++\n")
            checkpoint_path = os.path.join(storage_path, RunName, Base_Checkpoint)
            for id in Base_Policy_restore:
                restored = algorithm.get_policy(id).from_checkpoint(checkpoint_path)[id]
                algorithm.get_policy(id).set_state(restored.get_state())
                print(f"Loaded policy {id}")
            print("\n++++++++++++++++++++++ Checkpoint Loaded +++++++++++++++++++++\n")

    def on_train_result(self, *, algorithm, result, **kwargs):
        self.Artifacts.on_train_result(algorithm=algorithm, result=result, **kwargs)
        self.WandbCallBack.on_train_result(algorithm=algorithm, result=result, **kwargs)
    
    def on_episode_step(self, *, episode, **kwargs):
        common_info = episode._last_infos.get("__common__", {})
        lane_metric = common_info.get("lane_time", None)
        if lane_metric is not None:
            episode.custom_metrics["cont_lane_time"] = (lane_metric)
    


#Create RunName directory inside Folder, with RunDescription inside it
# === Create Directory and Save Description ===
run_path = os.path.join(Folder, RunName)
os.makedirs(run_path, exist_ok=True)

description_path = os.path.join(run_path, "description.txt")
with open(description_path, "w") as f:
    f.write(RunDescription)

print(f"Created run directory and saved description at: {description_path}")

# === Load YAML experiment configuration ===
with open(ConfigFile) as f:
    yaml_config = yaml.load(f, Loader=yaml.FullLoader)

alg_config = yaml_config['alg_config']
env_config = yaml_config['env_config']
uav_config = yaml_config['uav_config']

# === Environment Registration ===
def env_creator(cfg):
    return AerialBattle(env_config, uav_config, cfg['reward_version'], discretize=True)
register_env("aerial_battle", env_creator)


# Dummy env for observation/action space extraction
dummy_env = AerialBattle(env_config=env_config, UAV_config=uav_config)
obs_space = dummy_env.get_observation_space('agent_0_0')
act_space = dummy_env.get_action_space("agent_1_0")
dummy_env.close()

# === Define Multi-Agent Policy Specs ===
policies = {
    "team_0": (None, obs_space, act_space, {
        "model": {"fcnet_hiddens": [512, 512], "fcnet_activation": 'tanh'},
    }),
    "team_1": (None, obs_space, act_space, {
        "model": {"fcnet_hiddens": [512, 512], "fcnet_activation": 'tanh'},
    }),
}

# Policy assignment function
def policy_mapping_fn(agent_id, episode=0, **kwargs):
    if agent_id.startswith("agent_0"):
        return "team_0"
    if agent_id.startswith("agent_1"):
        return "team_1"


# === Algorithm Configuration ===
algo_config = (
    SACConfig()
    .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    .environment(env="aerial_battle", env_config={'reward_version': tune.grid_search([1,2,3,4,5,6,7,8])})
    .training(
        train_batch_size=tune.grid_search(alg_config['batch_size_per_learner']),
        gamma=tune.grid_search(alg_config['gamma']),

        twin_q=True,
        #initial_alpha=2,
        #n_step=3,
        actor_lr=alg_config['lr'],
        critic_lr = 0.0001,
        alpha_lr = 0.0001,
        grad_clip=50,
        replay_buffer_config={
            'type': 'MultiAgentPrioritizedReplayBuffer',
            'prioritized_replay_alpha': 0.6,
            'prioritized_replay_beta': 0.4,
            'prioritized_replay_eps': 1e-6,
        }
    )
    .env_runners(
        num_env_runners=15,
        num_envs_per_env_runner=2,
        num_cpus_per_env_runner=1,
        num_gpus_per_env_runner=0.065,
        batch_mode="truncate_episodes",
        sample_timeout_s=120
    )
    .multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=['team_0'],  # Only train team_0
    )
    .callbacks(CallbacksBroker)  # Combined callbacks
)


# === Ray Tune Tuner Configuration ===
tuner = tune.Tuner(
    trainable="SAC",
    param_space=algo_config,
    tune_config=TuneConfig(
        trial_name_creator=lambda trial: f"trial_{trial.trial_id[:10]}",
        trial_dirname_creator=lambda trial: f"trial_{trial.trial_id[:10]}"
    ),
    run_config=RunConfig(
        name=RunName,
        storage_path=storage_path,
        stop={"training_iteration": alg_config['train_iterations']},
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_at_end=True,
            checkpoint_frequency=alg_config['checkpoint_freq'],
        ),
        failure_config=tune.FailureConfig(max_failures=2),
    )
)

# === Run the Experiment ===
tuner.fit()