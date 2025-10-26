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
import trueskill as ts


# === Configuration Paths ===
Folder = 'Training_Runs'
RunName = 'FinalTrain_Pursuit_Shaping1'
RunDescription = "New set of trainings to refine reward, overall implementation and performance."

ConfigFile = 'Train_Run_config.yaml'
Base_Checkpoint = ''
Adversary_Base_Checkpoint = ''
Base_Policy_restore = []  # Policies to restore from checkpoint
policies_to_train=['team_0']

Checkpoint_Window = {} #{checkpoint_000004/team_0 : Rating(), ...}
Current_Pair = {}

# Path where training data and checkpoints will be stored
storage_path = os.path.join(os.getcwd(), Folder)

# Set your WandB API key for logging
os.environ["WANDB_API_KEY"] = "1b8b77cc6fc3631890702b9ecbfed2fdc1551347"
wandb.login(key=os.environ["WANDB_API_KEY"])  # once, at the top of your script


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


# === Custom Callbacks ===
def ExecuteEpisode(env, algorithm, checkpoint_dir, i):
    checkpoint_dir_i = os.path.join(checkpoint_dir, f"{i}")
    os.makedirs(checkpoint_dir_i, exist_ok=True)

    obs, _ = env.reset(testing=True)
    terminated = {"__all__": False}
    frames = []

    print(f"Evaluation round {i}...")

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

    if Adversary_Base_Checkpoint:
        return env.get_winning_team()

def TrueSkill(rank, Current_Pair_rating, Starting_Agents_Number):
    rate_team_0 = [Current_Pair_rating['team_0']] * Starting_Agents_Number
    rate_team_1 = [Current_Pair_rating['team_1']] * Starting_Agents_Number

    (new_rate_team_0,), (new_rate_team_1,) = ts.rate([rate_team_0, rate_team_1], rank)

    return new_rate_team_0, new_rate_team_1

def score(r):  # conservative for safety
    return r.mu - 3.0 * r.sigma

# Save videos, trajectories, and telemetry plots on each checkpoint
class SaveArtifactsOnCheckpoint(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        if algorithm.iteration % alg_config['checkpoint_freq'] == 0:
            trial_name = os.path.basename(algorithm._logdir)
            trial_dir = os.path.join(storage_path, RunName, trial_name)
            checkpoint_dir = os.path.join(trial_dir, f"checkpoint_{result['training_iteration']-1}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            env = algorithm.env_creator({'reward_version': 1})  # Must be num_env_runners = 1

            for i in range(8):  # Save 5 rollouts
                ExecuteEpisode(env, algorithm, checkpoint_dir, i)

            print(f"Finished Checkpoint at {checkpoint_dir}")

class SelfPlayRoundEvaluatorCheckpoint(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.Checkpoint_Window = Checkpoint_Window #{checkpoint_000004/team_0 : Rating(), ...}
        self.Current_Pair = Current_Pair # {team_0 : checkpoint_3999/team_1, team_1: checkpoint_3999/team_1} always num_teams entries
    
    def LoadCheckpoint(self, algorithm, checkpoint, id, pair_id, checkpoint_dir):
            print("\n++++++++++++++++++++++ Loading Checkpoint +++++++++++++++++++++\n")
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
            print(checkpoint_path)

            restored = algorithm.get_policy(id).from_checkpoint(checkpoint_path)[id]
            weights = restored.get_weights()
            algorithm.get_policy(id).set_weights(weights)
            print(f"Loaded policy {id}")

            self.Current_Pair[pair_id] = f"{checkpoint}/{id}"
            print(self.Checkpoint_Window)
            print(f"Current_Pairing: {self.Current_Pair}")
            print("\n++++++++++++++++++++++ Checkpoint Loaded +++++++++++++++++++++\n")
        
    def RandomOpponentDraw(self, temperature: float = 1.0, k: float = 3.0):
        keys = self.Checkpoint_Window.keys()
        mus = np.array([self.Checkpoint_Window[k].mu for k in keys], float)
        sig = np.array([self.Checkpoint_Window[k].sigma for k in keys], float)

        scores = mus - k * sig
        t = max(1e-9, float(temperature))
        z = (scores - scores.max()) / t
        np.clip(z, -700, 700, out=z)  # avoid overflow
        w = np.exp(z)
        probability_softmax = w / w.sum()

        keys = np.array(list(self.Checkpoint_Window.keys()))
        drawn_key = np.random.choice(keys, p=probability_softmax)

        if drawn_key!=Base_Checkpoint: 
            return drawn_key.split('/', 1)
        else:
            return drawn_key, 'team_0'

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        if algorithm.iteration % alg_config['checkpoint_freq'] == 0:
            trial_name = os.path.basename(algorithm._logdir)
            trial_dir = os.path.join(storage_path, RunName, trial_name)
            checkpoint_dir = os.path.join(trial_dir, f"Round_{algorithm.iteration // alg_config['checkpoint_freq']}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            env = algorithm.env_creator({'reward_version': 1})  # Must be num_env_runners = 1

            starting_agents_number = env_config['alive_agents_start']
            Current_Pair_rating = {}
            for key in self.Current_Pair.keys():
                Current_Pair_rating[key] = self.Checkpoint_Window[self.Current_Pair[key]]

            for i in range(15 * starting_agents_number):  # roughly computed necessary trials for TrueSkill updates 
                winning_team = ExecuteEpisode(env, algorithm, checkpoint_dir, i)
                print(f"result: {winning_team}")
                if winning_team == 'draw':
                    rank = [0, 0]
                    new_team_0, new_team_1 = TrueSkill(rank, Current_Pair_rating, starting_agents_number)
                    Current_Pair_rating['team_0'] = new_team_0
                    Current_Pair_rating['team_1'] = new_team_1
                else:
                    rank = [1, 1]
                    rank[int(winning_team[-1])] = 0
                    new_team_0, new_team_1 = TrueSkill(rank, Current_Pair_rating, starting_agents_number)
                    Current_Pair_rating['team_0'] = new_team_0
                    Current_Pair_rating['team_1'] = new_team_1
            
            #save higher score checkpoint, update score adversary checkpoint in window.
            if score(new_team_0) >= score(new_team_1):
                # update losing adversary score in window
                self.Checkpoint_Window[self.Current_Pair['team_1']] = new_team_1

                #load new adversary in team_1
                adversary, policy_id = self.RandomOpponentDraw()
                if adversary == Base_Checkpoint or adversary==Adversary_Base_Checkpoint:
                    self.LoadCheckpoint(algorithm, adversary, policy_id, 'team_1', os.path.join(storage_path, RunName))
                else: 
                    self.LoadCheckpoint(algorithm, adversary, policy_id, 'team_1', trial_dir)

                # save tune created checkpoint to window with its rating
                checkpoint_id = (algorithm.iteration // alg_config['checkpoint_freq'])-1
                checkpoint_name = f"checkpoint_{checkpoint_id:06d}/team_0"
                self.Checkpoint_Window[checkpoint_name] = new_team_0
            
            else:
                # update losing adversary score in window
                self.Checkpoint_Window[self.Current_Pair['team_0']] = new_team_0

                #load new adversary in team_0
                adversary, policy_id = self.RandomOpponentDraw()
                if adversary == Base_Checkpoint or adversary==Adversary_Base_Checkpoint:
                    self.LoadCheckpoint(algorithm, adversary, policy_id, 'team_0', os.path.join(storage_path, RunName))
                else: 
                    self.LoadCheckpoint(algorithm, adversary, policy_id, 'team_0', trial_dir)

                # save tune created checkpoint to window with its rating
                checkpoint_id = (algorithm.iteration // alg_config['checkpoint_freq'])-1
                checkpoint_name = f"checkpoint_{checkpoint_id:06d}/team_1"
                self.Checkpoint_Window[checkpoint_name] = new_team_1
        return self.Checkpoint_Window, self.Current_Pair
                    
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
        metrics["kills_mean"] = env_metrics["custom_metrics"].get("kills_mean", 0)
        metrics["attack_steps_max"] = env_metrics["custom_metrics"].get("attack_steps_max", 0)
        metrics["attack_steps_mean"] = env_metrics["custom_metrics"].get("attack_steps_mean", 0)

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

    def on_train_result_SelfPlay(self, *, algorithm, result, Checkpoint_Window, Current_Pair, **kwargs):
        trial_name = os.path.basename(algorithm._logdir)
        super().on_train_result(algorithm=algorithm, result=result, **kwargs)

        # Init W&B once
        if not self.initialized:
            wandb.init(
                project="aerial-battle",
                group=f"{RunName}",
                name=f'{RunName}/{trial_name}',
                config=algorithm.config,
                mode="online",
            )
            # Define the global step metric for the run
            wandb.define_metric("training_iteration")
            wandb.define_metric("*", step_metric="training_iteration")
            # Prepare the persistent pairings table
            self.pairing_Table = wandb.Table(columns=["round", "team_0", "team_1"])
            self.initialized = True

        # Use a single step for everything in this callback
        step = int(result.get("training_iteration", algorithm.iteration))
        round_no = step // alg_config["checkpoint_freq"]

        # -------- Collect scalar metrics --------
        metrics = {}
        env_metrics = result.get("env_runners", {})
        metrics["reward_mean"] = env_metrics.get("episode_reward_mean")
        metrics["reward_max"] = env_metrics.get("episode_reward_max")
        metrics["reward_min"] = env_metrics.get("episode_reward_min")
        metrics["episode_len_mean"] = env_metrics.get("episode_len_mean")
        metrics["kills_mean"] = env_metrics.get("custom_metrics", {}).get("kills_mean", 0)
        metrics["attack_steps_max"] = env_metrics.get("custom_metrics", {}).get("attack_steps_max", 0)
        metrics["attack_steps_mean"] = env_metrics.get("custom_metrics", {}).get("attack_steps_mean", 0)

        learner_stats = result.get("info", {}).get("learner", {}).get("team_0", {}).get("learner_stats", {})
        for key in ["alpha_value", "actor_loss", "critic_loss", "target_entropy"]:
            if key in learner_stats:
                metrics[key] = learner_stats[key]

        # Filter NaNs/None
        metrics = {k: float(v) for k, v in metrics.items()
                if v is not None and not (isinstance(v, float) and math.isnan(v))}

        # -------- Bar chart data (optional) --------
        plots = {}
        items = list(Checkpoint_Window.items())
        if items:
            keys = np.array([k for k, _ in items], dtype=object)
            mus  = np.array([r.mu for _, r in items], dtype=float)
            sigs = np.array([r.sigma for _, r in items], dtype=float)
            cons = mus - 3.0 * sigs

            order = np.argsort(cons)[::-1][:50]
            bar_table = wandb.Table(columns=["checkpoint", "conservative_score"])
            for idx in order.tolist():
                bar_table.add_data(keys[idx], float(cons[idx]))

            plots["Conservative Score per Checkpoint"] = wandb.plot.bar(
                bar_table, "checkpoint", "conservative_score",
                title="Conservative Score (μ - 3σ) by Checkpoint"
            )

        # -------- Single log with a unified step --------
        payload = {}
        payload.update(metrics)
        if plots:
            payload.update(plots)

        # -------- Pairings table: append once per round --------
        already_logged_rounds = set(row[0] for row in self.pairing_Table.data)
        if round_no not in already_logged_rounds:
            self.pairing_Table.add_data(
                round_no,
                Current_Pair.get("team_0"),
                Current_Pair.get("team_1"),
            )
            print(self.pairing_Table.data)
            payload["Pairings per Round"] = self.pairing_Table

        wandb.log(payload, step=step)

# Broker to combine multiple callbacks and restore from a base checkpoint
class CallbacksBroker(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        if Adversary_Base_Checkpoint:
            self.EvaluationCallback = SelfPlayRoundEvaluatorCheckpoint()
        else:
            self.EvaluationCallback = SaveArtifactsOnCheckpoint()
        self.WandbCallBack = CustomWandbCallback()
    
    def on_algorithm_init(self, *, algorithm, metrics_logger=None, **kwargs):
        if Base_Checkpoint and Adversary_Base_Checkpoint:
            print("\n++++++++++++++++++++++ Loading Checkpoint +++++++++++++++++++++\n")
            checkpoint_path = os.path.join(storage_path, RunName, Base_Checkpoint)
            adversary_checkpoint_path = os.path.join(storage_path, RunName, Adversary_Base_Checkpoint)
            for id in Base_Policy_restore:
                if id == 'team_0':
                    checkpoint = checkpoint_path
                else:
                    checkpoint = adversary_checkpoint_path

                restored = algorithm.get_policy(id).from_checkpoint(checkpoint)['team_0']

                weights = restored.get_weights()
                algorithm.get_policy(id).set_weights(weights)
                print(f"Loaded policy {id}")

                if id == 'team_0':
                    Checkpoint_Window[f"{Base_Checkpoint}/{'team_0'}"] = ts.Rating(mu=25.000, sigma=8.333)
                    Current_Pair[id] = f"{Base_Checkpoint}/{'team_0'}"
                else:
                    Checkpoint_Window[f"{Adversary_Base_Checkpoint}/{'team_0'}"] = ts.Rating(mu=25.000, sigma=8.333)
                    Current_Pair[id] = f"{Adversary_Base_Checkpoint}/{'team_0'}"

            print(f"Current_Pairing: {Current_Pair}")
            print("\n++++++++++++++++++++++ Checkpoint Loaded +++++++++++++++++++++\n")

        elif Base_Checkpoint:
            print("\n++++++++++++++++++++++ Loading Checkpoint +++++++++++++++++++++\n")
            checkpoint_path = os.path.join(storage_path, RunName, Base_Checkpoint)
            restored = algorithm.get_policy(id).from_checkpoint(checkpoint_path)['team_0']
            weights = restored.get_weights()
            algorithm.get_policy(id).set_weights(weights)
            print(f"Loaded policy {id}")
            print("\n++++++++++++++++++++++ Checkpoint Loaded +++++++++++++++++++++\n")

    def on_train_result(self, *, algorithm, result, **kwargs):
        if Adversary_Base_Checkpoint:
            Checkpoint_Window, Current_Pair = self.EvaluationCallback.on_train_result(algorithm=algorithm, result=result, **kwargs)
            self.WandbCallBack.on_train_result_SelfPlay(algorithm=algorithm, result=result, Checkpoint_Window=Checkpoint_Window,
                                            Current_Pair=Current_Pair, **kwargs)
        else:
            self.EvaluationCallback.on_train_result(algorithm=algorithm, result=result, **kwargs)
            self.WandbCallBack.on_train_result(algorithm=algorithm, result=result, **kwargs)

    
    def on_episode_step(self, *, episode, **kwargs):
        common_info = episode._last_infos.get("__common__", {})
        attack_metric = common_info.get("attack_steps", None)
        kill_metric = common_info.get("kills", None)
        if attack_metric is not None:
            episode.custom_metrics["attack_steps"] = (attack_metric)
        if kill_metric is not None:
            episode.custom_metrics["kills"] = (kill_metric)
    

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
        "model": {"fcnet_hiddens": [256, 256], "fcnet_activation": 'relu'},
    }),
    "team_1": (None, obs_space, act_space, {
        "model": {"fcnet_hiddens": [256, 256], "fcnet_activation": 'relu'},
    }),
}

# Policy assignment function
def policy_mapping_fn(agent_id, episode=0, **kwargs):
    if agent_id.startswith("agent_0"):
        return "team_0"
    if agent_id.startswith("agent_1"):
        return "team_1"

def name_creator(trial):
    name = f"Trial_{trial.config['gamma']}_{trial.config['train_batch_size']}_" \
           f"{trial.config['env_config']['reward_version']}_" \
           f"{trial.config['replay_buffer_config']['capacity']}"
    return name

# === Algorithm Configuration ===
algo_config = (
    SACConfig()
    .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    .environment(env="aerial_battle", env_config={'reward_version': tune.grid_search([1,2,3,4])})
    .training(
        train_batch_size=tune.grid_search(alg_config['batch_size_per_learner']),
        gamma=tune.grid_search(alg_config['gamma']),

        optimization_config = {
            'actor_learning_rate': 0.00003,
            'critic_learning_rate': 0.0003,
            'entropy_learning_rate': 0.0003
            },
        initial_alpha = 1,
        tau = 0.005,
        grad_clip=50,
        replay_buffer_config={
            'type': 'MultiAgentReplayBuffer',
            'capacity': tune.grid_search([500000]),
        }
    )
    .env_runners(
        num_env_runners=11,
        num_envs_per_env_runner=1,
        num_cpus_per_env_runner=1,
        num_gpus_per_env_runner=0,
        batch_mode="truncate_episodes",
        sample_timeout_s=120
    )
    .multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=policies_to_train,  # Only train team_0
    )
    .callbacks(CallbacksBroker)  # Combined callbacks
)

# === Ray Tune Tuner Configuration ===
tuner = tune.Tuner(
    trainable="SAC",
    param_space=algo_config,
    tune_config=TuneConfig(
        trial_name_creator=lambda trial: name_creator(trial),
        trial_dirname_creator=lambda trial: name_creator(trial)
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