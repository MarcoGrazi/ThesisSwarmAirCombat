import math
import os
from typing import Optional

import imageio
import wandb
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune import RunConfig, TuneConfig

# =========================
# Imports / Setup
# =========================
# This script:
# 1) loads experiment + environment config from YAML
# 2) registers a multi-agent RLlib environment
# 3) sets up SAC training with Tune grid-search
# 4) optionally runs self-play "league rounds" using TrueSkill for opponent selection
# 5) saves artifacts (videos, plots, telemetry) periodically

from EnvironmentClass import AerialBattle
import numpy as np
import yaml
import torch
import ray
from ray import tune
from ray.rllib.algorithms.sac import SACConfig
from ray.tune.registry import register_env
import trueskill as ts
import pickle
import json
import matplotlib.pyplot as plt
from matplotlib import cm

ConfigFile = 'Train_Run_config.yaml'

# =========================
# Load YAML experiment configuration
# =========================
# Expect YAML keys:
# - alg_config: training, logging, checkpointing, self-play parameters
# - env_config: environment settings (physics frequency, reward versions, etc.)
# - uav_config: aircraft models with aero + PID + limits + cones etc.
with open(ConfigFile) as f:
    yaml_config = yaml.load(f, Loader=yaml.FullLoader)

alg_config = yaml_config['alg_config']
env_config = yaml_config['env_config']
uav_config = yaml_config['uav_config']

# =========================
# External logging setup (W&B)
# =========================
# WANDB_API_KEY is read from config and stored in env var for wandb login.
# This enables logging metrics from callback hooks during training.
os.environ["WANDB_API_KEY"] = alg_config["WANDB_API_KEY"]
wandb.login(key=os.environ["WANDB_API_KEY"])  # once, at the top of your script


# =========================
# Run folder / artifact paths
# =========================
# This creates a run directory and stores a copy of the YAML config
# as "description.txt" for reproducibility.
Folder = alg_config['artifacts_folder']
RunName = alg_config['run_name']
RunDescription = (open(ConfigFile).read())

Checkpoints = alg_config['initial_checkpoints']  # initial seeds for self-play pool
Ratings = {}               # checkpoint_name -> TrueSkill rating
Current_Match = {}         # team_id -> checkpoint_name
Match_History = []         # list of Current_Match snapshots across rounds
policies_to_train = alg_config['policies_to_train']

storage_path = os.path.join(os.getcwd(), Folder)
run_path = os.path.join(Folder, RunName)
os.makedirs(run_path, exist_ok=True)

description_path = os.path.join(run_path, "description.txt")
with open(description_path, "w") as f:
    f.write(RunDescription)

print(f"Created run directory and saved description at: {description_path}")


# ============================================================
# Evaluation / Artifact helpers
# ============================================================

def ExecuteEpisode(env, algorithm, checkpoint_dir, i):
    """
    Runs a single evaluation rollout (deterministic policy actions) and saves:
    - episode video
    - trajectory visualization
    - telemetry CSV/plots
    - reward plots

    If self-play is active, returns the winning team ("team_0", "team_1", ...) or "draw".
    """
    checkpoint_dir_i = os.path.join(checkpoint_dir, f"{i}")
    os.makedirs(checkpoint_dir_i, exist_ok=True)

    # testing=True is used to make initialization more stable / repeatable
    obs, _ = env.reset(testing=True)
    terminated = {"__all__": False}
    frames = []

    print(f"Evaluation round {i}...")

    # Run until termination or a hard limit to cap rollout length
    while not terminated["__all__"] and len(frames) < alg_config['checkpoint_length']:
        actions = {}

        # Multi-agent: compute an action for each alive agent using its mapped policy
        for agent_id, agent_obs in obs.items():
            policy_id = policy_mapping_fn(agent_id)
            policy = algorithm.get_policy(policy_id)
            action, _, _ = policy.compute_single_action(agent_obs, explore=False)
            actions[agent_id] = action

        # Step the environment forward using the actions dict
        obs, _, terminated, truncated, _ = env.step(actions)

        # Treat truncation as termination for evaluation bookkeeping
        terminated["__all__"] = terminated["__all__"] or truncated["__all__"]

        # Capture rendering frames for video
        if hasattr(env, "render"):
            frames.append(env.render(mode="rgb_array"))

    # Save episode video for debugging / qualitative assessment
    video_path = os.path.join(checkpoint_dir_i, f"episode_video.mp4")
    imageio.mimsave(video_path, frames, fps=10)

    # Save additional visualizations if environment supports them
    if hasattr(env, "render_trajectory"):
        env.render_trajectory(checkpoint_dir_i)
    if hasattr(env, "plot_telemetry"):
        env.plot_telemetry(checkpoint_dir_i)
    if hasattr(env, "plot_rewards"):
        env.plot_rewards(checkpoint_dir_i)

    # In self-play mode, evaluation decides which team "wins" for rating update
    if len(Checkpoints) > 1:
        return env.get_winning_team()


def LoadCheckpoint(algorithm, checkpoint_name, team_id, policy_id):
    """
    Loads weights from a Ray checkpoint folder and applies them to a target policy (team_id).

    checkpoint_name: directory name under the run storage path
    policy_id: the policy key stored inside the checkpoint (e.g., 'team_0')
    team_id: the target policy in this current training run to receive those weights
    """
    checkpoint_path = os.path.join(storage_path, RunName, checkpoint_name)

    # from_checkpoint returns a dict of restored policies; we grab the requested one
    restored = algorithm.get_policy(policy_id).from_checkpoint(checkpoint_path)[policy_id]
    weights = restored.get_weights()

    # Apply weights into the active training policy
    algorithm.get_policy(team_id).set_weights(weights)
    print(f"Loaded policy {policy_id} for {team_id}")


def Export_Weights(algorithm, export_path, policy_id):
    """
    Exports a policy's weights as a standalone file (weights.pkl).
    This is used for self-play league management without relying on Ray checkpoint timing.
    """
    os.makedirs(export_path, exist_ok=True)

    policy = algorithm.get_policy(policy_id)
    weights = policy.get_weights()

    weights_file = os.path.join(export_path, f"weights.pkl")
    with open(weights_file, "wb") as f:
        pickle.dump(weights, f)

    print(f"Exported weights to {weights_file}")


def Import_Weights(algorithm, import_path, team_id):
    """
    Loads weights.pkl from disk and applies them to a target policy (team_id).
    Used to swap league opponents in/out between rounds.
    """
    weights_file = os.path.join(import_path, f"weights.pkl")
    with open(weights_file, "rb") as f:
        weights = pickle.load(f)

    algorithm.get_policy(team_id).set_weights(weights)
    print(f"Imported weights from {weights_file} to {team_id}")


# ============================================================
# TrueSkill utilities
# ============================================================

def TrueSkill(rank, Current_Match_rating, Starting_Agents_Number):
    """
    Applies a TrueSkill update to the teams in the current match.

    rank: array-like ranking (0 = winner, higher = worse). Equal ranks = draw.
    Current_Match_rating: dict {team_key: ts.Rating}
    Starting_Agents_Number: number of planes per team that are alive at match start

    Note: We represent each team as a group of identical players (same rating repeated),
    then collapse back to a single rating after update.
    """
    rate_teams = []
    for key in Current_Match_rating.keys():
        rate_teams.append([Current_Match_rating[key]] * Starting_Agents_Number)

    new_rates = ts.rate(rate_teams, rank)

    # Collapse each team list back into one representative rating
    new_rate_teams = []
    for team_rates in new_rates:
        new_rate_teams.append(team_rates[0])

    return new_rate_teams


def score(r):
    """
    Conservative TrueSkill score (lower confidence bound).
    Useful for selecting "best" policy with uncertainty penalty.
    """
    return r.mu - 3.0 * r.sigma


def plot_and_save_matching_history(Matching_History, Ratings, save_dir="plots"):
    """
    Stores:
    - matching_history.json: round-by-round matchups
    - ratings_gaussians.png: Gaussian curves per checkpoint rating (mu/sigma)

    Matching_History: list of dicts mapping team -> checkpoint_name
    Ratings: dict mapping checkpoint_name -> ts.Rating
    """
    os.makedirs(save_dir, exist_ok=True)

    json_path = os.path.join(save_dir, "matching_history.json")
    with open(json_path, "w") as f:
        json.dump(Matching_History, f, indent=2)
    print(f"Saved Matching_History to {json_path}")

    if not Ratings:
        print("Ratings dictionary is empty — skipping Gaussian plot.")
        return

    plt.figure(figsize=(12, 6))

    # Build an x-range that covers all checkpoint distributions
    all_mus = np.array([r.mu for r in Ratings.values()])
    all_sigmas = np.array([r.sigma for r in Ratings.values()])
    xmin = (all_mus - 4 * all_sigmas).min()
    xmax = (all_mus + 4 * all_sigmas).max()
    x = np.linspace(xmin, xmax, 400)

    colors = cm.get_cmap('tab20', len(Ratings))

    for idx, (checkpoint, rating) in enumerate(Ratings.items()):
        mu = rating.mu
        sigma = rating.sigma
        y = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        plt.plot(x, y, label=f"{checkpoint} (μ={mu:.1f}, σ={sigma:.1f})", color=colors(idx))

    plt.title("TrueSkill Gaussian Distributions for Each Checkpoint")
    plt.xlabel("Skill Estimate")
    plt.ylabel("Probability Density")
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    ratings_path = os.path.join(save_dir, "ratings_gaussians.png")
    plt.savefig(ratings_path)
    plt.close()
    print(f"Saved Gaussian Ratings plot to {ratings_path}")


# ============================================================
# Callbacks
# ============================================================
# Two evaluation modes:
# 1) Standard training: save artifacts periodically
# 2) Self-play training: evaluate, update ratings, rotate opponents, export/import weights

class SaveArtifactsOnCheckpoint(DefaultCallbacks):
    """
    Periodically runs a few deterministic evaluation episodes and saves artifacts.
    Useful for non-self-play runs.
    """
    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        if algorithm.iteration % alg_config['checkpoint_freq'] == 0:
            trial_name = os.path.basename(algorithm._logdir)
            trial_dir = os.path.join(storage_path, RunName, trial_name)

            # Create a checkpoint directory for this training iteration
            checkpoint_dir = os.path.join(trial_dir, f"checkpoint_{result['training_iteration']-1}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Run a separate evaluation env instance to generate videos/plots
            env = algorithm.env_creator({'reward_version': 1})
            num_evaluation_episodes = alg_config['num_evaluation_episodes']

            for i in range(num_evaluation_episodes):
                ExecuteEpisode(env, algorithm, checkpoint_dir, i)

            print(f"Finished Checkpoint at {checkpoint_dir}")


class SelfPlayRoundEvaluatorCheckpoint(DefaultCallbacks):
    """
    Self-play league manager:
    - evaluates current match for N episodes
    - updates TrueSkill ratings
    - exports new weights for teams
    - reassigns opponents for next round
    """
    def __init__(self):
        super().__init__()
        self.Checkpoints = Checkpoints
        self.Current_Match = Current_Match
        self.Ratings = Ratings

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        if algorithm.iteration % alg_config['checkpoint_freq'] == 0:
            trial_name = os.path.basename(algorithm._logdir)
            trial_dir = os.path.join(storage_path, RunName, trial_name)

            # Round directory groups evaluation artifacts per self-play generation
            checkpoint_dir = os.path.join(
                trial_dir,
                f"Round_{algorithm.iteration // alg_config['checkpoint_freq']}"
            )
            os.makedirs(checkpoint_dir, exist_ok=True)

            env = algorithm.env_creator({'reward_version': 1})

            num_evaluation_episodes = alg_config['num_evaluation_episodes']
            starting_agents_number = env_config['alive_agents_start']

            # Collect current match ratings for TrueSkill update loop
            Current_Match_rating = {}
            for key in self.Current_Match.keys():
                Current_Match_rating[key] = self.Ratings[self.Current_Match[key]]

            # Evaluate multiple episodes to stabilize TrueSkill updates
            for i in range(num_evaluation_episodes):
                winning_team = ExecuteEpisode(env, algorithm, checkpoint_dir, i)
                print(f"result: {winning_team}")

                if winning_team == 'draw':
                    # Equal rank for draw: all teams get same placement
                    rank = np.zeros(len(self.Current_Match))
                else:
                    # Winner rank 0, others rank 1
                    rank = np.ones(len(self.Current_Match))
                    rank[int(winning_team[-1])] = 0

                # Apply TrueSkill update
                new_rate_teams = TrueSkill(rank, Current_Match_rating, starting_agents_number)
                for key in Current_Match_rating.keys():
                    Current_Match_rating[key] = new_rate_teams[int(key[-1])]

                print(f"Current Match Ratings: {Current_Match_rating}")

            # Persist updated ratings back into global pool
            for key in self.Current_Match.keys():
                self.Ratings[self.Current_Match[key]] = Current_Match_rating[key]

            # Select winner using conservative score
            scores = [score(r) for r in Current_Match_rating.values()]
            maxkey = f"team_{np.argmax(scores, axis=0)}"
            round_id = (algorithm.iteration // alg_config["checkpoint_freq"])

            # Export winner weights under a round-tagged name
            winner_old_check = self.Current_Match[maxkey]
            winner_old_name = winner_old_check.split('P')[0][:-2]
            plane_model = winner_old_check.split('P')[1]
            Rating = Current_Match_rating[maxkey]

            winner_new_check = f"{winner_old_name}_{round_id}P{plane_model}"
            export_path = os.path.join(storage_path, RunName, trial_name, winner_new_check)
            Export_Weights(algorithm, export_path, maxkey)

            # Replace old checkpoint entry with new one (keeps pool "current")
            self.Checkpoints[self.Checkpoints.index(winner_old_check)] = winner_new_check
            self.Ratings[winner_new_check] = Rating
            self.Ratings.pop(winner_old_check)

            # Move winner into fixed "champion slot" (team_0) for next round
            Import_Weights(algorithm, export_path, 'team_0')

            # Export losers too (so pool remains consistent across rounds)
            for loser in self.Current_Match.keys():
                if loser != maxkey:
                    loser_old_check = self.Current_Match[loser]
                    loser_old_name = loser_old_check.split('P')[0][:-2]
                    plane_model = loser_old_check.split('P')[1]
                    Rating = Current_Match_rating[loser]

                    loser_new_check = f"{loser_old_name}_{round_id}P{plane_model}"
                    export_path = os.path.join(storage_path, RunName, trial_name, loser_new_check)

                    # Export the current weights for that loser team into its new file
                    Export_Weights(algorithm, export_path, loser)

                    self.Checkpoints[self.Checkpoints.index(loser_old_check)] = loser_new_check
                    self.Ratings[loser_new_check] = Rating
                    self.Ratings.pop(loser_old_check)

            # Build next round pairings: champion is team_0, contenders sampled randomly
            self.Current_Match = {}
            self.Current_Match['team_0'] = winner_new_check

            remaining_checkpoints = [ck for ck in self.Checkpoints if ck != winner_new_check]
            print(f"Remaining Checkpoints for Contenders: {remaining_checkpoints}")

            for i in range(1, env_config['team_number']):
                random_index = np.random.randint(len(remaining_checkpoints))
                contender_check = remaining_checkpoints[random_index]

                self.Current_Match[f'team_{i}'] = contender_check
                import_path = os.path.join(storage_path, RunName, trial_name, contender_check)
                Import_Weights(algorithm, import_path, f'team_{i}')

                remaining_checkpoints.pop(random_index)

            print(f"Next Round Pairings: {self.Current_Match}")

        return self.Checkpoints, self.Current_Match, self.Ratings


class CustomWandbCallback(DefaultCallbacks):
    """
    Logs key training metrics into W&B.
    Has a standard mode and a self-play mode (so you can also log ratings later if desired).
    """
    def __init__(self):
        super().__init__()
        self.initialized = False

    def on_train_result(self, *, algorithm, result, **kwargs):
        trial_name = os.path.basename(algorithm._logdir)
        super().on_train_result(algorithm=algorithm, result=result, **kwargs)

        if not self.initialized:
            wandb.init(
                project="aerial-battle",
                group=f"{RunName}",
                name=f'{RunName}/{trial_name}',
                config=algorithm.config,
                mode="online"
            )
            self.initialized = True

        env_metrics = result.get("env_runners", {})
        metrics = {
            "reward_mean": env_metrics.get("episode_reward_mean"),
            "reward_max": env_metrics.get("episode_reward_max"),
            "reward_min": env_metrics.get("episode_reward_min"),
            "episode_len_mean": env_metrics.get("episode_len_mean"),
            "kills_mean": env_metrics.get("custom_metrics", {}).get("kills_mean", 0),
            "attack_steps_max": env_metrics.get("custom_metrics", {}).get("attack_steps_max", 0),
            "attack_steps_mean": env_metrics.get("custom_metrics", {}).get("attack_steps_mean", 0),
        }

        learner_stats = (
            result.get("info", {}).get("learner", {}).get("team_0", {}).get("learner_stats", {})
        )
        for key in ["alpha_value", "actor_loss", "critic_loss", "target_entropy"]:
            if key in learner_stats:
                metrics[key] = learner_stats[key]

        # Keep only valid scalars
        metrics = {
            k: float(v) for k, v in metrics.items()
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        }

        wandb.log(metrics, step=result['training_iteration'])

    def on_train_result_SelfPlay(self, *, algorithm, result, Ratings, Current_Match, **kwargs):
        """
        Same logging but can be extended to also log:
        - round number
        - current match pairing
        - TrueSkill mu/sigma per active team
        """
        trial_name = os.path.basename(algorithm._logdir)
        super().on_train_result(algorithm=algorithm, result=result, **kwargs)

        if not self.initialized:
            wandb.init(
                project="aerial-battle",
                group=f"{RunName}",
                name=f"{RunName}/{trial_name}",
                config=algorithm.config,
                mode="online",
            )
            wandb.define_metric("training_iteration")
            wandb.define_metric("*", step_metric="training_iteration")
            self.initialized = True

        step = result.get("training_iteration")
        env_metrics = result.get("env_runners", {})

        metrics = {
            "reward_mean": env_metrics.get("episode_reward_mean"),
            "reward_max": env_metrics.get("episode_reward_max"),
            "reward_min": env_metrics.get("episode_reward_min"),
            "episode_len_mean": env_metrics.get("episode_len_mean"),
            "kills_mean": env_metrics.get("custom_metrics", {}).get("kills_mean", 0),
            "attack_steps_max": env_metrics.get("custom_metrics", {}).get("attack_steps_max", 0),
            "attack_steps_mean": env_metrics.get("custom_metrics", {}).get("attack_steps_mean", 0),
        }

        learner_stats = (
            result.get("info", {})
                .get("learner", {})
                .get("team_0", {})
                .get("learner_stats", {})
        )

        for key in ["alpha_value", "actor_loss", "critic_loss", "target_entropy"]:
            if key in learner_stats:
                metrics[key] = learner_stats[key]

        metrics = {
            k: float(v)
            for k, v in metrics.items()
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        }

        wandb.log(metrics, step=step)


class CallbacksBroker(DefaultCallbacks):
    """
    Single callback class that:
    - initializes self-play checkpoints or a single starting checkpoint
    - runs evaluation logic (self-play or artifact saving)
    - logs to W&B
    - pushes env custom metrics into RLlib episode metrics
    """
    def __init__(self):
        super().__init__()
        if len(Checkpoints) > 1:
            self.EvaluationCallback = SelfPlayRoundEvaluatorCheckpoint()
        else:
            self.EvaluationCallback = SaveArtifactsOnCheckpoint()
        self.WandbCallBack = CustomWandbCallback()

    def on_algorithm_init(self, *, algorithm, metrics_logger=None, **kwargs):
        """
        Initialization phase:
        - If self-play: load each seed checkpoint, export weights into league-format files, init ratings
        - If single checkpoint: load it into team_0 (resume or fine-tune)
        """
        if len(Checkpoints) > 1:
            trial_name = os.path.basename(algorithm._logdir)

            print("\n++++++++++++++++++++++ Loading Checkpoints for SelfPlay +++++++++++++++++++++\n")
            for t in range(len(Checkpoints)):
                checkpoint = Checkpoints[t]

                print(f"\n++++++++++++++++++++++ Initializing Checkpoint: {checkpoint}+++++++++++++++++++++")
                id = checkpoint.split('/')[1]
                checkpoint_name = checkpoint.split('/')[0]
                plane_model = checkpoint.split('/')[2]

                # Load into a staging policy slot (team_0), then export into league store
                LoadCheckpoint(algorithm, checkpoint_name, 'team_0', id)

                new_init_check = f"{checkpoint_name}_{0}P{plane_model}"
                export_path = os.path.join(storage_path, RunName, trial_name, new_init_check)
                Export_Weights(algorithm, export_path, 'team_0')

                Ratings[new_init_check] = ts.Rating()
                Checkpoints[Checkpoints.index(checkpoint)] = new_init_check

            print("\n++++++++++++++++++++++ All Checkpoints Initialized +++++++++++++++++++++\n")

            # Load initial match: first N checkpoints fill team slots
            for t in range(env_config['team_number']):
                print(f"\n++++++++++++++++++++++ Loading Checkpoint: {Checkpoints[t]}+++++++++++++++++++++")
                Current_Match[f'team_{t}'] = Checkpoints[t]
                import_path = os.path.join(storage_path, RunName, trial_name, Checkpoints[t])
                Import_Weights(algorithm, import_path, f'team_{t}')

            print("\n++++++++++++++++++++++ Checkpoints Loaded +++++++++++++++++++++\n")

            Match_History.append(Current_Match.copy())
            print(f"Initial Self-Play Pairings: {Current_Match}")

        elif len(Checkpoints) == 1:
            checkpoint = Checkpoints[0]
            print(f"\n++++++++++++++++++++++ Loading Checkpoint: {checkpoint}+++++++++++++++++++++\n")
            id = checkpoint.split('/')[1]
            checkpoint_name = checkpoint.split('/')[0]
            LoadCheckpoint(algorithm, checkpoint_name, 'team_0', id)
            print("\n++++++++++++++++++++++ Checkpoint Loaded +++++++++++++++++++++\n")

    def on_episode_created(self, *, episode, base_env, **kwargs):
        """
        Before each episode, update the plane model selection based on the current match pairing.
        This enables league opponents trained on different aircraft models to be evaluated correctly.
        """
        env = base_env.get_sub_environments()[0]
        if len(Checkpoints) > 1:
            for t in Current_Match.keys():
                plane_model = int(Current_Match[t].split('P')[1])
                env.set_plane_model(t, plane_model)

    def on_train_result(self, *, algorithm, result, **kwargs):
        """
        Main training hook:
        - run evaluation / self-play rotation every checkpoint_freq iterations
        - log metrics
        - store match history for final plots
        """
        global Checkpoints, Current_Match, Ratings, Match_History

        if len(Checkpoints) > 1:
            Checkpoints, Current_Match, Ratings = self.EvaluationCallback.on_train_result(
                algorithm=algorithm, result=result, **kwargs
            )

            self.WandbCallBack.on_train_result_SelfPlay(
                algorithm=algorithm, result=result, Ratings=Ratings, Current_Match=Current_Match, **kwargs
            )

            if algorithm.iteration % alg_config['checkpoint_freq'] == 0:
                Match_History.append(Current_Match.copy())

            if algorithm.iteration == alg_config['train_iterations']:
                plot_and_save_matching_history(
                    Match_History, Ratings,
                    save_dir=os.path.join(storage_path, RunName, "PLOTS")
                )
        else:
            self.EvaluationCallback.on_train_result(algorithm=algorithm, result=result, **kwargs)
            self.WandbCallBack.on_train_result(algorithm=algorithm, result=result, **kwargs)

    def on_episode_step(self, *, episode, **kwargs):
        """
        Pull environment-level custom metrics from info["__common__"]
        and report them as episode custom metrics in RLlib.
        """
        common_info = episode._last_infos.get("__common__", {})
        attack_metric = common_info.get("attack_steps", None)
        kill_metric = common_info.get("kills", None)

        if attack_metric is not None:
            episode.custom_metrics["attack_steps"] = attack_metric
        if kill_metric is not None:
            episode.custom_metrics["kills"] = kill_metric


# ============================================================
# Environment registration
# ============================================================

def env_creator(cfg):
    """
    RLlib env factory.
    cfg contains the env_config passed from .environment(env_config=...).
    """
    return AerialBattle(env_config, uav_config, cfg['reward_version'], discretize=True)

register_env("aerial_battle", env_creator)


# ============================================================
# Extract observation/action spaces (for policy specs)
# ============================================================
dummy_env = AerialBattle(env_config=env_config, UAV_config=uav_config)
obs_space = dummy_env.get_observation_space('agent_0_0')
act_space = dummy_env.get_action_space("agent_1_0")
dummy_env.close()


# ============================================================
# Multi-agent policy definitions
# ============================================================
# One policy per team. RLlib maps agents to policies using policy_mapping_fn.
# You can define more policies than teams (e.g., for future expansion / multi-team training).
policies = {
    "team_0": (None, obs_space, act_space, {
        "model": {
            "fcnet_hiddens": tune.grid_search(alg_config['fcnet_hiddens']),
            "fcnet_activation": tune.grid_search(alg_config['fcnet_activation']),
        },
    }),
    "team_1": (None, obs_space, act_space, {
        "model": {
            "fcnet_hiddens": tune.grid_search(alg_config['fcnet_hiddens']),
            "fcnet_activation": tune.grid_search(alg_config['fcnet_activation']),
        },
    }),
    "team_2": (None, obs_space, act_space, {
        "model": {
            "fcnet_hiddens": tune.grid_search(alg_config['fcnet_hiddens']),
            "fcnet_activation": tune.grid_search(alg_config['fcnet_activation']),
        },
    }),
    "team_3": (None, obs_space, act_space, {
        "model": {
            "fcnet_hiddens": tune.grid_search(alg_config['fcnet_hiddens']),
            "fcnet_activation": tune.grid_search(alg_config['fcnet_activation']),
        },
    }),
}

def policy_mapping_fn(agent_id, episode=0, **kwargs):
    """
    Maps environment agent IDs to team policies.
    agent_0_* -> team_0
    agent_1_* -> team_1
    """
    if agent_id.startswith("agent_0"):
        return "team_0"
    if agent_id.startswith("agent_1"):
        return "team_1"


def name_creator(trial):
    """
    Creates a readable trial folder name based on key hyperparameters.
    Useful for scanning Tune output directories.
    """
    name = (
        f"Trial_{trial.config['gamma']}_{trial.config['train_batch_size']}_"
        f"{trial.config['env_config']['reward_version']}_"
        f"{trial.config['replay_buffer_config']['capacity']}"
    )
    return name


# ============================================================
# SAC algorithm config
# ============================================================
algo_config = (
    SACConfig()
    .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    .environment(
        env="aerial_battle",
        env_config={'reward_version': tune.grid_search(list(env_config['reward_versions'].keys()))}
    )
    .training(
        train_batch_size=tune.grid_search(alg_config['batch_size_per_learner']),
        gamma=tune.grid_search(alg_config['gamma']),
        optimization_config={
            'actor_learning_rate': tune.grid_search(alg_config['actor_learning_rate']),
            'critic_learning_rate': tune.grid_search(alg_config['critic_learning_rate']),
            'entropy_learning_rate': tune.grid_search(alg_config['entropy_learning_rate'])
        },
        initial_alpha=tune.grid_search(alg_config['initial_alpha']),
        tau=tune.grid_search(alg_config['tau']),
        grad_clip=50,
        replay_buffer_config={
            'type': 'MultiAgentReplayBuffer',
            'capacity': tune.grid_search(alg_config['replay_buffer_capacity']),
        }
    )
    .env_runners(
        num_env_runners=alg_config['num_env_runners'],
        num_envs_per_env_runner=alg_config['num_envs_per_env_runner'],
        num_cpus_per_env_runner=alg_config['num_cpus_per_env_runner'],
        num_gpus_per_env_runner=alg_config['num_gpus_per_env_runner'],
        batch_mode=alg_config['batch_mode'],
        sample_timeout_s=alg_config['sample_timeout_s']
    )
    .multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=policies_to_train,
    )
    .callbacks(CallbacksBroker)
)


# ============================================================
# Tune Tuner config + run
# ============================================================
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

# Start the experiment
tuner.fit()
