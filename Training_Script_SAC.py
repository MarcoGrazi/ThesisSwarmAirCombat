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
import pickle
import json
import matplotlib.pyplot as plt
from matplotlib import cm


ConfigFile = 'Train_Run_config.yaml'

# === Load YAML experiment configuration ===
with open(ConfigFile) as f:
    yaml_config = yaml.load(f, Loader=yaml.FullLoader)

alg_config = yaml_config['alg_config']
env_config = yaml_config['env_config']
uav_config = yaml_config['uav_config']

# Set your WandB API key for logging
os.environ["WANDB_API_KEY"] = alg_config["WANDB_API_KEY"]
wandb.login(key=os.environ["WANDB_API_KEY"])  # once, at the top of your script


#Create RunName directory inside Folder, with RunDescription inside it
# === Create Directory and Save Description ===
Folder = alg_config['artifacts_folder']
RunName = alg_config['run_name']
RunDescription = (open(ConfigFile).read())

Checkpoints = alg_config['initial_checkpoints']  # List of checkpoints to initialize the TrueSkill window
Ratings = {}
Current_Match = {}
Match_History = []
policies_to_train = alg_config['policies_to_train']

storage_path = os.path.join(os.getcwd(), Folder)
run_path = os.path.join(Folder, RunName)
os.makedirs(run_path, exist_ok=True)

description_path = os.path.join(run_path, "description.txt")
with open(description_path, "w") as f:
    f.write(RunDescription)

print(f"Created run directory and saved description at: {description_path}")

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

    if len(Checkpoints)>1:
        return env.get_winning_team()

def LoadCheckpoint(algorithm, checkpoint_name, team_id, policy_id):
    checkpoint_path = os.path.join(storage_path, RunName, checkpoint_name)
    restored = algorithm.get_policy(policy_id).from_checkpoint(checkpoint_path)[policy_id]
    weights = restored.get_weights()
    algorithm.get_policy(team_id).set_weights(weights)
    print(f"Loaded policy {policy_id} for {team_id}")

def Export_Weights(algorithm, export_path, policy_id):
    os.makedirs(export_path, exist_ok=True)

    policy = algorithm.get_policy(policy_id)
    weights = policy.get_weights()

    weights_file = os.path.join(export_path, f"weights.pkl")

    with open(weights_file, "wb") as f:
        pickle.dump(weights, f)

    print(f"Exported weights to {weights_file}")

def Import_Weights(algorithm, import_path, team_id):
    weights_file = os.path.join(import_path, f"weights.pkl")

    with open(weights_file, "rb") as f:
        weights = pickle.load(f)

    algorithm.get_policy(team_id).set_weights(weights)

    print(f"Imported weights from {weights_file} to {team_id}")

def TrueSkill(rank, Current_Match_rating, Starting_Agents_Number):
    rate_teams = []
    for key in Current_Match_rating.keys():
        rate_teams.append([Current_Match_rating[key]] * Starting_Agents_Number)

    new_rates = ts.rate(rate_teams, rank)
    
    new_rate_teams = []
    for team_rates in new_rates:
        new_rate_teams.append(team_rates[0])

    return new_rate_teams

def score(r):  # conservative for safety
    return r.mu - 3.0 * r.sigma

def plot_and_save_matching_history(Matching_History, Ratings, save_dir="plots"):
    """
    Matching_History: list of dicts (one dict per round)
    Ratings: dict of {checkpoint_name: TrueSkillRating(mu, sigma)}
    """

    # 1. Ensure output directory exists
    os.makedirs(save_dir, exist_ok=True)

    # 2. Save Matching_History as JSON
    json_path = os.path.join(save_dir, "matching_history.json")
    with open(json_path, "w") as f:
        json.dump(Matching_History, f, indent=2)
    print(f"Saved Matching_History to {json_path}")

    # -------------------------
    # 3. Plot Gaussian curves for Ratings
    # -------------------------
    if not Ratings:
        print("Ratings dictionary is empty — skipping Gaussian plot.")
        return

    plt.figure(figsize=(12, 6))

    # Collect all mus to set a reasonable x-range
    all_mus = np.array([r.mu for r in Ratings.values()])
    all_sigmas = np.array([r.sigma for r in Ratings.values()])

    # Create x-axis range covering all distributions
    xmin = (all_mus - 4 * all_sigmas).min()
    xmax = (all_mus + 4 * all_sigmas).max()
    x = np.linspace(xmin, xmax, 400)

    # Colormap for distinct colors
    colors = cm.get_cmap('tab20', len(Ratings))

    for idx, (checkpoint, rating) in enumerate(Ratings.items()):
        mu = rating.mu
        sigma = rating.sigma

        # Gaussian formula
        y = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        plt.plot(x, y, label=f"{checkpoint} (μ={mu:.1f}, σ={sigma:.1f})",
                 color=colors(idx))

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

# Save videos, trajectories, and telemetry plots on each checkpoint
class SaveArtifactsOnCheckpoint(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        if algorithm.iteration % alg_config['checkpoint_freq'] == 0:
            trial_name = os.path.basename(algorithm._logdir)
            trial_dir = os.path.join(storage_path, RunName, trial_name)
            checkpoint_dir = os.path.join(trial_dir, f"checkpoint_{result['training_iteration']-1}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            env = algorithm.env_creator({'reward_version': 1})  # Must be num_env_runners = 1
            num_evaluation_episodes = alg_config['num_evaluation_episodes']

            for i in range(num_evaluation_episodes):  # Save 5 rollouts
                ExecuteEpisode(env, algorithm, checkpoint_dir, i)

            print(f"Finished Checkpoint at {checkpoint_dir}")

class SelfPlayRoundEvaluatorCheckpoint(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.Checkpoints = Checkpoints 
        self.Current_Match = Current_Match 
        self.Ratings = Ratings
        
    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        if algorithm.iteration % alg_config['checkpoint_freq'] == 0:
            trial_name = os.path.basename(algorithm._logdir)
            trial_dir = os.path.join(storage_path, RunName, trial_name)
            checkpoint_dir = os.path.join(trial_dir, f"Round_{algorithm.iteration // alg_config['checkpoint_freq']}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            env = algorithm.env_creator({'reward_version': 1})  # Must be num_env_runners = 1

            num_evaluation_episodes = alg_config['num_evaluation_episodes']
            starting_agents_number = env_config['alive_agents_start']
            Current_Match_rating = {}
            for key in self.Current_Match.keys():
                Current_Match_rating[key] = self.Ratings[self.Current_Match[key]]

            for i in range(num_evaluation_episodes):  # roughly computed necessary trials for TrueSkill updates 
                winning_team = ExecuteEpisode(env, algorithm, checkpoint_dir, i)
                print(f"result: {winning_team}")
                if winning_team == 'draw':
                    rank = np.zeros(len(self.Current_Match))
                    new_rate_teams = TrueSkill(rank, Current_Match_rating, starting_agents_number)
                    for key in Current_Match_rating.keys():
                        Current_Match_rating[key] = new_rate_teams[int(key[-1])]
                else:
                    rank = np.ones(len(self.Current_Match))
                    rank[int(winning_team[-1])] = 0
                    new_rate_teams = TrueSkill(rank, Current_Match_rating, starting_agents_number)
                    for key in Current_Match_rating.keys():
                        Current_Match_rating[key] = new_rate_teams[int(key[-1])]
                print(f"Current Match Ratings: {Current_Match_rating}")

            # Update Ratings
            for key in self.Current_Match.keys():
                self.Ratings[self.Current_Match[key]] = Current_Match_rating[key]
            
            # Prepare next round pairings based on updated ratings:
            # ray will automatically create a checkpoint
            scores = [score(r) for r in Current_Match_rating.values()]
            maxkey = f"team_{np.argmax(scores, axis=0)}"
            round_id = (algorithm.iteration // alg_config["checkpoint_freq"])
            
            # with ray 2.48.0, I had problems saving a checkpoint with .save_to_Checkpoint, while the ray.tune worked fine.
            # this has another problem which is that tune creates checkpoints after the on_train_result callback, so the created 
            # checkpoint has not been created yet here. 
            # in the end I decided to use a pkl file to save the weights instead of ray checkpoint system. The winner and loser(s) are saved
            # and then the first gets moved to the non-training position, team_0. the next match contenders are selected randomly 
            # from the remaining checkpoints. 

            # Update Checkpoints dictionary with winner's new checkpoint
            winner_old_check = self.Current_Match[maxkey]
            winner_old_name = winner_old_check.split('P')[0][:-2]
            plane_model = winner_old_check.split('P')[1]  # assuming format 'checkpoint_name/plane_model'
            Rating = Current_Match_rating[maxkey]

            winner_new_check = f"{winner_old_name}_{round_id}P{plane_model}"
            export_path = os.path.join(storage_path, RunName, trial_name, winner_new_check)
            Export_Weights(algorithm, export_path, maxkey)

            self.Checkpoints[self.Checkpoints.index(winner_old_check)] = winner_new_check
            self.Ratings[winner_new_check] = Rating
            self.Ratings.pop(winner_old_check)

            Import_Weights(algorithm, export_path, 'team_0')

            # Update Checkpoints dictionary with new contender's checkpoint
            for loser in self.Current_Match.keys():
                if loser != maxkey:
                    loser_old_check = self.Current_Match[loser]
                    loser_old_name = loser_old_check.split('P')[0][:-2]
                    plane_model = loser_old_check.split('P')[1]  # assuming format 'checkpoint_name/plane_model'
                    Rating = Current_Match_rating[loser]

                    loser_new_check = f"{loser_old_name}_{round_id}P{plane_model}"
                    export_path = os.path.join(storage_path, RunName, trial_name, loser_new_check)
                    Export_Weights(algorithm, export_path, maxkey)

                    self.Checkpoints[self.Checkpoints.index(loser_old_check)] = loser_new_check
                    self.Ratings[loser_new_check] = Rating
                    self.Ratings.pop(loser_old_check)
            

            self.Current_Match = {}
            self.Current_Match['team_0'] = winner_new_check

            # Select a new contender randomly from remaining checkpoints    
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

    def on_train_result_SelfPlay(self, *, algorithm, result, Ratings, Current_Match, **kwargs):
        trial_name = os.path.basename(algorithm._logdir)
        super().on_train_result(algorithm=algorithm, result=result, **kwargs)

        # Initialize W&B only once
        if not self.initialized:
            wandb.init(
                project="aerial-battle",
                group=f"{RunName}",
                name=f"{RunName}/{trial_name}",
                config=algorithm.config,
                mode="online",
            )

            # Define step metric
            wandb.define_metric("training_iteration")
            wandb.define_metric("*", step_metric="training_iteration")

            self.initialized = True

        # Use RLlib's iteration as global step
        step = result.get("training_iteration")
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

        learner_stats = (
            result.get("info", {})
                .get("learner", {})
                .get("team_0", {})
                .get("learner_stats", {})
        )

        for key in ["alpha_value", "actor_loss", "critic_loss", "target_entropy"]:
            if key in learner_stats:
                metrics[key] = learner_stats[key]

        # Clean NaNs and None
        metrics = {
            k: float(v)
            for k, v in metrics.items()
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        }

        # -------- Log scalar metrics -------- 
        wandb.log(metrics, step=step)

# Broker to combine multiple callbacks and restore from a base checkpoint
class CallbacksBroker(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        if len(Checkpoints)>1:
            self.EvaluationCallback = SelfPlayRoundEvaluatorCheckpoint()
        else:
            self.EvaluationCallback = SaveArtifactsOnCheckpoint()
        self.WandbCallBack = CustomWandbCallback()
    
    def on_algorithm_init(self, *, algorithm, metrics_logger=None, **kwargs):
        if len(Checkpoints)>1:
            trial_name = os.path.basename(algorithm._logdir)

            print("\n++++++++++++++++++++++ Loading Checkpoints for SelfPlay +++++++++++++++++++++\n")
            for t in range(env_config['team_number']):
                checkpoint = Checkpoints[t]

                print(f"\n++++++++++++++++++++++ Loading Checkpoint: {checkpoint}+++++++++++++++++++++\n")
                id = checkpoint.split('/')[1]
                checkpoint_name = checkpoint.split('/')[0]
                plane_model = checkpoint.split('/')[2]  # assuming format 'checkpoint_name/team_x/plane_model'
                LoadCheckpoint(algorithm, checkpoint_name, f'team_{t}', id)
                print("\n++++++++++++++++++++++ Checkpoint Loaded +++++++++++++++++++++\n")
                
                new_init_check = f"{checkpoint_name}_{0}P{plane_model}"
                export_path = os.path.join(storage_path, RunName, trial_name, new_init_check)
                Export_Weights(algorithm, export_path, f'team_{t}')
                Ratings[new_init_check] = ts.Rating()
                Current_Match[f'team_{t}'] = new_init_check
                Checkpoints[Checkpoints.index(checkpoint)] = new_init_check

            Match_History.append(Current_Match.copy())
            print(Match_History)
            print(f"Initial Self-Play Pairings: {Current_Match}")

        elif len(Checkpoints)==1:
            checkpoint = Checkpoints[0]
            print(f"\n++++++++++++++++++++++ Loading Checkpoint: {checkpoint}+++++++++++++++++++++\n")
            id = checkpoint.split('/')[1]
            checkpoint_name = checkpoint.split('/')[0]
            LoadCheckpoint(algorithm, checkpoint_name, 'team_0', id)
            print("\n++++++++++++++++++++++ Checkpoint Loaded +++++++++++++++++++++\n")

    def on_episode_created(self, *, episode, base_env, **kwargs):
        env = base_env.get_sub_environments()[0]
        if len(Checkpoints)>1:
            for t in Current_Match.keys():
                plane_model = int(Current_Match[t].split('P')[1])
                env.set_plane_model(t, plane_model)

    def on_train_result(self, *, algorithm, result, **kwargs):
        global Checkpoints, Current_Match, Ratings, Match_History

        if len(Checkpoints)>1:
            Checkpoints, Current_Match, Ratings = self.EvaluationCallback.on_train_result(algorithm=algorithm, result=result, **kwargs)
            self.WandbCallBack.on_train_result_SelfPlay(algorithm=algorithm, result=result, Ratings=Ratings,
                                            Current_Match=Current_Match, **kwargs)
            
            if algorithm.iteration % alg_config['checkpoint_freq'] == 0:
                Match_History.append(Current_Match.copy())
            
            if algorithm.iteration == alg_config['train_iterations']:
                plot_and_save_matching_history(Match_History, Ratings, save_dir=os.path.join(storage_path, RunName, "PLOTS"))
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
        "model": {"fcnet_hiddens": tune.grid_search(alg_config['fcnet_hiddens']), "fcnet_activation": tune.grid_search(alg_config['fcnet_activation'])},
    }),
    "team_1": (None, obs_space, act_space, {
        "model": {"fcnet_hiddens": tune.grid_search(alg_config['fcnet_hiddens']), "fcnet_activation": tune.grid_search(alg_config['fcnet_activation'])},
    }),
    "team_2": (None, obs_space, act_space, {
        "model": {"fcnet_hiddens": tune.grid_search(alg_config['fcnet_hiddens']), "fcnet_activation": tune.grid_search(alg_config['fcnet_activation'])},
    }),
    "team_3": (None, obs_space, act_space, {
        "model": {"fcnet_hiddens": tune.grid_search(alg_config['fcnet_hiddens']), "fcnet_activation": tune.grid_search(alg_config['fcnet_activation'])},
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
    .environment(env="aerial_battle", env_config={'reward_version': tune.grid_search(list(env_config['reward_versions'].keys()))})
    .training(
        train_batch_size=tune.grid_search(alg_config['batch_size_per_learner']),
        gamma=tune.grid_search(alg_config['gamma']),

        optimization_config = {
            'actor_learning_rate': tune.grid_search(alg_config['actor_learning_rate']),
            'critic_learning_rate': tune.grid_search(alg_config['critic_learning_rate']),
            'entropy_learning_rate': tune.grid_search(alg_config['entropy_learning_rate'])
            },
        initial_alpha = tune.grid_search(alg_config['initial_alpha']),
        tau = tune.grid_search(alg_config['tau']),
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