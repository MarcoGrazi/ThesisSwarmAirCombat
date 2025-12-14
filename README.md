# Aerial Battle – Multi-Agent RL Air Combat Framework

## Overview

This repository implements a **research-oriented simulation and training framework for multi-agent aerial combat**, designed to develop, train, and evaluate combinations of reinforcement learning (RL) policies and parametric fixed-wing aircraft models.

At its core, the project is a **tool for co-design**:

- **Control policies** learned via deep reinforcement learning  
- **Aircraft configurations**, including mass, inertia, aerodynamic coefficients, thrust limits, control authority, and vulnerability cones  

Rather than treating the aircraft as a fixed platform, the framework enables **systematic exploration of how airframe parameters interact with learned behavior**, under a shared combat environment and reward structure.

---

## Run Instructions:
Basic set up:
- go into Train_Run_config.py and change Wandb authentication key, artifacts folder and run_name
- If running an experiment with an initial checkpoint, be sure to create the folder run_name, and put inside the Checkpoint with name matching the one specified in the checkpoint_name part of the initial_checkpoint field
  
Different types of training can be run on this system:
- 1 vs dummy: ensure you specify 1 or none starting checkpoints, alive_agents_start = 1 and dummy settings are not 'none'
- 1 vs 1: just disable dummy settings by putting dummy_type = 'none'. You can choose to train both team's policy or just one: policies_to_train = ['team_0', 'team_1']
- 1 vs 1 Tournament: put multiple starting checkpoints in initial_checkpoints, this will start a self-play tournament. Make sure to check train_iteration and checkpoint_freq since this will determine after how many iterations you have an evaluation round, and how many there will be in the whole run. Also make sure to check num_evaluation_episodes is set according to the TrueSkill reference table
- 2 vs 2 (1 agent, 1 dummy): as a result of how the code is structured there is this possibility. From 1vs dummy set up, just set alive_agents_start = 2
- 2 vs 2: 1 vs 1 with alive agents_start = 2
- n vs n Tournament: basically 1 vs 1 tournament with alive_agents_start=n. Be careful though, agent_number_team must be >= n. Also consider that if the loaded checkpoints were trained with agent_number_team != current agent_number_team the observation spaces will not match, and thus it will be impossible to run the experiment
- n vs n vs n: by setting team_number > 2 we can model scenarios with more than 2 competing teams.
   
## Key Capabilities

The system supports:

- Multi-agent, multi-team aerial engagements  
- Self-play and league-based training with **TrueSkill** rating  
- Parametric aircraft models selectable per agent and per training round  
- Full export of telemetry, trajectories, rewards, and rendered videos  

This makes the framework suitable not only for training competitive agents, but also for **comparative evaluation of aircraft designs under identical learning and engagement conditions**.

---

## Modeling Assumptions and Simplifications

To keep the problem tractable and suitable for large-scale reinforcement learning, several **intentional simplifications** are applied.

### Fixed-Wing Dynamics

- Aircraft are modeled as **rigid bodies** with simplified aerodynamic force and moment models  
- High-fidelity effects such as compressibility, structural flexibility, or detailed propulsion dynamics are not modeled  

### Abstracted Sensing and Targeting

- Missile engagement is based on **geometric attack and defense cones**, relative orientation, and distance  
- There is no explicit radar, seeker, electronic warfare, or countermeasure modeling  

### No Explicit Missile Dynamics

- Missile flight is abstracted into a **probabilistic hit model**, driven by lock quality and engagement geometry  
- Individual missile trajectories are not simulated  

### Simplified Environment

- The environment is bounded but flat  
- No terrain masking, weather, or environmental obstacles are modeled  

### Policy-Level Control Abstraction

- Agents do not command individual control surfaces directly  
- Instead, they issue **high-level normalized commands** (directional intent and speed), tracked by an internal PID-controlled flight model  

These simplifications are deliberate.  
The goal is **not high-fidelity flight simulation**, but **consistent, controllable, and scalable experiments** where learning dynamics and design trade-offs can be studied in isolation.

---

## Research Context

This codebase was developed as part of an **academic research effort** focused on:

- Learning-based aerial combat  
- Self-play and league-based evaluation  
- Co-evolution of agents and parametric aircraft platforms  

### Thesis Reference

📄 **Thesis reference:**  
*(Link to thesis text will be added here)*

The thesis provides:

- Formal problem definition  
- Justification of modeling choices  
- Detailed discussion of reward shaping and engagement metrics  
- Experimental results and analysis  

For theoretical background and methodological motivation, the thesis should be considered the **primary reference**, with this repository serving as the **executable experimental platform**.

---

# Future Improvement Paths (Work for Evolution)

This project already integrates **custom fixed-wing physics, multi-agent RL, self-play leagues, and extensive artifact logging**.  
Future improvements should focus on **making evolution safer, faster, and easier to extend**, especially at scale.

---

## 1. Reduce Global Mutable State in Training

League state (checkpoint list, ratings, current match, match history) is currently managed through **module-level global variables** shared across callbacks.

### Recommended Evolution

Introduce a `LeagueManager` object responsible for:

- Checkpoint pool management  
- TrueSkill ratings  
- Current round pairings  
- Match history  
- Persistence (`league_state.json`)  

Callbacks should act as thin wrappers calling methods such as:

- `LeagueManager.update_round(...)`  
- `LeagueManager.sample_pairings(...)`  

### Why

- Easier recovery after crashes  
- Cleaner separation of concerns  
- Fewer side effects across Ray workers and trials  

---

## 2. Stop Encoding Metadata in Checkpoint Folder Names

League logic currently depends on parsing strings such as:

checkpoint_name_PplaneModel
checkpoint/team_x/plane_model


Renaming a folder can silently break training logic.

### Recommended Evolution

Store semantic information explicitly:

weights.pkl
meta.json # origin checkpoint, policy id, plane model, round number, timestamp


### Why

- Robust to renames and format changes  
- Clearer and more maintainable league logic  

---

## 3. Split Environment Responsibilities into Smaller Modules

`AerialBattle` currently handles:

- Observation construction  
- Missile and cone logic  
- Reward shaping  
- Termination rules  
- Rendering and plotting utilities  

### Recommended Evolution

Extract functionality into lightweight modules:

- `combat.py` – cones, tone updates, firing logic  
- `observations.py` – relative features, polar conversions, closure metrics  
- `rewards.py` – reward computation (version-driven)  
- `rendering.py` – pygame rendering and plotting utilities  
- `termination.py` – collision and safety termination rules  

### Why

- Lower cognitive load  
- Easier isolated testing  
- Reduced risk when modifying individual subsystems  

---

## 4. Decouple Reward Computation from Termination Side Effects

`get_individual_reward()` currently both:

- Computes reward components  
- Applies termination logic and kills aircraft  

### Recommended Evolution

Make reward functions **pure**:

return (
reward_scalar,
reward_components,
termination_flags,
termination_reason
)


Apply state-changing effects (kills, termination) centrally in `env.step()`.

### Why

- Clearer debugging (“why did this agent die?”)  
- Easier experimentation with termination rules  

---

## 5. Make Visualization Imports Optional and Local

Training workers do not need visualization libraries unless artifacts are being generated.

### Recommended Evolution

Move heavy imports inside the functions that use them:

- `render()`  
- `plot_telemetry()`  
- `render_trajectory()`  

Keep core environment imports minimal.

### Why

- Faster startup  
- Fewer dependency issues on headless or remote workers  
- Clear separation between training and analysis tooling  

---

## 6. Add a Lightweight Debug Run Path

At scale, rare failures (NaNs, unstable episodes, unexpected terminations) are costly to reproduce.

### Recommended Evolution

Add a debug mode that triggers on anomalies and automatically:

- Saves the last *N* timesteps of telemetry  
- Exports a short video or trajectory snapshot  
- Writes a compact JSON summary:
  - termination reason  
  - last action  
  - key observation slices  

### Why

- Faster diagnosis of failures  
- Turns silent training issues into actionable debugging data  

