Overview

This repository implements a research-oriented simulation and training framework for multi-agent aerial combat, designed to develop, train, and evaluate combinations of reinforcement learning (RL) policies and parametric fixed-wing aircraft models.

At its core, the project is a tool for co-design:

Control policies (learned via deep reinforcement learning), and

Aircraft configurations (mass, inertia, aerodynamic coefficients, thrust, control authority, vulnerability cones, etc.)

Rather than treating the aircraft as a fixed platform, the framework allows systematic exploration of how different airframe parameters interact with learned behavior, under the same combat and reward structure.

The system supports:

Multi-agent, multi-team aerial engagements

Self-play and league-based training with rating (TrueSkill)

Parametric aircraft models selectable per agent and per training round

Full telemetry, trajectory, reward, and video export for analysis

This makes the framework suitable not only for training competitive agents, but also for comparative evaluation of aircraft designs under identical learning and engagement conditions.

Modeling assumptions and simplifications

To keep the problem tractable and suitable for large-scale RL training, several intentional simplifications are applied:

Rigid-body fixed-wing dynamics
Aircraft are modeled as rigid bodies with simplified aerodynamic force and moment models. High-fidelity effects (compressibility, structural flexibility, detailed propulsion dynamics) are not modeled.

Abstracted sensing and targeting
Missile engagement is based on geometric attack and defense cones, relative orientation, and distance. There is no explicit radar, seeker, or countermeasure modeling.

No explicit missile dynamics
Missile flight is abstracted into a probabilistic hit model driven by lock quality and relative geometry, rather than simulating missile trajectories.

Simplified environment and terrain
The environment is bounded but flat, with no terrain masking, weather, or obstacles.

Policy-level control abstraction
Agents do not command individual control surfaces directly. Instead, they issue high-level normalized commands (directional intent and speed), which are tracked by an internal PID-controlled flight model.

These simplifications are deliberate: the goal is not high-fidelity flight simulation, but consistent, controllable, and scalable experiments where learning dynamics and design trade-offs can be studied in isolation.

Research context

This codebase was developed as part of an academic research effort focused on learning-based aerial combat and co-evolution of agents and platforms.

📄 Thesis reference:
(Link to thesis text will be added here)

The thesis provides:

Formal problem definition

Justification of modeling choices

Detailed discussion of reward shaping and engagement metrics

Experimental results and analysis

For theoretical background and methodological motivation, the thesis should be considered the primary reference, with this repository serving as the executable experimental platform.



Future Improvement Paths for this Code:

This project is already doing a lot (custom fixed-wing physics + multi-agent RL + self-play league + artifact logging). The next improvements should focus on making evolution safer and faster (fewer silent breaks, easier debugging, easier to add features).

1) Reduce global mutable state in training

Right now, league state (checkpoints list, ratings, current match, match history) is managed through module-level globals and shared across callbacks. This works until you introduce multiple trials, restarts, partial failures, or refactors.

Recommended evolution

Introduce a LeagueManager object that owns:

checkpoint pool

TrueSkill ratings

current round pairing

match history

persistence (league_state.json)

Callbacks should be thin wrappers calling LeagueManager.update_round(...), LeagueManager.sample_pairings(...), etc.

Why

Easier resuming after crashes

Cleaner separation of concerns

Fewer side effects across Ray workers/trials

2) Stop encoding metadata in checkpoint folder names

A lot of logic depends on parsing strings like somethingPplane_model or checkpoint/team_x/model. Renaming a folder can silently break the league.

Recommended evolution

Keep folder names human-friendly, but store meaning in a small metadata file next to weights:

weights.pkl

meta.json (origin checkpoint, policy id, plane model, round number, timestamp)

Why

Robust to renames and format changes

Makes league logic clearer and less fragile

3) Split environment responsibilities into small modules

AerialBattle currently contains: observation building, missile/cone logic, reward shaping, termination rules, and rendering/plotting utilities. This increases cognitive load and makes changes riskier.

Recommended evolution

Extract into small helper modules (simple functions are enough):

combat.py: cones, tone updates, fire/kill probability

observations.py: relative features / polar conversions / closure metrics

rewards.py: reward computation by version (config-driven)

rendering.py: pygame render + plotting helpers

termination.py: collision/out-of-bounds/safety termination rules

Why

Easier to modify one area without breaking others

Easier unit testing on isolated logic

4) Decouple reward computation from termination side-effects

get_individual_reward() currently does both:

compute reward components

apply termination logic (and kills the aircraft)

Recommended evolution

Make reward functions pure:

return (reward_scalar, reward_components, termination_flags, termination_reason)

Apply kills/termination state updates in one place (the env step), based on returned flags.

Why

Debugging becomes straightforward (“why did this agent die?”)

You can change termination rules without rewriting reward logic

5) Make visualization imports optional and local

Training workers don’t need pygame/plotly/pandas/matplotlib unless you explicitly render or export artifacts. Importing everything at module load makes headless training heavier and more fragile.

Recommended evolution

Move heavy imports inside the functions that need them:

render(), plot_telemetry(), render_trajectory(), etc.

Keep environment core imports minimal.

Why

Faster startup time

Fewer dependency issues on remote workers

Cleaner separation of training vs analysis tooling

6) Add a lightweight “debug run” path

When you scale training, rare failures (NaNs, weird terminations, unstable episodes) become expensive to reproduce.

Recommended evolution

Add a debug mode that triggers on anomaly:

saves last N timesteps of telemetry

saves a short video/trajectory snapshot

writes a compact JSON summary (termination_reason, last action, last obs slice)

Why

Faster iteration when something breaks

Makes training failures actionable instead of mysterious
