import pygame
import yaml
import os

from fontTools.ttLib.tables.T_S_I__2 import table_T_S_I__2
from matplotlib.cm import get_cmap
import imageio
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import matplotlib.pyplot as plt
plt.switch_backend("Agg")
import plotly.graph_objects as go
from ray.rllib.utils.typing import AgentID

from Physics import FixedWingAircraft
import gymnasium as gym


class Aircraft:
    def __init__(self, UAV_config, rho, g, frequency, team_id, cone):
        # Initialize the aircraft's physical flight model
        self.physical_model = FixedWingAircraft(UAV_config, rho, g, frequency)

        # Missile system state
        self.missile_tone_attack = 0                # Readiness to fire (attack tone level)
        self.missile_tone_defence = 0               # Warning level of being targeted
        self.missile_target = "none"                # Target ID string or "base"/"none"

        # Status flags
        self.live = 1                               # 1 = alive, 0 = destroyed
        self.team = team_id                         # Team affiliation

        # PID control gains for basic flight controls
        self.gains = {
            'AoA':      {'kp': 5, 'ki': 0.1, 'kd': 0.8},   # Angle of Attack control
            'sideslip': {'kp': 5, 'ki': 0.1, 'kd': 1.2},   # Lateral balance / sideslip
            'roll':     {'kp': 5, 'ki': 0.1, 'kd': 1.0},   # Roll angle control
            'speed':    {'kp': 80, 'ki': 0.8, 'kd': 0.0}   # Throttle/speed regulation
        }

        # Stores the history of key flight data for analysis or plotting
        self.agent_telemetry = {
            'position': [],
            'orientation': [],
            'velocity': [],
            'angular_velocity': [],
            'acceleration': [],
            'AoA': [],
            'sideslip': [],
            'force': [],
            'moment': [],
            'commands': []
        }

        # State variables for PID controllers
        self.prev_errors = {'AoA': 0, 'sideslip': 0, 'roll': 0, 'speed': 0}
        self.integrals = {'AoA': 0, 'sideslip': 0, 'roll': 0, 'speed': 0}
        self.dt = 1.0 / frequency                    # Simulation time step duration (s)

        # Engagement parameters
        self.attack_vulnerability_cone = cone       # (angle_deg, min_dist, max_dist)

        # Dummy/AI mode attributes (used in scripted dummy behavior)
        self.dummy_type = "none"                    # "none" or dummy behavior name
        self.turn_radius = 0                        # Dummy circular turn radius (if used)
        self.direction = 0                          # Dummy turn direction: +1 or -1

    def set_dummy(self, type, turn_radius='500', direction=1.0):
        """
        Configure the aircraft as a scripted dummy target for simplified behavior.

        Args:
            type (str): Type of dummy behavior (e.g., "circle", "straight", etc.)
            turn_radius (float or str): Radius (in meters) of the circular path, if applicable.
            direction (float): Turning direction (+1 for clockwise, -1 for counter-clockwise).
        """
        self.dummy_type = type                  # Activate dummy behavior mode
        self.turn_radius = turn_radius          # Store specified turn radius
        self.direction = direction              # Direction of turn (clockwise or counter-clockwise)

    def is_dummy(self):
        """
        Check whether the aircraft is operating in dummy mode.

        Returns:
            bool: True if dummy behavior is active, False otherwise.
        """
        return self.dummy_type != 'none'

    def reset(self, position, orientation, speed, alive):
        """
        Resets the aircraft's state and telemetry to initial values.

        Args:
            position (array-like): Initial 3D position in world coordinates.
            orientation (array-like): Initial orientation (Euler angles: roll, pitch, yaw).
            speed (float): Initial forward speed (m/s).
            alive (bool or int): Whether the aircraft starts alive (1) or dead (0).
        """

        # Reset physical model state (position, orientation, velocity, etc.)
        self.physical_model.reset(position, orientation, speed)

        # Set aircraft alive status and missile system state
        self.live = alive
        self.missile_tone_attack = 0
        self.missile_tone_defence = 0
        self.missile_target = "none"

        # Reset PID controller state
        self.prev_errors = {'AoA': 0, 'sideslip': 0, 'roll': 0, 'speed': 0}
        self.integrals = {'AoA': 0, 'sideslip': 0, 'roll': 0, 'speed': 0}

        # Clear telemetry and initialize it with the current state from the physical model
        self.agent_telemetry = {
            'position': [],
            'orientation': [],
            'velocity': [],
            'angular_velocity': [],
            'acceleration': [],
            'AoA': [],
            'sideslip': [],
            'force': [],
            'moment': [],
            'commands': []
        }

        # Append the latest telemetry state from the physical model as the first recorded value
        for key in self.agent_telemetry.keys():
            self.agent_telemetry[key].append(self.physical_model.getTelemetry()[key][-1])

    def step(self, action, frequency_factor):
        """
        Advance the aircraft state by integrating control over multiple physics steps.

        Args:
            action (list or np.array): Target velocity vector in the aircraft **body frame**.
                                    Typically [forward_speed, vertical_angle, lateral_angle]
            frequency_factor (int): Number of physics updates per control step.
        """

        for i in range(frequency_factor):
            # --- Translate velocity vector in body frame to desired flight angles and speed ---
            # Expected: AoA (angle of attack), sideslip angle, roll angle, and speed (magnitude)
            AoA, sideslip, roll, throttle = self.Action_Translation_Layer(action[:-1], (i/frequency_factor))

            # --- Compute actuator inputs using PID control for each controlled dimension ---
            # Returns control surface commands: throttle, elevon, aileron, rudder
            t, e, a, r = self.PID_Control([AoA, sideslip, roll, throttle])

            # --- Choose control method depending on whether this aircraft is autonomous or scripted dummy ---
            if self.dummy_type == "none":
                # Live (non-dummy) aircraft controlled by PID and physics
                self.physical_model.step(t, e, a, r, action)
            else:
                # Dummy aircraft follows a scripted path (e.g., constant turn radius)
                self.physical_model.dummy_step(self.dummy_type, self.turn_radius, self.direction)

        # --- Log updated telemetry after the control step ---
        for key in self.agent_telemetry.keys():
            # Store latest physical telemetry (e.g., position, velocity, orientation, etc.)
            self.agent_telemetry[key].append(self.physical_model.getTelemetry()[key][-1])

    def Action_Translation_Layer(self, action, manouvre_progress):
        """
        Translates a high-level action (target velocity vector in body frame)
        into AoA, roll, and speed setpoints. All outputs are normalized in [-1, 1].

        Parameters:
            action (np.ndarray): [up_angle_norm, side_angle_norm, speed_norm]
                - up_angle_norm ∈ [-1, 1]: vertical angle (pitch) offset from forward
                - side_angle_norm ∈ [-1, 1]: lateral (yaw) offset from forward
                - speed_norm ∈ [-1, 1]: target speed (normalized)

        Returns:
            list: [AoA_norm, sideslip_norm, roll_norm, speed_norm]
                - AoA_norm: normalized angle of attack setpoint
                - sideslip_norm: normalized sideslip angle (0 if unused)
                - roll_norm: normalized roll angle setpoint
                - speed_norm: passed through unchanged
        """

        # === 1. Convert normalized input angles to radians ===
        max_angle_rad = np.deg2rad(30)  # maximum angular offset allowed by agent
        v_up   = action[0] * max_angle_rad    # vertical offset from forward
        v_side = action[1] * max_angle_rad    # lateral offset from forward
        v_speed = action[2]                  # speed stays normalized for now

        # === 2. Construct target direction unit vector in body frame ===
        # Body frame: X = forward, Y = right, Z = down
        vx = np.cos(v_up) * np.cos(v_side)  # forward component
        vy = np.cos(v_up) * np.sin(v_side)  # rightward component
        vz = np.sin(v_up)                   # downward component
        # The vector [vx, vy, vz] points in the direction the agent wants to fly

        # === 3. Mode switch: use direct angles for large requests ===
        # If either angular offset is large (> ~5°), use direct up/side as AoA/sideslip
        if v_up < 0 and abs(v_side) < (np.pi/10):
            # Direct control mode (use the agent's angles directly)
            AoA_rad = v_up
            sideslip_rad = v_side
            roll_rad = 0.0  # assume level bank in this mode
        else:
            # Full 3D vector tracking mode
            # AoA = angle between forward and target vector (from forward offset)
            AoA_rad = np.arccos(vx)

            # No sideslip control in this mode
            sideslip_rad = 0.0

            # Roll = angle between vertical (Z+) and projection of v_dir in Y–Z plane
            # This aligns the lift vector with the target direction
            roll_rad = np.arctan2(vy, vz)  # (signs assumed to match your control system)

            # output only roll command in the early phase of the manouvre, to prevent PID instability
            if manouvre_progress < 0.3:
                AoA_rad = 0

        # === 4. Normalize all outputs to [-1, 1] (compatible with PID_Control scaling)===
        AoA_norm = AoA_rad / (np.pi/6)          # range: [-1, 1] corresponds to ±30°
        sideslip_norm = sideslip_rad / (np.pi/10)
        roll_norm = roll_rad / np.pi        # range: [-1, 1] corresponds to ±180°
        return [AoA_norm, sideslip_norm, roll_norm, v_speed]

    def PID_Control(self, action):
        """
        PID controller for converting target flight states into control surface commands.

        Inputs:
            action: list or array of 4 elements
                [target_AoA, target_sideslip, target_roll, target_speed]
                All values normalized in range [-1, 1], except throttle ∈ [0, 1]

        Outputs:
            Tuple (throttle, elevator, aileron, rudder), where:
                - Throttle ∈ [0, 1]
                - Elevator, Aileron, Rudder ∈ [-1, 1]
        """

        # === Unpack and scale desired targets ===
        target_AoA, target_sideslip, target_roll, target_speed = action
        target_speed = np.clip(target_speed, 0.4, 1)  # Avoid stall speeds

        # Scale normalized values to physical units:
        target_AoA *= 30               # Angle of attack in degrees
        target_sideslip *= 5         # Sideslip stays in small angle range
        target_roll *= 180            # Roll target in degrees
        target_speed *= 343           # Convert normalized speed to m/s (approx speed of sound)

        # === Retrieve current state from telemetry ===
        telemetry = self.physical_model.getTelemetry()
        current_AoA = np.rad2deg(telemetry['AoA'][-1])
        current_sideslip = np.rad2deg(telemetry['sideslip'][-1])
        current_roll = np.rad2deg(telemetry['orientation'][-1][0])  # Roll angle in deg
        current_speed = np.linalg.norm(telemetry['velocity'][-1])   # Magnitude of velocity vector

        # === Compute PID error terms ===
        errors = {
            'AoA': target_AoA - current_AoA,
            'sideslip': target_sideslip - current_sideslip,
            'roll': ((target_roll - current_roll) + 180) % 360 - 180,  # Handle wraparound
            'speed': target_speed - current_speed
        }

        outputs = {}  # PID raw outputs for each control loop

        # === Compute PID control signals for each axis ===
        for key in errors:
            self.integrals[key] += errors[key] * self.dt
            derivative = (errors[key] - self.prev_errors[key]) / self.dt
            gains_k = self.gains[key]

            outputs[key] = (
                gains_k['kp'] * errors[key] +
                gains_k['ki'] * self.integrals[key] +
                gains_k['kd'] * derivative
            )
            self.prev_errors[key] = errors[key]

        # === Normalize outputs to actuator limits ===
        commands = {
            'elevator': -np.clip(outputs['AoA'] / 40, -1, 1),           # Inverted due to control surface direction
            'rudder': -np.clip(outputs['sideslip'] / 40, -1, 1),
            'aileron': np.clip(outputs['roll'] / 40, -1, 1),
            'throttle': np.clip(outputs['speed'] / 343, 0.0, 1.0)
        }

        return commands['throttle'], commands['elevator'], commands['aileron'], commands['rudder']

    def get_max_acc(self):
            """
            Returns the maximum achievable acceleration for missile evasion calculations.
            """
            return self.physical_model.get_max_acc()

    def kill(self):
        """
        Sets the aircraft as no longer alive (e.g., hit by a missile).
        """
        self.live = 0

    def is_alive(self):
        """
        Checks whether the aircraft is still alive.
        """
        return self.live

    def get_team(self):
        """
        Returns the team ID this aircraft belongs to.
        """
        return self.team

    def set_missile_tone_attack(self, tone, target):
        """
        Sets the current missile lock tone and the designated target.
        """
        self.missile_tone_attack = tone
        self.missile_target = target

    def get_missile_tone_attack(self):
        """
        Returns the current missile tone and target.
        """
        return self.missile_tone_attack, self.missile_target

    def set_missile_tone_defence(self, tone):
        """
        Sets the missile defence tone (based on being targeted).
        """
        self.missile_tone_defence = tone

    def get_missile_tone_defence(self):
        """
        Returns the current defensive missile tone.
        """
        return self.missile_tone_defence

    def get_cone(self):
        """
        Returns the attack/vulnerability cone parameters:
        [cone_angle_deg, min_dist, max_dist]
        """
        return self.attack_vulnerability_cone

    def get_pos(self):
        """
        Returns the current 3D position of the aircraft in the environment.
        """
        return self.physical_model.get_pos()

    def get_distance_from_centroid(self, bases):
        """
        Computes 2D (XY-plane) distance from the aircraft to the centroid of all bases.
        Used to evaluate positioning in the environment.
        """
        centroid = np.mean(bases, axis=0)[:2]
        return np.linalg.norm(centroid - self.get_pos()[:2])

    def get_absolute_vel(self):
        """
        Returns the absolute velocity of the aircraft in the inertial frame.
        """
        return self.physical_model.get_absolute_velocity()

    def get_physics_telemetry(self):
        """
        Returns the raw telemetry dictionary from the physical model
        (used mainly for rendering).
        """
        return self.physical_model.getTelemetry()

    def get_agent_telemetry(self):
        """
        Returns the internal telemetry dictionary recorded per timestep
        (used for logging and training).
        """
        return self.agent_telemetry

    def get_own_data(self, max_speed, max_size, bases):
        """
        Compiles a normalized observation vector for the agent.
        This includes:
        - altitude
        - acceleration
        - velocity
        - orientation (cos & sin)
        - angular velocity
        - AoA and sideslip (cos & sin)
        - distance to centroid of all bases
        - total specific energy
        - previous control commands
        - current missile tone (attack + defence)

        Args:
            max_vel (float): used to normalize linear velocity
            max_size (float): used to normalize spatial distances
            bases (List): list of all base positions in the scenario

        Returns:
            list: flattened, normalized observation vector (~28 values)
        """
        telemetry = self.get_agent_telemetry()
        current_state = []

        # Altitude (z-axis is positive down)
        current_state.append(telemetry['position'][-1][2] / 10000)  # 1 value

        # Acceleration (normalized by 20g)
        current_state.extend(np.array(telemetry["acceleration"][-1]) / (20 * 9.8))  # 3 values

        # Velocity (normalized)
        current_state.extend(np.array(telemetry["velocity"][-1]) / max_speed)  # 3 values

        # Orientation (cos and sin to avoid discontinuities)
        current_state.extend(np.cos(telemetry["orientation"][-1]))  # 3 values
        current_state.extend(np.sin(telemetry["orientation"][-1]))  # 3 values

        # Angular velocity (normalized)
        current_state.extend(np.array(telemetry["angular_velocity"][-1]) / 10)  # 3 values

        # Angle of Attack (as cos/sin)
        current_state.append(np.cos(telemetry["AoA"][-1]))  # 1
        current_state.append(np.sin(telemetry["AoA"][-1]))  # 1

        # Sideslip angle (as cos/sin)
        current_state.append(np.cos(telemetry["sideslip"][-1]))  # 1
        current_state.append(np.sin(telemetry["sideslip"][-1]))  # 1

        # Distance from base centroid (normalized)
        current_state.append(self.get_distance_from_centroid(bases) / max_size)  # 1

        # Missile tones
        current_state.append(self.missile_tone_attack)   # 1
        current_state.append(self.missile_tone_defence)  # 1

        return current_state



class AerialBattle(MultiAgentEnv):
    def __init__(self, env_config, UAV_config, reward_version=1, discretize=True):
        """
        Initialize the AerialCombat environment for multi-agent reinforcement learning.

        :param env_config: Dictionary of environment-specific parameters
        :param UAV_config: Dictionary with UAV-specific configuration (e.g., mass, aero coefficients)
        :param discretize: Boolean flag for discretized control (unused here but may affect action mapping)
        """
        super().__init__()

        # === Pygame Initialization (visualization only, one-time) ===
        if not hasattr(self, "_pygame_initialized"):
            pygame.init()
            self._screen = pygame.Surface(env_config['screen_size'])
            self._clock = pygame.time.Clock()
            self._agent_trails = {}
            self._pygame_initialized = True
            pygame.font.init()
            self.font = pygame.font.SysFont('Arial', 14)

        # === Core Simulation Parameters ===
        self.physics_frequency = env_config["physics_frequency"]           # Hz for physics engine
        self.action_frequency = env_config["action_frequency"]             # Hz for agent actions
        self.max_steps = env_config["max_episode_length"]                  # Max steps per episode
        self.frequency_factor = self.physics_frequency // self.action_frequency
        self.g = env_config['g']
        self.reward_version = reward_version

        # === Missile and Combat Configs ===
        self.missile_speed = env_config["missile_speed"]
        self.max_acc_missile = env_config["missile_max_acceleration"]
        self.stepwise_tone_increment = env_config['stepwise_tone_increment']
        self.collision_distance = env_config["collision_distance"]
        self.bases_vulnerability_distance = env_config["base_vulnerability_distance"]

        # === World and Agent Configuration ===
        self.env_size = env_config["env_size"]
        self.max_size = env_config["max_size"]
        self.min_bases_distance = env_config["min_bases_distance"]
        self.num_teams = env_config['team_number']
        self.num_agents_team = env_config["agent_number_team"]
        self.alive_agents_start = env_config['alive_agents_start']
        self.uav_mass = UAV_config['mass']
        self.discretize = discretize

        # === Control Resolution Parameters (for discretized or fine-grained action mapping) ===
        self.speed_step = env_config['speed_step']
        self.UpAngle_step = env_config['UpAngle_step']
        self.SideAngle_step = env_config['SideAngle_step']

        # === Dummy Flight Mode Parameters (for scripted behavior or non-learning actors) ===
        self.dummy = env_config["dummy"]
        self.turn_radius = env_config["dummy_turn_radius"]
        self.direction = env_config["dummy_direction"]

        # === Tracking + Stats ===
        self.agent_report = env_config["agent_report_name"]
        self.episode_rewards = {}     # Dict to accumulate rewards per agent
        self.episode_steps = 0
        self.attack_metric = 0
        self.kill_metric = 0

        # === Aircraft/Agent Initialization ===
        self.possible_agents = []     # List of all agent names
        self.Aircrafts = []           # List of Aircraft objects
        self.observation_spaces = {}  # Dict: agent_name → observation space
        self.action_spaces = {}       # Dict: agent_name → action space
        self.bases = []
        self.spawning_distance = env_config['spawning_distance']
        self.spawning_orientations = env_config['spawning_orientations']

        for i in range(self.num_teams):
            for j in range(self.num_agents_team):
                agent_name = f"agent_{i}_{j}"
                self.possible_agents.append(agent_name)

                # === Instantiate Aircraft ===
                self.Aircrafts.append(
                    Aircraft(
                        UAV_config,
                        env_config['rho'],
                        env_config['g'],
                        env_config['physics_frequency'],
                        i,  # team index
                        env_config['attack_vulnerability_cone']
                    )
                )

                # === Define Observation Space ===
                # obs = [own data (23)
                #        + 17×(other agents) [polar rel pos + polar_rel_vel + closure + tracking + flags]
                obs_dim = 23 + 17 * ((self.num_teams * self.num_agents_team) - 1)
                self.observation_spaces[agent_name] = gym.spaces.Box(
                    low=-1.5, high=1.5, shape=(obs_dim,), dtype=np.float64
                )

                # === Define Action Space ===
                # 3D continuous action: Up Angle, Side Angle, Speed in Body Frame + fire command
                self.action_spaces[agent_name] = gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(4,), dtype=np.float64
                )

        self.agents = self.possible_agents.copy()  # Initialize agents for current episode

    def get_observation_space(self, team_id):
        """
        Return the observation space for a given agent.

        :param team_id: String ID of the agent (e.g., "agent_0_1")
        :return: gym.spaces.Box representing the agent's observation space
        """
        return self.observation_spaces[team_id]

    def get_action_space(self, team_id):
        """
        Return the action space for a given agent.

        :param team_id: String ID of the agent (e.g., "agent_1_2")
        :return: gym.spaces.Box representing the agent's action space
        """
        return self.action_spaces[team_id]
    
    def point_on_circumference(self, x0, y0, r, theta):
        x = x0 + r * np.cos(theta)
        y = y0 + r * np.sin(theta)
        return (x, y)

    def init_airplane(self, aircraft, alive, testing, team):
        """
        Initialize (or respawn) an aircraft with randomized position, orientation, and speed,
        depending on its team, alive status, and testing mode.

        Parameters:
            aircraft : Aircraft object to initialize/reset
            alive    : bool — True if active (combat-ready), False if dead/spectator
            testing  : bool — If True, use deterministic spawn altitude for evaluation
        """

        centroid = np.mean(self.bases, axis=0)
        centroid[2] = self.env_size[2]/2
        delta = (2 * np.pi) / self.num_teams

        if alive:
            # === Alive agent: spawn near own base, inside vulnerability radius ===

            if not testing:
                # Training mode — more randomized spawn
                max_spawn_distance = self.spawning_distance

                # Random XY position around the base, clipped to stay inside map bounds
                x, y = self.point_on_circumference(centroid[0], centroid[1],
                                                    max_spawn_distance, team*delta)
                x = np.clip(x, 100, self.env_size[0]-100)
                y = np.clip(y, 100, self.env_size[0]-100)
                z = -self.env_size[2] / 2  # Midpoint in altitude (Z+ down) # Spawn within combat area

            else:
                # Testing mode — predictable altitude, closer to mid-range
                max_spawn_distance = self.spawning_distance

                # Random XY position around the base, clipped to stay inside map bounds
                x, y = self.point_on_circumference(centroid[0], centroid[1],
                                                    max_spawn_distance, team*delta)
                x = np.clip(x, 100, self.env_size[0]-100)
                y = np.clip(y, 100, self.env_size[0]-100)
                z = -self.env_size[2] / 2  # Midpoint in altitude (Z+ down) # Spawn within combat area

            rand_pos = [x, y, z]

        else:
            # === Dead or inactive agent: spawn within combat area ===
            rand_pos = [
                np.random.uniform(100, self.env_size[0] - 100),  # x position
                np.random.uniform(100, self.env_size[1] - 100),  # y position
                np.random.uniform(-(self.env_size[2] - 1000), -1000)  # Spawn far above
            ]


        # Compute orientation to target base (Euler angles or equivalent)
        rand_orient = self.orientation_to_target(rand_pos, centroid)
        discrete_pitch, discrete_yaw = self.spawning_orientations
        # Random roll ∈ [-10°, +10°]
        rand_orient[0] = np.deg2rad(np.random.uniform(-10, 10))  # roll

        # Discrete pitch from predefined list (e.g., [0, 5, -5])
        rand_orient[1] = np.deg2rad(np.random.choice(discrete_pitch))  # pitch

        # Add yaw offset to prevent identical spawn orientations
        rand_orient[2] += np.deg2rad(np.random.choice(discrete_yaw))   # yaw
        

        # === Initial airspeed between 150–200 m/s ===
        rand_speed = np.random.choice([120, 140, 180])

        # === Final step: apply the randomized state to the aircraft ===
        aircraft.reset(rand_pos, rand_orient, rand_speed, alive)

    def reset(self, seed=42, testing=False, options=None):
        """
        Reset the environment to its initial state.

        This includes:
        - Placing team bases with spacing constraints
        - Randomly selecting which aircraft are active
        - Initializing all aircraft positions, orientations, and status
        - Optionally inserting a dummy agent for scripted behavior
        - Resetting episode state and returning the initial observations

        :param seed: Random seed (unused here but can be wired in for reproducibility)
        :param options: Optional config dict (unused here, Gym API compatible)
        :return: (obs_dict, info_dict)
        """
        self.episode_steps = 0
        self.attack_metric = 0
        self.kill_metric = 0

        # === Place team bases ===
        # Ensure bases are separated by min_bases_distance
        self.bases.clear()
        for team in range(self.num_teams):
            delta = (2 * np.pi) / self.num_teams
            x, y = self.point_on_circumference(self.env_size[0]/2, self.env_size[1]/2,
                                                self.min_bases_distance, team*delta)
            z = 0
            self.bases.append(np.array([x, y, z]))

        # === Reset episode rewards ===
        for agent_id in self.possible_agents:
            self.episode_rewards[agent_id] = []

        # === Determine which agents are initially alive ===
        # alive_masks[team][agent] = 1 if active
        alive_masks = np.zeros((self.num_teams, self.num_agents_team))
        for _ in range(self.alive_agents_start):
            for t in range(self.num_teams):
                r_agent = np.random.randint(0, self.num_agents_team)
                alive_masks[t][r_agent] = 1  # Random alive agent per team

        # === Initialize aircraft states ===
        for t in range(self.num_teams):
            for a in range(self.num_agents_team):
                index = t * self.num_agents_team + a
                aircraft = self.Aircrafts[index]
                self.init_airplane(aircraft, alive_masks[t][a], testing=testing, team=t)

        # === Optionally convert one aircraft to a dummy agent ===
        if self.dummy != "none":
            t = np.random.randint(1, self.num_teams)  # Avoid team 0 which is the agent team by default
            alive_indexes = alive_masks[t].nonzero()
            a = alive_indexes[0][np.random.randint(0, len(alive_indexes[0]))]  # Choose a random alive agent

            index = t * self.num_agents_team + a
            aircraft = self.Aircrafts[index]

            self.init_airplane(aircraft, alive=True, testing=testing, team=t)
            aircraft.set_dummy(self.dummy, self.turn_radius, self.direction)

        # === Return initial observation and info ===
        return self.get_obs(), {'__common__': {'attack_steps' : self.attack_metric, 'kills': self.kill_metric}}

    def get_agent_ids(self):
        """
        Return a list of all agent IDs in the environment.

        :return: List of agent names (e.g., ["agent_0_0", "agent_1_2", ...])
        """
        return self.possible_agents


    def orientation_to_target(self, position, target_position):
        """
        Computes the orientation (roll, pitch, yaw) needed to face from `position`
        toward `target_position`. Roll is set to 0 since it's not required for basic orientation.

        Args:
            position (np.array): The current 3D position [x, y, z]
            target_position (np.array): The 3D target position [x, y, z]

        Returns:
            np.array: [roll, pitch, yaw] orientation in radians
        """

        # Compute the direction vector from current position to target
        direction = target_position - position

        # Normalize the direction vector to get unit vector
        norm_dir = direction / np.linalg.norm(direction)
        dx, dy, dz = norm_dir

        # Compute yaw (heading), angle around Z axis (positive yaw = left turn)
        yaw = np.arctan2(dy, dx)

        # Compute pitch (elevation), angle around Y axis
        # Negate dz because positive Z is *downward* in this convention
        pitch = np.arcsin(-dz)

        # Roll is not used for pointing direction, so it is set to 0
        roll = 0.0

        # Return the orientation vector in radians
        return np.array([roll, pitch, yaw])

    def relative_pos(self, i, j, type):
        """
        Computes the relative position vector between aircraft i and:
        - aircraft j (if type == "aircraft"), or
        - base j (if type == "base").

        Args:
            i (int): Index of the reference aircraft.
            j (int): Index of the target aircraft or base.
            type (str): Either 'aircraft' or 'base'.

        Returns:
            np.ndarray: Relative position vector (3D) from i to j.
        """
        rel_pos = []

        if type == "aircraft":
            # Relative position from aircraft i to aircraft j
            rel_pos = np.array(self.Aircrafts[j].get_pos()) - np.array(self.Aircrafts[i].get_pos())
        
        elif type == "base":
            # Relative position from aircraft i to base j
            rel_pos = np.array(self.bases[j]) - np.array(self.Aircrafts[i].get_pos())

        # Clean up small numerical noise
        rel_pos = np.where(np.abs(rel_pos) < 1e-6, 0.0, rel_pos)

        return rel_pos

    def relative_polar_pos(self, i, j, type):
        """
        Converts the relative position vector between two objects (aircraft or base)
        into a normalized polar representation: [r, cos(θ), sin(θ), cos(φ), sin(φ)].

        Args:
            i (int): Index of reference aircraft.
            j (int): Index of target (aircraft or base).
            type (str): 'aircraft' or 'base'.

        Returns:
            list: Normalized distance and encoded elevation/azimuth angles.
        """
        # Get relative position vector from aircraft i to object j
        rel_pos = self.relative_pos(i, j, type)
        x, y, z = rel_pos

        # Euclidean distance between the two positions
        r = np.linalg.norm(rel_pos)

        if r < 1e-6:
            # Prevent division by zero or undefined angles
            theta = 0.0
            phi = 0.0
        else:
            # θ: Elevation from forward axis (X), 0 = ahead, π = behind
            theta = np.arccos(z / r)

            # φ: Azimuth (angle around X axis, from X towards Y/Z plane)
            phi = np.arctan2(y, x)

        # Output normalized distance and sin/cos of angles (good for ML)
        return [r / self.max_size, np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)]

    def relative_vel(self, i, j, type):
        """
        Compute the relative velocity vector from aircraft i to another object j.

        Args:
            i (int): Index of reference aircraft.
            j (int): Index of target (aircraft or base).
            type (str): Type of target: "aircraft" or "base".

        Returns:
            np.ndarray: 3D relative velocity vector.
        """
        rel_vel = []

        if type == "aircraft":
            # Relative velocity: target's velocity - own velocity
            rel_vel = np.array(self.Aircrafts[j].get_absolute_vel()) - np.array(self.Aircrafts[i].get_absolute_vel())
        elif type == "base":
            # Base assumed stationary, so relative velocity is just negative of own velocity
            rel_vel = np.zeros(3) - np.array(self.Aircrafts[i].get_absolute_vel())

        # Zero out very small values to avoid floating point noise
        rel_vel = np.where(np.abs(rel_vel) < 1e-6, 0.0, rel_vel)

        return rel_vel

    def relative_polar_vel(self, i, j, type):
        """
        Convert relative velocity vector to polar coordinates for agent i relative to j.

        Args:
            i (int): Index of the reference aircraft.
            j (int): Index of the target aircraft or base.
            type (str): "aircraft" or "base".

        Returns:
            List[float]: [normalized_speed, cos(θ), sin(θ), cos(φ), sin(φ)]
                        where θ is inclination from Z, and φ is azimuth in XY plane.
        """
        rel_vel = self.relative_vel(i, j, type)
        x, y, z = rel_vel

        r = np.linalg.norm(rel_vel)  # Speed magnitude

        if r == 0:
            # No relative motion → direction is undefined; set to zero safely
            theta = 0.0
            phi = 0.0
        else:
            # θ: Inclination from Z-axis
            theta = np.arccos(z / r)

            # φ: Azimuth in XY plane
            phi = np.arctan2(y, x)

        # Normalize speed by speed of sound (343 m/s) and convert to polar components
        return [r / 343, np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)]
    
    def relative_polar_pos_Body(self, i, j, type):
        """
        Computes the relative polar position of agent j w.r.t agent i,
        expressed in the **body frame** of agent i.

        Args:
            i (int): Index of reference aircraft (whose frame is used)
            j (int): Index of target (aircraft or base)
            type (str): Either 'aircraft' or 'base'

        Returns:
            list: [normalized distance, cos(θ), sin(θ), cos(φ), sin(φ)]
                θ: elevation from X-axis in body frame
                φ: azimuth angle around X-axis in YZ plane
        """
        # Get relative position in global frame
        rel_pos = self.relative_pos(i, j, type)  # 3D vector

        # Get orientation of aircraft i
        orientation = self.Aircrafts[i].get_physics_telemetry()['orientation'][-1]
        yaw, pitch, roll = orientation[2], orientation[1], orientation[0]

        # Rotate rel_pos into the aircraft body frame (vehicle-to-body)
        R = self.vehicle_to_body(roll, pitch, yaw) # Note: pitch = AoA, yaw = sideslip in this context
        rel_body = R @ rel_pos
        x, y, z = rel_body

        # Compute polar coordinates in body frame
        r = np.linalg.norm(rel_body)

        if r < 1e-6:
            theta = 0.0
            phi = 0.0
        else:
            theta = np.arccos(z / r)      # Elevation from forward (x)
            phi = np.arctan2(y, x)        # Azimuth in x-y plane

        return [r / self.max_size, np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)]
    
    def relative_polar_vel_Body(self, i, j, type):
        """
        Computes the relative polar velocity of agent j w.r.t. agent i,
        expressed in the **body frame** of agent i.

        Args:
            i (int): Index of reference aircraft (whose frame is used)
            j (int): Index of target (aircraft or base)
            type (str): 'aircraft' or 'base'

        Returns:
            list: [normalized speed, cos(θ), sin(θ), cos(φ), sin(φ)]
                θ: elevation from X-axis (forward) in body frame
                φ: azimuth in YZ plane of body frame
        """
        # Get relative velocity in global frame
        rel_vel = self.relative_vel(i, j, type)

        # Get orientation of aircraft i
        orientation = self.Aircrafts[i].get_physics_telemetry()['orientation'][-1]
        roll, pitch, yaw = orientation

        # Rotate velocity vector into body frame
        R = self.vehicle_to_body(roll, pitch, yaw)
        rel_body_vel = R @ rel_vel
        x, y, z = rel_body_vel

        # Compute magnitude and polar angles
        speed = np.linalg.norm(rel_body_vel)

        if speed < 1e-6:
            theta = 0.0
            phi = 0.0
        else:
            theta = np.arccos(z / speed)  # Elevation from body X-axis
            phi = np.arctan2(y, x)        # Azimuth in body YZ plane

        return [speed / 343, np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)]
    
    def get_obs(self):
        """
        Construct the full observation dictionary for all alive agents.

        Each observation vector includes:
        - Own aircraft state (normalized)
        - Relative polar positions, closure rates, and tracking info to all other agents
        - Relative polar positions and velocities to all bases
        - Friend-or-foe (FoF) flags for both aircraft and bases

        :return: Dictionary of {agent_id: observation_vector (np.array)}
        """
        observations = {}

        for i, i_agent in enumerate(self.possible_agents):
            if self.Aircrafts[i].is_alive():
                # === Own state features ===
                # Includes normalized internal telemetry + environment scaling factors
                i_obs = self.Aircrafts[i].get_own_data(
                    max_speed=343,         # Max speed normalization constant [m/s]
                    max_size=self.max_size,
                    bases=self.bases
                )

                # === Add info about other aircraft ===
                for j, j_agent in enumerate(self.possible_agents):
                    if j_agent != i_agent:
                        # Relative position (polar)
                        i_obs.extend(self.relative_polar_pos_Body(i, j, 'aircraft'))

                        i_obs.extend(self.relative_polar_vel_Body(i, j, 'aircraft'))

                        # Closure rate (relative velocity along line of sight)
                        i_obs.append(self.get_closure_rate_norm(self.Aircrafts[i], self.Aircrafts[j]))

                        # Track and adverse angles (cos/sin encoded for continuity)
                        track, adverse = self.get_track_adverse_angles_norm(self.Aircrafts[i], self.Aircrafts[j])
                        i_obs.extend([
                            np.cos(track), np.sin(track),
                            np.cos(adverse), np.sin(adverse)
                        ])

                        # Target status flags
                        i_obs.append(self.Aircrafts[j].is_alive())  # 0 or 1
                        i_obs.append(self.Aircrafts[i].get_team() != self.Aircrafts[j].get_team())  # FoF: 0 = friend, 1 = foe

                # === Postprocessing: numerical stability and clipping ===
                i_obs = np.where(np.abs(i_obs) < 1e-9, 0.0, i_obs)  # Eliminate near-zero float noise
                observations[i_agent] = np.array(np.clip(i_obs, -1, 1))  # Clip to [-1, 1] for normalized input

        return observations


    def check_collision(self, i):
        """
        Check whether the i-th aircraft has collided with the ground or another aircraft.

        Collision conditions:
        - Z position (altitude) is above ground (positive Z = down → ground at Z = 0)
        - Proximity to another *alive* aircraft is below collision threshold

        :param i: Index of the aircraft to check
        :return: True if a collision occurred, else False
        """
        collided = False
        position = self.Aircrafts[i].get_pos()

        # === Ground collision ===
        # Z > 0 in this system means below ground (Z=0 is terrain surface)
        if position[2] > 0:
            collided = True

        # === Mid-air collision with another aircraft ===
        for j, aircraft in enumerate(self.Aircrafts):
            if j != i and aircraft.is_alive():
                distance = np.linalg.norm(self.relative_pos(i, j, 'aircraft'))
                if distance < self.collision_distance:
                    collided = True

        return collided

    def check_missile_tone(self, agent_index):
        # Get current missile tone state for attacker
        new_missile_tone_attack, new_missile_target = self.Aircrafts[agent_index].get_missile_tone_attack()
        new_missile_tone_defence = self.Aircrafts[agent_index].get_missile_tone_defence()

        # Get attacker's cone and state
        attack_cone = self.Aircrafts[agent_index].get_cone()
        attack_pos = self.Aircrafts[agent_index].get_pos()
        attack_vel = self.Aircrafts[agent_index].get_absolute_vel()
        team = self.Aircrafts[agent_index].get_team()

        possible_targets = []   # enemy aircraft that intersect our cone
        base_target = -1        # index of an enemy base that is within vulnerability and in cone
        max_defence_tone = 0    # max tone this agent receives from enemies

        # Only perform tone updates if not already locked defensively
        if new_missile_tone_defence < 0.5:
            for i, aircraft in enumerate(self.Aircrafts):
                if i != agent_index and aircraft.is_alive() and aircraft.get_team() != team:
                    # Check mutual intersection of cones
                    defence_cone = aircraft.get_cone()
                    defence_pos = aircraft.get_pos()
                    defence_vel = aircraft.get_absolute_vel()

                    # Attacker cone → defender (can I hit them?)
                    intersect = self.check_intersect_cones(attack_cone, attack_pos, attack_vel,
                                                        defence_cone, defence_pos, defence_vel)

                    # Defender cone → attacker (can they hit me?)
                    intersected = self.check_intersect_cones(defence_cone, defence_pos, defence_vel,
                                                            attack_cone, attack_pos, attack_vel)

                    if intersect:
                        possible_targets.append(self.possible_agents[i])

                    # Track max defensive tone from enemies
                    defence_tone, _ = aircraft.get_missile_tone_attack()
                    if intersected and defence_tone > max_defence_tone:
                        max_defence_tone = defence_tone

            # Check base targeting conditions
            for i, base in enumerate(self.bases):
                if i != team:
                    attack_angle, attack_min_dist, attack_max_dist = attack_cone
                    is_in_cone = self.is_within_cone(attack_pos, attack_vel, base,
                                                    attack_angle, attack_min_dist, attack_max_dist)

                    dist = np.linalg.norm(self.relative_pos(agent_index, i, 'base'))
                    is_in_vuln = dist < self.bases_vulnerability_distance

                    if is_in_cone and is_in_vuln:
                        base_target = i

        # === Decision logic: who to lock on ===

        if base_target != -1:
            # If we're aiming at a base
            if new_missile_target == "base":
                new_missile_tone_attack += self.stepwise_tone_increment
            elif new_missile_target == "none":
                new_missile_target = "base"
                new_missile_tone_attack = self.stepwise_tone_increment

        elif len(possible_targets) > 0:
            # If we're aiming at an aircraft
            if new_missile_target in possible_targets:
                new_missile_tone_attack = np.clip(
                    new_missile_tone_attack + self.stepwise_tone_increment, 0, 1
                )
            else:
                new_missile_target = np.random.choice(possible_targets)
                new_missile_tone_attack = self.stepwise_tone_increment

        else:
            # No valid targets
            new_missile_target = "none"
            new_missile_tone_attack = 0

        # Update defensive tone from strongest adversary
        new_missile_tone_defence = max_defence_tone

        return new_missile_tone_attack, new_missile_tone_defence, new_missile_target

    def is_within_cone(self, cone_origin, cone_direction, target_position, angle_deg, min_dist, max_dist):
        half_angle_deg = angle_deg / 2

        # Compute vector from the cone origin to the target
        vector_to_target = np.array(target_position) - np.array(cone_origin)
        distance = np.linalg.norm(vector_to_target)

        # If the target is outside the allowed distance bounds, it's outside the cone
        if distance < min_dist or distance > max_dist:
            return False

        # Normalize the vector from origin to target
        direction_to_target = vector_to_target / distance

        # Normalize the cone axis direction
        cone_direction = cone_direction / np.linalg.norm(cone_direction)

        # Compute the cosine of the angle between cone direction and target vector
        dot = np.dot(cone_direction, direction_to_target)

        # Check if this angle is within the cone's half-angle (converted to radians)
        return dot >= np.cos(np.deg2rad(half_angle_deg))

    def check_intersect_cones(self, attack_cone, attack_pos, attack_vel, defence_cone, defence_pos, defence_vel):
        """
        Determines whether two aircraft are in mutual missile engagement positions.

        Specifically:
        - Checks if the defender is inside the attacker's forward-facing attack cone.
        - Simultaneously checks if the attacker is inside the defender's rearward vulnerability cone.

        Args:
            attack_cone (tuple): (half_angle_deg, min_dist, max_dist) for the attacker's cone.
            attack_pos (array): Position of the attacking aircraft (3D).
            attack_vel (array): Velocity vector of the attacker (used as cone direction).
            defence_cone (tuple): (half_angle_deg, min_dist, max_dist) for the defender's rear cone.
            defence_pos (array): Position of the defending aircraft.
            defence_vel (array): Velocity of the defender (used for rearward cone).

        Returns:
            bool: True if both aircraft are in each other's targeting cone, False otherwise.
        """

        # Unpack cone parameters
        attack_angle, attack_min_dist, attack_max_dist = attack_cone
        defence_angle, defence_min_dist, defence_max_dist = defence_cone

        # Check if defender is within attacker's forward missile cone
        attacker_check = self.is_within_cone(
            attack_pos,         # origin of the cone
            attack_vel,         # direction of the cone
            defence_pos,        # target position
            attack_angle,
            attack_min_dist,
            attack_max_dist
        )

        # Check if attacker is within defender's rearward vulnerability cone
        defender_check = self.is_within_cone(
            defence_pos,
            -defence_vel,       # use negative velocity for rear-facing cone
            attack_pos,
            defence_angle,
            defence_min_dist,
            defence_max_dist
        )

        # Return True only if both conditions are met
        return attacker_check and defender_check


    def fire(self, agent_index, missile_target, missile_tone):
        """
        Attempt to fire a missile from the agent at the specified target, 
        if the missile tone (lock quality) exceeds a firing threshold.

        Supports both aircraft and base targets.

        :param agent_index: Index of the firing aircraft
        :param missile_target: Target agent ID (str) or 'base'
        :param missile_tone: Lock quality [0.0–1.0] — higher = stronger lock
        :return: ID of the killed target (str), or 'none' if no kill
        """
        kill = 'none'

        # === Fire only if lock is strong enough ===
        if missile_tone > 0.5:
            # === Aircraft target ===
            if missile_target != 'base':
                target_index = self.possible_agents.index(missile_target)

                hit = self.check_kill_probability(
                    attacker=agent_index,
                    defender=target_index,
                    tone=missile_tone,
                    type='aircraft'
                )

                kill = missile_target if hit else 'none'
                self.Aircrafts[agent_index].set_missile_tone_attack(0, 'none')

            # === Base target ===
            else:
                hit = self.check_kill_probability(
                    attacker=None,
                    defender=None,
                    tone=missile_tone,
                    type='base'
                )

                kill = 'base' if hit else 'none'
                self.Aircrafts[agent_index].set_missile_tone_attack(0, 'none')

        return kill

    def check_kill_probability(self, attacker, defender, tone, type):
        """
        Evaluate whether a missile hit is successful based on tone strength and target type.

        For aircraft:
            Probability is a function of:
            - tone quality (lock)
            - time to intercept (distance / missile speed)
            - missile-to-target acceleration ratio

        For base:
            - Probability is equal to tone (direct hit model)

        :param attacker: Index of the firing aircraft (None if attacking base)
        :param defender: Index of the target aircraft (None if attacking base)
        :param tone: Lock quality [0.0–1.0]
        :param type: 'aircraft' or 'base'
        :return: True if hit occurs, else False
        """
        is_hit = False
        bernoulli_threshold = 0

        if type == 'aircraft':
            # === Estimate intercept time ===
            distance = np.linalg.norm(self.relative_pos(defender, attacker, 'aircraft'))
            T_intercept = distance / self.missile_speed

            # === Use ratio of missile acceleration to target maneuverability ===
            max_am = self.max_acc_missile
            max_aa = self.Aircrafts[defender].get_max_acc()

            # Kill probability heuristic
            maneuver_ratio = np.clip(max_am / max_aa, 0.1, 2.0)
            bernoulli_threshold = 0.2 + 0.8 * (tone * (1 / (1 + T_intercept)) * maneuver_ratio)

        elif type == 'base':
            # Simpler model for static base: probability proportional to tone
            bernoulli_threshold = tone

        # === Sample a Bernoulli trial ===
        sample = np.random.uniform(0.0, 1.0)
        if sample < bernoulli_threshold:
            is_hit = True

        return is_hit


    def get_track_adverse_angles_norm(self, aircraft, target_aircraft):
        agent_pos = np.array(aircraft.get_agent_telemetry()['position'][-1])
        roll, pitch, yaw = aircraft.get_agent_telemetry()['orientation'][-1]
        agent_R = self.body_to_vehicle(roll, pitch, yaw)
        agent_forward = agent_R @ np.array([1, 0, 0])
        agent_forward_unit = agent_forward / np.linalg.norm(agent_forward)

        target_pos = np.array(target_aircraft.get_agent_telemetry()['position'][-1])
        roll, pitch, yaw = target_aircraft.get_agent_telemetry()['orientation'][-1]
        target_R = self.body_to_vehicle(roll, pitch, yaw)
        target_forward = target_R @ np.array([1, 0, 0])
        target_forward_unit = target_forward / np.linalg.norm(target_forward)

        # vector from agent to target (LOS)
        los_vec = target_pos - agent_pos
        los_vec_unit = los_vec / np.linalg.norm(los_vec)

        track_angle = np.arccos(np.clip(agent_forward_unit @ los_vec_unit, -1.0, 1.0))

        # vector from agent to target (LOS)
        los_vec = agent_pos - target_pos
        los_vec_unit = los_vec / np.linalg.norm(los_vec)
        adverse_angle = np.arccos(np.clip(target_forward_unit @ los_vec_unit, -1.0, 1.0))

        return track_angle / np.pi, adverse_angle/np.pi
    
    def get_closure_rate_norm(self, aircraft, target_aircraft):
        agent_pos = np.array(aircraft.get_agent_telemetry()['position'][-1])
        agent_vel = aircraft.get_absolute_vel()
        target_pos = np.array(target_aircraft.get_agent_telemetry()['position'][-1])
        target_vel = target_aircraft.get_absolute_vel()
        
        rel_vel = target_vel - agent_vel
        los_vec = target_pos - agent_pos
        los_unit = los_vec / (np.linalg.norm(los_vec) + 1e-6)  # prevent div by zero
        closure = -np.dot(rel_vel, los_unit)
        
        #normalization based on arctangent function 
        return np.atan(np.deg2rad(closure)) / np.atan(np.deg2rad(686))


    def get_individual_reward(self, agent_index, action, kill, missile_tone_attack, missile_tone_defence, missile_target):
        terminated = False
        truncated = False
        reward_Flight = {}
        reward_Pursuit = {}
        Total_Reward = {}
        aircraft = self.Aircrafts[agent_index]
        team = aircraft.get_team()
        telemetry = aircraft.get_agent_telemetry()
        vel = telemetry['velocity'][-1]
        prev_vel = telemetry['velocity'][-2]
        altitude = -telemetry['position'][-1][2]
        prev_altitude = -telemetry['position'][-2][2]
        acceleration_body = telemetry['acceleration'][-1]
        actions = telemetry['commands']

        Versions = {
            1: {
                'AL': 0.5,
                'CS': 0.5,

                'P': 0.3,
                'CR': 0.7,

                'GFW': 0.1,
                'PW': 0.9
            },
            2: {
                'AL': 0.5,
                'CS': 0.5,

                'P': 0.5,
                'CR': 0.5,

                'GFW': 0.1,
                'PW': 0.9
            },
            3: {
                'AL': 0.5,
                'CS': 0.5,

                'P': 0.7,
                'CR': 0.3,

                'GFW': 0.1,
                'PW': 0.9
            }
        }

        #### Flight Related Rewards ####
        a_A = 15
        mid_A = 0.25
        abs_alt = abs(self.env_size[2]/2 - altitude) / (self.env_size[2]/2)
        reward_Flight['Altitude'] = -((1/(1 + np.exp(-a_A * (abs_alt - mid_A)))) * 
                                      Versions[self.reward_version]['AL'])
        reward_Flight['Altitude'] += ((abs(self.env_size[2]/2 - altitude) < 1000) *
                                      (1000/np.clip(abs(self.env_size[2]/2 - altitude), 100, 1000)) 
                                      * Versions[self.reward_version]['AL'])
        
        
        a_S = 15
        mid_S = 0.2
        abs_speed = abs(280-vel[0]) / 200
        reward_Flight['Cruise Speed'] = -((1/(1 + np.exp(-a_S * (abs_speed - mid_S)))) * 
                                          Versions[self.reward_version]['CS'])
        reward_Flight['Cruise Speed'] = ((abs(280-vel[0]) < 70) *
                                      (70/np.clip(abs(self.env_size[2]/2 - altitude), 7, 70)) 
                                      * Versions[self.reward_version]['CS'])
        
        normalized_reward_Flight = sum(reward_Flight.values())


        #### Pursuit related Rewards ####
        # Choose Enemy Plane
        closest_enemy_plane = None
        c = 0
        dist = 1000000
        for i, enemy_aircraft in enumerate(self.Aircrafts):
            if enemy_aircraft.get_team() != team and enemy_aircraft.is_alive():
                rel_pos = np.linalg.norm(self.relative_pos(agent_index, i, 'aircraft'))
                if rel_pos < dist:
                    dist = rel_pos
                    closest_enemy_plane = enemy_aircraft


        if closest_enemy_plane is not None:
            track_angle, adverse_angle = self.get_track_adverse_angles_norm(aircraft, closest_enemy_plane)

            # Pursuit_angle
            shaped_pursuit = np.tan((adverse_angle-track_angle)*(np.pi/2.5)) / np.tan(np.pi/2.5)
            reward_Pursuit['Pursuit'] = shaped_pursuit * Versions[self.reward_version]['P']

            # Closure subject to minimum distance and adverse angle tuning
            closure_dist_norm = (1+self.get_closure_rate_norm(aircraft, closest_enemy_plane)) * (adverse_angle-track_angle)
            reward_Pursuit['Closure'] = closure_dist_norm * Versions[self.reward_version]['CR']


            if missile_target != 'base':
                Total_Reward['Attack'] = 5 * missile_tone_attack * track_angle
                self.attack_metric += 1
            Total_Reward['Defence'] = -7 * missile_tone_defence * adverse_angle

        else:
            #TODO: insert here some guidance to go towards the base and destroy it
            reward_Pursuit['Pursuit'] = 0
            reward_Pursuit['Closure'] = 0
        
        #Sparse Pursuit Rewards:
        
        if missile_target == 'base':
            Total_Reward['Attack'] = 0  #TODO: change in subsequent trainings to destroy the base

        if kill != 'none':
            Total_Reward['Kill'] = 2000
            self.kill_metric += 1

        normalized_reward_Pursuit = sum(reward_Pursuit.values())


        #### Reward Merge ####
        Total_Reward.update(reward_Flight)
        Total_Reward.update(reward_Pursuit)

        normalized_total_reward = (Versions[self.reward_version]['GFW'] * normalized_reward_Flight +
                                    Versions[self.reward_version]['PW'] * normalized_reward_Pursuit)

        #### Termination Condition Rewards ####
        acc=0
        if len(telemetry['acceleration']) > 5:
            acc = np.linalg.norm(np.mean(telemetry['acceleration'][-5: -1]))

        #check collision or over-g
        if (self.check_collision(agent_index) 
            or acc >= (20*9.81) 
            or vel[0]<80 
            or altitude>self.env_size[2]
            or aircraft.get_distance_from_centroid(self.bases) > self.max_size):
            self.Aircrafts[agent_index].kill()
            terminated = True
            normalized_total_reward = -1000


        self.episode_rewards[self.possible_agents[agent_index]].append(Total_Reward.copy())
        return normalized_total_reward, terminated, truncated

    def CLI_report(self, telemetry, action):
        """
        Print a human-readable report of the latest aircraft telemetry and control action.

        :param telemetry: Dictionary containing aircraft telemetry (position, velocity, etc.)
        :param action: Tuple or array-like of control inputs: (throttle, elevon, aileron, rudder, fire)
        """
        t, e, a, r, f = action  # Control surface commands

        # Extract latest orientation in radians
        roll, pitch, yaw = telemetry['orientation'][-1]

        # === Status Report ===
        print(f"Position:     {telemetry['position'][-1].round(2)}")
        print(f"Velocity:     {telemetry['velocity'][-1].round(2)}  Acceleration: {telemetry['acceleration'][-1].round(2)}")
        print(f"Orientation:  Roll={np.rad2deg(roll):.2f}°, Pitch={np.rad2deg(pitch):.2f}°, Yaw={np.rad2deg(yaw):.2f}°")
        print(f"AoA:          {telemetry['AoA'][-1]:.3f}   Sideslip: {telemetry['sideslip'][-1]:.3f}")
        print(f"Forces:       {telemetry['force'][-1].round(2)}")
        print(f"Moments:      {telemetry['moment'][-1].round(2)}")
        print(f"Controls:     Throttle={t:.2f}, Elevon={e:.1f}, Aileron={a:.1f}, Rudder={r:.1f}, Fire={f}")
        print()  # Blank line for spacing

    def discretizer(self, action):
        """
        Discretize continuous control inputs using environment-defined step sizes.

        Each control input is clipped to its allowable range and then snapped to the nearest
        discretized step. This is useful for environments or policies that operate on
        quantized actions instead of continuous values.

        :param action: Array or tuple of 5 control values:
                    (AoA, Sideslip, Roll, Speed, Fire)
                    All expected to be in the range [-1, 1] or [0, 1] (fire is passed through)
        :return: Numpy array of discretized control values
        """
        UpAngle, SideAngle, speed, fire = action

        def round_to_step(value, step, min_val, max_val):
            value = np.clip(value, min_val, max_val)
            steps = round((value - min_val) / step)
            return min_val + steps * step

        # Discretize each control input based on configured resolution
        speed_discretized     = round_to_step(speed,     self.speed_step,     0.0, 1.0)
        UpAngle_discretized       = round_to_step(UpAngle,       self.UpAngle_step,       -1.0, 1.0)
        SideAngle_discretized      = round_to_step(SideAngle,      self.SideAngle_step,      -1.0, 1.0)

        return np.array([
            UpAngle_discretized,
            SideAngle_discretized,
            speed_discretized,
            fire  # Fire command remains as-is (discrete binary or thresholded elsewhere)
        ])

    def step(self, action_dict):
        """
        Advance the environment one timestep using the provided actions.

        Handles:
        - Per-agent action stepping
        - Missile tone computation and firing
        - Reward assignment
        - Agent and episode termination
        - Truncation checks

        :param action_dict: Dict mapping agent_id -> action vector
        :return: Tuple of (observations, rewards, terminations, truncations, info)
        """
        self.episode_steps += 1
        rewards = {}
        terminated = {'__all__': True}
        truncated = {'__all__': False}

        for player, a in action_dict.items():
            agent_index = self.possible_agents.index(player)

            if self.Aircrafts[agent_index].is_alive():
                # === Process action ===
                action = self.discretizer(a) if self.discretize else a
                action[-1] = 1

                # Step physics model
                self.Aircrafts[agent_index].step(action, self.frequency_factor)

                # === Evaluate missile lock tone (pre-fire) ===
                missile_tone_attack, missile_tone_defence, missile_target = self.check_missile_tone(agent_index)

                # === Fire missile if fire command is active ===
                kill = 'none'
                if action[-1] > 0.5:
                    kill = self.fire(agent_index, missile_target, missile_tone_attack)
                    # Re-evaluate tones post-fire (may change due to aircraft killed)
                    missile_tone_attack, missile_tone_defence, missile_target = self.check_missile_tone(agent_index)

                # === Handle kill ===
                if kill != 'none':
                    if kill == 'base':
                        terminated['__all__'] = True
                    else:
                        victim_index = self.possible_agents.index(missile_target)
                        self.Aircrafts[victim_index].kill()

                # === Compute agent reward, termination, and truncation ===
                rewards[player], terminated[player], truncated[player] = (
                    self.get_individual_reward(
                        agent_index, action, kill,
                        missile_tone_attack, missile_tone_defence,
                        missile_target
                    )
                )

                # If any agent is still alive, keep episode running
                if not terminated[player]:
                    terminated['__all__'] = False

                # Update missile tone memory for RL agent
                self.Aircrafts[agent_index].set_missile_tone_attack(missile_tone_attack, missile_target)
                self.Aircrafts[agent_index].set_missile_tone_defence(missile_tone_defence)

            else:
                # === Dead agents return no reward and are marked terminated ===
                rewards[player] = 0.0
                terminated[player] = True
                truncated[player] = False

        # === Global episode termination due to step limit ===
        if self.episode_steps > self.max_steps:
            terminated['__all__'] = True

        
        # === Special case: end if dummy aircraft dies ===
        for team in range(self.num_teams):
            at_least_one_alive = 0
            for a in range(self.num_agents_team):
                if self.Aircrafts[team*self.num_agents_team + a].is_alive():
                    at_least_one_alive += 1
            if at_least_one_alive == 0:
                for k in terminated.keys():
                    terminated[k] = True
                    terminated['__all__'] = True
        

        # === Optional: Add team-based rewards here ===
        # (e.g., base destruction, shared victories, assists, etc.)
        return self.get_obs(), rewards, terminated, truncated, {'__common__': {'attack_steps' : self.attack_metric, 'kills': self.kill_metric}}

    
    def body_to_vehicle(self, roll, pitch, yaw):
        """
        Compute rotation matrix from body frame to world (vehicle/inertial) frame.

        This is the transpose (i.e., inverse) of the vehicle-to-body rotation matrix
        since rotation matrices are orthonormal (R⁻¹ = Rᵀ).

        :param roll: Roll angle [rad]
        :param pitch: Pitch angle [rad]
        :param yaw: Yaw angle [rad]
        :return: 3x3 rotation matrix (body → world)
        """
        return self.vehicle_to_body(roll, pitch, yaw).T

    def vehicle_to_body(self, roll, pitch, yaw):
        """
        Compute rotation matrix from world (vehicle/inertial) frame to body frame.

        Rotation sequence follows aerospace convention: ZYX (yaw → pitch → roll)
        - Rotate about Z (yaw), then Y (pitch), then X (roll)

        :param roll: Roll angle [rad]
        :param pitch: Pitch angle [rad]
        :param yaw: Yaw angle [rad]
        :return: 3x3 rotation matrix (world → body)
        """

        # Trig shorthands
        cr = np.cos(roll)
        sr = np.sin(roll)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cy = np.cos(yaw)
        sy = np.sin(yaw)

        # Construct ZYX rotation matrix
        R = np.array([
            [cp*cy,                 cp*sy,                 -sp],
            [sr*sp*cy - cr*sy,      sr*sp*sy + cr*cy,      sr*cp],
            [cr*sp*cy + sr*sy,      cr*sp*sy - sr*cy,      cr*cp]
            ])
        return R

    def render(self, screen_size=(800, 800), mode='rgb_array', altitude_range=(0, 100)):
        """
        Render the current environment state using Pygame.
        Supports 'rgb_array' output for video logging or 'human' for interactive display.

        :param screen_size: Tuple[int, int] - Width and height of the screen in pixels
        :param mode: 'rgb_array' returns a NumPy array; 'human' displays a window
        :param altitude_range: Tuple[float, float] - Range of altitudes mapped to alpha transparency
        :return: RGB image as NumPy array if mode == 'rgb_array', else None
        """
        # Background color (simulated sea or sky)
        SEA_BLUE = (0, 105, 148)
        self._screen.fill(SEA_BLUE)

        # Define team colors (looped per team)
        TEAM_COLORS = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]

        def altitude_to_alpha(z, z_min, z_max):
            """Convert altitude (z, positive down) to alpha (transparency)."""
            z = np.clip(z, z_min, z_max)
            norm = 1 - (z - z_min) / (z_max - z_min)
            return int(norm * 255)

        def rotate_point(x, y, angle_rad):
            """
            Rotates a point (x, y) counterclockwise by angle_rad around the origin.
            """
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)

        if not self.bases:
            # No bases to render — return empty screen
            return np.zeros((screen_size[1], screen_size[0], 3), dtype=np.uint8)

        # === Compute screen scaling ===
        all_positions = [b[:2] for b in self.bases] + [
            ac.get_physics_telemetry()["position"][-1][:2]
            for ac in self.Aircrafts
            if ac.get_physics_telemetry()["position"]
        ]

        all_positions = np.array(all_positions)
        min_x, min_y = np.min(all_positions, axis=0)
        max_x, max_y = np.max(all_positions, axis=0)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        padding = 100
        scale_x = (screen_size[0] - padding) / max((max_x - min_x), 1)
        scale_y = (screen_size[1] - padding) / max((max_y - min_y), 1)
        scale = min(scale_x, scale_y)

        # === Draw bases ===
        for i, base in enumerate(self.bases):
            x, y, _ = base
            screen_x = int((x - center_x) * scale + screen_size[0] / 2)
            screen_y = int(screen_size[1] / 2 - (y - center_y) * scale)
            color = TEAM_COLORS[i % len(TEAM_COLORS)]

            # Draw vulnerability radius
            vuln_radius_px = int(self.bases_vulnerability_distance * scale)
            s_vuln = pygame.Surface((vuln_radius_px * 2, vuln_radius_px * 2), pygame.SRCALPHA)
            pygame.draw.circle(s_vuln, (*color, 40), (vuln_radius_px, vuln_radius_px), vuln_radius_px, width=1)
            self._screen.blit(s_vuln, (screen_x - vuln_radius_px, screen_y - vuln_radius_px))

            # Draw base marker
            base_radius = max(3, int(10 * scale))
            s_base = pygame.Surface((base_radius * 2, base_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s_base, (*color, 180), (base_radius, base_radius), base_radius)
            self._screen.blit(s_base, (screen_x - base_radius, screen_y - base_radius))

        # === Draw aircrafts ===
        for idx, aircraft in enumerate(self.Aircrafts):
            telemetry = aircraft.get_physics_telemetry()

            if not telemetry["position"]:
                continue

            pos = telemetry["position"][-1]
            ori = telemetry["orientation"][-1]
            x, y, z = pos
            yaw = ori[2]

            screen_x = int((x - center_x) * scale + screen_size[0] / 2)
            screen_y = int(screen_size[1] / 2 - (y - center_y) * scale)

            if aircraft.is_alive():
                team_id = idx // self.num_agents_team
                color = TEAM_COLORS[team_id % len(TEAM_COLORS)]
                alpha = altitude_to_alpha(z, *altitude_range)

                # Draw aircraft shape (triangle)
                shape = [(7, 0), (-4, 3), (-4, -3)]
                rotated = [rotate_point(px, py, -yaw) for px, py in shape]
                points = [(screen_x + px, screen_y + py) for px, py in rotated]
                pygame.draw.polygon(self._screen, (*color, alpha), points)

                # Draw attack and defense cones
                cone_params = aircraft.get_cone()
                if cone_params is not None:
                    cone_angle_deg, cone_min_dist, cone_max_dist = cone_params
                    num_segments = 15
                    min_len_px = cone_min_dist * scale
                    max_len_px = cone_max_dist * scale

                    for label, angle_offset, a in [
                        ("attack", 0, 60),
                        ("defence", np.pi, 50)
                    ]:
                        cone_angle = yaw + angle_offset
                        cone_pts = []

                        # Outer arc
                        for i in range(num_segments + 1):
                            ang = cone_angle - np.radians(cone_angle_deg) / 2 + i * np.radians(cone_angle_deg) / num_segments
                            dx = max_len_px * np.cos(ang)
                            dy = -max_len_px * np.sin(ang)
                            cone_pts.append((screen_x + dx, screen_y + dy))

                        # Inner arc (reversed)
                        for i in reversed(range(num_segments + 1)):
                            ang = cone_angle - np.radians(cone_angle_deg) / 2 + i * np.radians(cone_angle_deg) / num_segments
                            dx = min_len_px * np.cos(ang)
                            dy = -min_len_px * np.sin(ang)
                            cone_pts.append((screen_x + dx, screen_y + dy))

                        cone_surface = pygame.Surface(screen_size, pygame.SRCALPHA)
                        pygame.draw.polygon(cone_surface, (*color, a), cone_pts)
                        self._screen.blit(cone_surface, (0, 0))

                # Label: ID, altitude, velocity, missile tone
                missile_tone, missile_target = aircraft.get_missile_tone_attack()
                target_str = str(missile_target) if missile_target is not None else "None"
                label_text = (f"{idx} - alt:{z:.2f} - vel:{telemetry['velocity'][-1][0]:.2f} - "
                            f"{aircraft.is_alive()}: {missile_tone:.2f} → {target_str}")
                label = self.font.render(label_text, True, color)
                self._screen.blit(label, (screen_x + 10, screen_y - 10))

            else:
                # Draw dead aircraft as gray dot
                radius = 8
                dead_color = (0, 0, 0, 100)
                s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, dead_color, (radius, radius), radius)
                self._screen.blit(s, (screen_x - radius, screen_y - radius))

                label_text = f"{idx} - alt:{z:.2f}"
                label = self.font.render(label_text, True, dead_color)
                self._screen.blit(label, (screen_x + 10, screen_y - 10))

        # === Return or display ===
        if mode == 'rgb_array':
            # Transpose for correct image layout (H, W, C)
            rgb_array = pygame.surfarray.array3d(self._screen)
            return np.transpose(rgb_array, (1, 0, 2))
        elif mode == 'human':
            if not pygame.get_init():
                pygame.init()
            window = pygame.display.set_mode(screen_size)
            window.blit(self._screen, (0, 0))
            pygame.display.flip()

    def render_trajectory(self, save_folder, every_n=50, scale=10.0):
        """
        Visualize the trajectories, velocities, and orientations of all agents in a single 3D plot.

        Args:
        - save_folder: directory to save the plot image
        - every_n: interval to plot velocity/orientation arrows
        - scale: length scale for arrows
        """

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        cmap = get_cmap("tab10")
        max_range = 0
        all_positions = []

        for idx, agent in enumerate(self.possible_agents):
            telemetry = self.Aircrafts[idx].get_physics_telemetry()
            positions = np.array(telemetry['position'])
            velocities = np.array(telemetry['velocity'])
            orientations = np.array(telemetry['orientation'])

            color = cmap(idx % 10)
            label_prefix = f"{agent}"

            # Plot trajectory
            ax.plot(positions[:, 0], positions[:, 1], -positions[:, 2],
                    color=color, linewidth=2, label=f'{label_prefix} Trajectory')

            all_positions.append(positions)

            # Plot velocity and orientation arrows
            for i in range(0, len(positions), every_n):
                pos = [positions[i][0], positions[i][1], -positions[i][2]]
                roll, pitch, yaw = orientations[i]
                R = self.body_to_vehicle(roll, pitch, yaw)
                vel = R @ velocities[i]
                vel[2] = -vel[2]

                ax.quiver(*pos, *vel, length=scale, color=color, normalize=True)

                # Orientation body frame axes
                body_axes = {
                    'Nose': (R @ np.array([1, 0, 0]), 'red'),
                    'Wing': (R @ np.array([0, -1, 0]), 'green'),
                    'Tail': (R @ np.array([0, 0, 1]), 'magenta')
                }

                for label, (vec, axis_color) in body_axes.items():
                    vec[2] = -vec[2]
                    ax.quiver(*pos, *vec, length=scale * 0.5, color=axis_color, normalize=True)

            # Keep track of global bounds
            max_range = max(max_range, np.ptp(positions, axis=0).max())

        # Combine all positions to compute global center
        all_positions = np.concatenate(all_positions, axis=0)
        center = all_positions.mean(axis=0)

        ax.set_xlim(center[0] - max_range / 2, center[0] + max_range / 2)
        ax.set_ylim(center[1] - max_range / 2, center[1] + max_range / 2)
        ax.set_zlim(-center[2] - max_range / 2, -center[2] + max_range / 2)

        ax.set_xlabel('X (East)')
        ax.set_ylabel('Y (North)')
        ax.set_zlabel('Altitude (Up)')
        ax.set_title('3D Trajectories with Velocity and Orientation (All Agents)')
        ax.legend(loc='upper right')

        plt.tight_layout()

        os.makedirs(save_folder, exist_ok=True)

        path = os.path.join(save_folder, "all_agents_trajectory.png")
        plt.savefig(path)

        # Convert matplotlib fig to plotly and save interactive HTML
        plotly_fig = self.mpl3d_to_plotly(fig)  # Pass the current fig
        html_path = os.path.join(save_folder, "all_agents_trajectory.html")
        plotly_fig.write_html(html_path)

        plt.close()

    def mpl3d_to_plotly(self, fig):
        """
        Convert a 3D Matplotlib figure into a Plotly 3D figure.

        This function extracts 3D line data (e.g., aircraft trajectories)
        and reproduces them using Plotly for interactive visualization.

        :param fig: Matplotlib figure object containing 3D axes
        :return: Plotly figure (go.Figure) with 3D line traces
        """
        def mpl_rgba_to_plotly(rgba):
            """
            Convert an RGBA tuple from Matplotlib to Plotly-compatible RGBA string.
            """
            r, g, b, a = rgba
            return f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {a})"

        plotly_fig = go.Figure()

        # Extract 3D axes only
        axes_3d = [ax for ax in fig.get_axes() if hasattr(ax, 'get_proj')]

        for ax in axes_3d:
            # Process each 3D line (e.g., flight path)
            for line in ax.lines:
                x, y, z = line.get_data_3d()
                label = line.get_label()
                plotly_fig.add_trace(go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode='lines',
                    name=label if label != '_nolegend_' else None,
                    line=dict(
                        color=mpl_rgba_to_plotly(line.get_color()),
                        width=line.get_linewidth()
                    )
                ))

            # Copy axis labels to Plotly layout
            plotly_fig.update_layout(
                scene=dict(
                    xaxis_title=ax.get_xlabel(),
                    yaxis_title=ax.get_ylabel(),
                    zaxis_title=ax.get_zlabel()
                )
            )

        return plotly_fig

    def plot_telemetry(self, save_folder):
        """
        Plot and save telemetry data for each agent.

        This function creates a CSV and a multi-subplot figure per agent, including:
        - Position, velocity, acceleration
        - Orientation (Euler angles)
        - Forces and moments
        - Control inputs
        - Wind angles (AoA and sideslip)

        Args:
            save_folder (str): Directory where plots and CSVs will be saved.
        """
        for idx, agent in enumerate(self.possible_agents):
            telemetry = self.Aircrafts[idx].get_agent_telemetry()

            # === Extract telemetry as numpy arrays ===
            pos = np.array(telemetry['position'])
            vel = np.array(telemetry['velocity'])
            accel = np.array(telemetry['acceleration'])
            eulers = np.rad2deg(np.array(telemetry['orientation']))  # Convert to degrees
            force = np.array(telemetry['force'])
            moment = np.array(telemetry['moment'])
            cmds = np.array(telemetry['commands'])
            AoA = np.rad2deg(np.array(telemetry['AoA']))
            sideslip = np.rad2deg(np.array(telemetry['sideslip']))

            # === Save to CSV ===
            df = pd.DataFrame({
                'pos_x': pos[:, 0], 'pos_y': pos[:, 1], 'pos_z': pos[:, 2],
                'vel_x': vel[:, 0], 'vel_y': vel[:, 1], 'vel_z': vel[:, 2],
                'acc_x': accel[:, 0], 'acc_y': accel[:, 1], 'acc_z': accel[:, 2],
                'roll_deg': eulers[:, 0], 'pitch_deg': eulers[:, 1], 'yaw_deg': eulers[:, 2],
                'force_x': force[:, 0], 'force_y': force[:, 1], 'force_z': force[:, 2],
                'moment_x': moment[:, 0], 'moment_y': moment[:, 1], 'moment_z': moment[:, 2],
                'UpAngle': cmds[:, 0], 'SideAngle': cmds[:, 1], 'Speed': cmds[:, 2], 'Fire': cmds[:, 3],
                'AoA': AoA, 'sideslip': sideslip,
            })

            csv_path = os.path.join(save_folder, f"{agent}_telemetry.csv")
            df.to_csv(csv_path, index=False)

            # === Plot telemetry ===
            fig, axs = plt.subplots(8, 1, figsize=(18, 18), sharex=True)

            # --- Position ---
            axs[0].plot(pos[:, 0], label='X')
            axs[0].plot(pos[:, 1], label='Y')
            axs[0].plot(pos[:, 2], label='Z')
            axs[0].set_title("Position (m)")
            axs[0].legend()

            # --- Velocity ---
            axs[1].plot(vel[:, 0], label='Vx')
            axs[1].plot(vel[:, 1], label='Vy')
            axs[1].plot(vel[:, 2], label='Vz')
            axs[1].set_title("Velocity (m/s)")
            axs[1].legend()

            # --- Acceleration ---
            axs[2].plot(accel[:, 0], label='Ax')
            axs[2].plot(accel[:, 1], label='Ay')
            axs[2].plot(accel[:, 2], label='Az')
            axs[2].set_title("Acceleration (m/s²)")
            axs[2].legend()

            # --- Orientation (Euler Angles) ---
            axs[3].plot(eulers[:, 0], label='Roll (°)')
            axs[3].plot(eulers[:, 1], label='Pitch (°)')
            axs[3].plot(eulers[:, 2], label='Yaw (°)')
            axs[3].set_title("Euler Angles (degrees)")
            axs[3].legend()

            # --- Force ---
            axs[4].plot(force[:, 0], label='Fx')
            axs[4].plot(force[:, 1], label='Fy')
            axs[4].plot(force[:, 2], label='Fz')
            axs[4].set_title("Forces (N)")
            axs[4].legend()

            # --- Moment ---
            axs[5].plot(moment[:, 0], label='Mx')
            axs[5].plot(moment[:, 1], label='My')
            axs[5].plot(moment[:, 2], label='Mz')
            axs[5].set_title("Moments (Nm)")
            axs[5].legend()

            # --- Control Inputs ---
            axs[6].plot(cmds[:, 0], label='UpAngle')
            axs[6].plot(cmds[:, 1], label='SideAngle')
            axs[6].plot(cmds[:, 2], label='Speed')
            axs[6].plot(cmds[:, 3], label='Fire')
            axs[6].set_title("Control Inputs")
            axs[6].legend()

            # --- Wind Angles ---
            axs[7].plot(AoA, label='AoA (°)')
            axs[7].plot(sideslip, label='Sideslip (°)')
            axs[7].set_title("Wind Angles")
            axs[7].legend()

            # Final layout adjustments
            plt.tight_layout()

            # === Save figure ===
            fig_path = os.path.join(save_folder, f"{agent}_telemetry_plot.png")
            plt.savefig(fig_path)
            plt.close()

    def plot_rewards(self, save_folder):
        """
        Generate and save line plots showing per-agent reward components over time.

        Each agent gets its own subplot, with multiple lines for different reward components.
        Rewards are stored as a list of dictionaries in self.episode_rewards[agent_name].

        Args:
            save_folder (str): Directory where the output plot will be saved.
        """
        os.makedirs(save_folder, exist_ok=True)  # Ensure output folder exists

        num_agents = len(self.episode_rewards)
        fig, axs = plt.subplots(num_agents, 1, figsize=(12, 4 * num_agents), constrained_layout=True)

        if num_agents == 1:
            axs = [axs]  # Ensure axs is iterable even for single agent

        for idx, (agent_name, rewards_list) in enumerate(self.episode_rewards.items()):
            if not rewards_list:
                continue  # Skip if no rewards collected for this agent

            # Get all reward keys used over the episode
            all_keys = set().union(*(r.keys() for r in rewards_list))

            timesteps = list(range(len(rewards_list)))

            # Plot each component as its own line
            for key in sorted(all_keys):  # Sorted for consistent legend order
                values = [r.get(key, 0.0) for r in rewards_list]
                axs[idx].plot(timesteps, values, label=key)

            axs[idx].set_title(f"Agent: {agent_name}")
            axs[idx].set_xlabel("Timestep")
            axs[idx].set_ylabel("Reward Value")
            axs[idx].legend()
            axs[idx].grid(True)

        # Set common title and save
        plt.suptitle("Per-Agent Reward Components Over Time", fontsize=16)
        save_path = os.path.join(save_folder, "agent_rewards.png")
        plt.savefig(save_path)
        plt.close()


def Test_env():
    """
    Test function to simulate a multi-agent AerialCombat environment run
    with fixed actions, rendering and saving results.

    Workflow:
    - Load YAML config
    - Initialize environment
    - Run simulation using predefined actions
    - Render and save video, telemetry, and reward data
    """

    # Load test configuration from YAML file
    with open("Train_Run_config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Create the AerialCombat environment
    env = AerialBattle(config["env_config"], config['uav_config'])

    # Initialize list to collect rendered frames
    images = []

    # Reset the environment and get initial observations
    for i in range(100):
        env.reset()

    observations = env.reset()
    images.append(env.render())  # Render the initial state

    # Define fixed actions per agent for evaluation
    # Format: [Up_Angle, Side_Angle, Speed, Fire], all normalized in body frame
    predefined_actions = [
        [0.000, 0, 1, 0],
        [0.000, 0, 1, 0],
        [0.000, 0, 1, 0],
        [0.000, 0, 1, 0],
        [0.000, 0, 1, 0]
    ]

    a = 0  # Action index pointer

    # Run simulation for 20 steps per predefined action set
    for step in range(len(predefined_actions) * 150):
        # Update the action index every 50 steps
        if step % 150 == 0 and step != 0:
            a += 1

        # Build action dictionary for alive agents
        actions = {}
        for i, agent_id in enumerate(env.get_agent_ids()):
            if env.Aircrafts[i].is_alive():
                actions[agent_id] = predefined_actions[a]

        # Step the environment
        obs, rewards, terminated, truncated, infos = env.step(actions)

        # Render and store the current frame
        images.append(env.render())

        # Console output
        print("Dones:", terminated)

        # Early exit if all agents are terminated
        if all(terminated.values()):
            print("All agents are done.")
            break

    # Save the episode animation as a video
    imageio.mimsave("ENV_TEST/test_1.mp4", [np.array(img) for img in images], fps=10)

    # Export trajectory plots, telemetry, and rewards
    env.render_trajectory("ENV_TEST")
    env.plot_telemetry("ENV_TEST")
    env.plot_rewards("ENV_TEST")

    # Clean up environment (if needed)
    env.close()

#Test_env()

