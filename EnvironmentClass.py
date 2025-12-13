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
    """
    High-level aircraft entity wrapper around the low-level physics model.

    Responsibilities:
      - Owns a FixedWingAircraft physics model (6DOF integration).
      - Implements control logic (PID loops) that convert high-level action targets
        into actuator commands (throttle/elevator/aileron/rudder).
      - Supports scripted "dummy" behavior for target aircraft (line/curve/random).
      - Tracks per-agent telemetry for training/logging (separate from physics telemetry).
      - Stores game/engagement state (alive/team, missile tone, target selection).

    This class is typically the unit that the environment interacts with.
    """

    def __init__(self, UAV_config, rho, g, frequency, team_id):
        """
        Create an aircraft with a physical model and control/engagement state.

        Parameters
        ----------
        UAV_config : dict
            Aircraft configuration:
              - physics parameters (mass, aero coeffs, etc.)
              - PID gains + rate limits
              - engagement cones
        rho : float
            Air density [kg/m^3].
        g : float
            Gravity [m/s^2].
        frequency : float
            Physics update rate [Hz] used to define dt for PID integration.
        team_id : int
            Team identifier for multi-team environments.
        """

        # --- Low-level flight dynamics model (6DOF) ---
        self.physical_model = FixedWingAircraft(UAV_config, rho, g, frequency)

        # --- Missile/engagement state ---
        # These values are typically used by the environment to compute rewards, UI, or firing logic.
        self.missile_tone_attack = 0     # "lock quality" / readiness to fire (agent perspective)
        self.missile_tone_defence = 0    # "incoming threat" warning level (defensive cue)
        self.missile_target = "none"     # target ID string or special values: "base" / "none"

        # --- Basic state flags ---
        self.live = 1                    # 1 = alive, 0 = destroyed
        self.team = team_id              # team affiliation

        # --- PID control configuration ---
        # gains: dict with kp/ki/kd per channel (AoA, sideslip, roll, speed).
        # rate_limits: per-actuator slew rate limits (units depend on your normalization).
        self.gains = UAV_config['gains']
        self.rate_limits = UAV_config['rate_limits']

        # Last actuator commands (used by slew-rate limiter).
        self.last_cmd = {
            'elevator': 0.0,
            'aileron': 0.0,
            'rudder': 0.0,
            'throttle': 0.0,
        }

        # --- Per-agent telemetry (sampled once per "control step") ---
        # This is distinct from physical_model telemetry (which is per physics step).
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

        # --- PID internal state ---
        # prev_errors: last error value for derivative term
        # integrals: accumulated error for integral term
        self.prev_errors = {'AoA': 0, 'sideslip': 0, 'roll': 0, 'speed': 0}
        self.integrals = {'AoA': 0, 'sideslip': 0, 'roll': 0, 'speed': 0}
        self.dt = 1.0 / frequency  # time step used by PID update

        # --- Engagement parameters ---
        # Each cone often encodes: (cone_angle_deg, min_dist, max_dist)
        self.attack_cone = UAV_config['attack_cone']
        self.defence_cone = UAV_config['defence_cone']

        # --- Dummy/scripted behavior mode ---
        # dummy_type:
        #   - "none": controlled aircraft (PID + physics)
        #   - "line"/"curve"/"fixed": uses physical_model.dummy_step()
        #   - "random": periodically switches between line and curve with random parameters
        self.dummy_type = "none"
        self.turn_radius = 0
        self.direction = 0
        self.speed = 0

        # Random dummy behavior configuration
        self.change_speed = False
        self.random_change_period = 250
        self.random_change_period_options = []
        self.dummy_turn_options = []
        self.dummy_dir_options = []
        self.dummy_speed_options = []

        # Random dummy bookkeeping
        self.random_dummy_counter = 0
        self.random_dummy_type = "none"

    def set_dummy(self,
                  type,
                  change_speed=False,
                  random_change_period_options=[250],
                  dummy_turn_options=[5000],
                  dummy_dir_options=[1, -1],
                  dummy_speed_options=[0]):
        """
        Configure scripted "dummy" behavior used to create non-learning targets.

        Parameters
        ----------
        type : str
            Dummy mode name:
              - "line", "curve", "fixed": directly executed by dummy_step()
              - "random": internally selects line/curve and periodically changes parameters
        change_speed : bool
            If True (in "random" mode), speed may randomly change using dummy_speed_options.
        random_change_period_options : list[int]
            Candidate durations (in control steps) between random changes.
        dummy_turn_options : list[float]
            Candidate turn radii [m] used for "curve".
        dummy_dir_options : list[int]
            Candidate directions (+1 or -1) for turning.
        dummy_speed_options : list[float]
            Candidate speed deltas applied when change_speed is enabled.
        """

        # Store configuration for later random sampling
        self.dummy_type = type
        self.change_speed = change_speed
        self.random_change_period_options = random_change_period_options
        self.dummy_turn_options = dummy_turn_options
        self.dummy_dir_options = dummy_dir_options
        self.dummy_speed_options = dummy_speed_options

        # Initialize random dummy parameters
        self.turn_radius = np.random.choice(self.dummy_turn_options)
        self.direction = np.random.choice(self.dummy_dir_options)
        self.random_change_period = np.random.choice(self.random_change_period_options)

        # Initial random motion type (line or curve)
        self.random_dummy_type = np.random.choice(['line', 'curve'])
        self.random_dummy_counter = 0

    def is_dummy(self):
        """
        Returns True if this aircraft is operating in scripted dummy mode.
        """
        return self.dummy_type != 'none'

    def reset(self, position, orientation, speed, alive, UAV_config):
        """
        Reset the aircraft state and internal telemetry.

        This resets both:
          - The underlying physics model state (position/orientation/velocity)
          - Control state (PID integrators/errors)
          - Engagement state (missile tones/targets)
          - Per-agent telemetry buffer

        Parameters
        ----------
        position : array-like (3,)
            Initial world position [m].
        orientation : array-like (3,)
            Initial Euler angles [rad] (roll, pitch, yaw).
        speed : float
            Initial forward speed [m/s] along body X axis.
        alive : int/bool
            Alive flag (1/True = alive, 0/False = dead).
        UAV_config : dict
            Updated configuration; allows swapping gains/cones at reset time.
        """

        # Update engagement and control parameters (supports per-episode reconfiguration)
        self.attack_cone = UAV_config['attack_cone']
        self.defence_cone = UAV_config['defence_cone']
        self.gains = UAV_config['gains']
        self.rate_limits = UAV_config['rate_limits']

        # Reset low-level flight dynamics
        self.physical_model.reset(position, orientation, speed, UAV_config)

        # Store nominal speed for dummy mode use
        self.speed = speed

        # Reset status flags and missile tones
        self.live = alive
        self.missile_tone_attack = 0
        self.missile_tone_defence = 0
        self.missile_target = "none"

        # Reset PID controller memory
        self.prev_errors = {'AoA': 0, 'sideslip': 0, 'roll': 0, 'speed': 0}
        self.integrals = {'AoA': 0, 'sideslip': 0, 'roll': 0, 'speed': 0}

        # Reset per-agent telemetry buffer
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

        # Record initial state snapshot based on the physics model's latest telemetry
        for key in self.agent_telemetry.keys():
            self.agent_telemetry[key].append(self.physical_model.getTelemetry()[key][-1])

    def step(self, action, frequency_factor):
        """
        Advance aircraft dynamics for one *control step*.

        This wrapper supports a multi-rate setup:
          - The environment/agent provides one action per control step
          - The physics may integrate multiple smaller steps per control step

        Parameters
        ----------
        action : array-like
            High-level action vector.
            In your usage:
              - action[:-1] is interpreted by Action_Translation_Layer as
                [up_angle_norm, side_angle_norm, speed_norm]
              - action[-1] may represent an extra command (e.g. fire) and is passed through for logging
        frequency_factor : int
            Number of physics integration substeps per control step.
        """

        # Perform N internal physics updates for one external action.
        for i in range(frequency_factor):
            # Current AoA is used for damping / smooth transitions in the translation layer.
            current_AoA_rad = self.physical_model.getTelemetry()['AoA'][-1]

            # Convert high-level action into normalized setpoints:
            # AoA, sideslip, roll, throttle/speed.
            AoA, sideslip, roll, throttle = self.Action_Translation_Layer(
                action[:-1],
                (i / frequency_factor),
                current_AoA_rad
            )

            # Convert setpoints into actuator commands via PID loops.
            t, e, a, r = self.PID_Control([AoA, sideslip, roll, throttle])

            # Choose physical vs dummy behavior.
            if self.dummy_type == "none":
                # Normal aircraft: physics step with commanded actuators.
                self.physical_model.step(t, e, a, r, action)

            elif self.dummy_type == "random":
                # Random dummy: scripted motion, periodically re-samples its path parameters.
                self.physical_model.dummy_step(self.random_dummy_type, self.turn_radius, self.direction, self.speed)
                self.random_dummy_counter += 1

                # After the change period, resample path parameters (and optionally speed).
                if self.random_dummy_counter > self.random_change_period * frequency_factor:
                    self.random_dummy_counter = 0
                    self.random_change_period = np.random.choice(self.random_change_period_options)
                    self.random_dummy_type = np.random.choice(['line', 'curve'])
                    self.turn_radius = np.random.choice(self.dummy_turn_options)
                    self.direction = np.random.choice(self.dummy_dir_options)

                    if self.change_speed:
                        # Randomly adjust speed but keep it within operational bounds.
                        self.speed = np.clip(self.speed + np.random.choice(self.dummy_speed_options), 100, 343)

            else:
                # Fixed dummy: executes the specified kinematic profile (line/curve/fixed).
                self.physical_model.dummy_step(self.dummy_type, self.turn_radius, self.direction, self.speed)

        # Record one telemetry sample per control step (post-integration).
        for key in self.agent_telemetry.keys():
            self.agent_telemetry[key].append(self.physical_model.getTelemetry()[key][-1])

    def Action_Translation_Layer(self, action, manouvre_progress, current_AoA_rad):
        """
        Translate a compact, high-level action into flight setpoints for the PID controller.

        Conceptually:
          - The agent provides a desired "direction" and "speed" (normalized).
          - This layer converts that direction into:
              AoA setpoint (pitch/vertical intent),
              roll setpoint (bank to steer),
              sideslip setpoint (optionally),
              speed setpoint.

        Parameters
        ----------
        action : array-like length 3
            [up_angle_norm, side_angle_norm, speed_norm]
              up_angle_norm   : desired vertical angular offset (normalized [-1,1])
              side_angle_norm : desired lateral angular offset (normalized [-1,1])
              speed_norm      : desired speed command (normalized)
        manouvre_progress : float
            0..1 progress indicator used to phase in commands smoothly during a maneuver.
        current_AoA_rad : float
            Current AoA [rad], used for gradual AoA changes in early maneuver phase.

        Returns
        -------
        list length 4
            [AoA_norm, sideslip_norm, roll_norm, speed_norm]
        """

        # 1) Convert normalized angular commands into radians
        # max_angle_rad bounds how aggressively the agent can request a direction change.
        max_angle_rad = np.deg2rad(40)
        v_up = action[0] * max_angle_rad
        v_side = action[1] * max_angle_rad

        # Speed is passed through normalized; it is scaled later by PID_Control.
        v_speed = action[2]

        # 2) Convert direction request into a unit vector in body coordinates.
        # Body axis convention: X forward, Y right, Z down.
        # This defines the direction the agent wants the velocity vector to point.
        vx = np.cos(v_up) * np.cos(v_side)
        vy = np.cos(v_up) * np.sin(v_side)
        vz = np.sin(v_up)

        # 3) Two control modes:
        #    - For certain angle regimes, interpret inputs as direct AoA/sideslip.
        #    - Otherwise, compute roll that aligns lift/steering with the requested direction.
        if v_up < 0 and abs(v_side) < np.deg2rad(5):
            # Direct-angle mode: treat the agent’s angles as direct setpoints.
            AoA_rad = v_up
            sideslip_rad = v_side
            roll_rad = 0.0
        else:
            # Vector-tracking mode:
            # AoA is derived from forward component; roll steers in the Y-Z plane.
            AoA_rad = np.arccos(vx)
            sideslip_rad = 0.0

            # Roll command is derived from the desired lateral/vertical direction.
            roll_rad = np.arctan2(vy, vz)

            # During the early part of a maneuver, soften AoA changes to reduce controller transients.
            if manouvre_progress < 0.6 and AoA_rad > np.deg2rad(10):
                AoA_rad = np.clip(current_AoA_rad * 1.025, -AoA_rad, AoA_rad)

        # 4) Normalize setpoints back into standardized ranges.
        # These normalized outputs match PID_Control’s expected scaling rules.
        AoA_norm = AoA_rad / np.deg2rad(40)
        sideslip_norm = sideslip_rad / np.deg2rad(40)
        roll_norm = roll_rad / np.pi

        return [AoA_norm, sideslip_norm, roll_norm, v_speed]

    def rate_limit(self, name, cmd, dt):
        """
        Slew-rate limiter for actuator commands.

        Prevents unrealistically fast command changes by limiting the maximum
        change per time step.

        Parameters
        ----------
        name : str
            One of: 'elevator', 'aileron', 'rudder', 'throttle'
        cmd : float
            Desired command (already clipped to valid range).
        dt : float
            Time step [s].

        Returns
        -------
        float
            Rate-limited command, also stored as last_cmd[name] for next step.
        """

        rl = self.rate_limits[name]  # max command change per second (in normalized units)
        if rl <= 0 or dt <= 0:
            self.last_cmd[name] = cmd
            return cmd

        prev = self.last_cmd[name]
        max_step = rl * dt
        delta = cmd - prev

        # Clamp the change to +/- max_step.
        if delta > max_step:
            cmd = prev + max_step
        elif delta < -max_step:
            cmd = prev - max_step

        self.last_cmd[name] = cmd
        return cmd

    def PID_Control(self, action):
        """
        PID controller mapping target flight setpoints -> actuator commands.

        Input setpoints are normalized. Internally they are scaled into physical
        units to compute errors versus telemetry values.

        Parameters
        ----------
        action : array-like length 4
            [target_AoA_norm, target_sideslip_norm, target_roll_norm, target_speed_norm]

        Returns
        -------
        (throttle, elevator, aileron, rudder) : tuple[float, float, float, float]
            throttle in [-1, 1] (as currently clipped here),
            elevator/aileron/rudder in [-1, 1].
        """

        # Unpack setpoints (normalized)
        target_AoA, target_sideslip, target_roll, target_speed = action

        # Keep target speed in a reasonable normalized range (prevents overly low setpoints).
        target_speed = np.clip(target_speed, 0.4, 1)

        # Scale normalized setpoints to physical units used for error computation.
        target_AoA *= 50
        target_sideslip *= 50
        target_roll *= 180
        target_speed *= 343

        # Read current state from physics telemetry (latest sample).
        telemetry = self.physical_model.getTelemetry()
        current_AoA = np.rad2deg(telemetry['AoA'][-1])
        current_sideslip = np.rad2deg(telemetry['sideslip'][-1])
        current_roll = np.rad2deg(telemetry['orientation'][-1][0])
        current_speed = np.linalg.norm(telemetry['velocity'][-1])

        # Compute control errors for each loop.
        # Roll error uses wrap-around arithmetic to avoid discontinuities near +/-180 deg.
        errors = {
            'AoA': target_AoA - current_AoA,
            'sideslip': target_sideslip - current_sideslip,
            'roll': ((target_roll - current_roll) + 180) % 360 - 180,
            'speed': target_speed - current_speed
        }

        outputs = {}

        # Compute PID output for each controlled variable.
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

        # Convert PID outputs into normalized actuator commands.
        # The scaling factors (e.g., /40, /343) represent "how many units correspond to full deflection."
        commands = {
            'elevator': -np.clip(outputs['AoA'] / 40, -1, 1),
            'rudder': -np.clip(outputs['sideslip'] / 40, -1, 1),
            'aileron': np.clip(outputs['roll'] / 40, -1, 1),
            'throttle': np.clip(outputs['speed'] / 343, -1, 1)
        }

        # Apply slew-rate limits to each actuator.
        commands['elevator'] = self.rate_limit('elevator', commands['elevator'], self.dt)
        commands['aileron']  = self.rate_limit('aileron',  commands['aileron'],  self.dt)
        commands['rudder']   = self.rate_limit('rudder',   commands['rudder'],   self.dt)
        commands['throttle'] = self.rate_limit('throttle', commands['throttle'], self.dt)

        # Final clip to ensure numerical safety and Python float types.
        commands['elevator'] = float(np.clip(commands['elevator'], -1.0, 1.0))
        commands['aileron']  = float(np.clip(commands['aileron'],  -1.0, 1.0))
        commands['rudder']   = float(np.clip(commands['rudder'],   -1.0, 1.0))
        commands['throttle'] = float(np.clip(commands['throttle'], -1, 1))

        return commands['throttle'], commands['elevator'], commands['aileron'], commands['rudder']

    def kill(self):
        """Mark the aircraft as destroyed (no longer alive)."""
        self.live = 0

    def is_alive(self):
        """Return alive flag (1/0)."""
        return self.live

    def get_team(self):
        """Return the team identifier for this aircraft."""
        return self.team

    def set_missile_tone_attack(self, tone, target):
        """
        Set offensive missile tone and the currently designated target.

        Parameters
        ----------
        tone : float/int
            Attack tone level (interpretation depends on your environment).
        target : str
            Target identifier string (agent ID) or special values like "base"/"none".
        """
        self.missile_tone_attack = tone
        self.missile_target = target

    def get_missile_tone_attack(self):
        """Return (attack_tone, target_id)."""
        return self.missile_tone_attack, self.missile_target

    def set_missile_tone_defence(self, tone):
        """
        Set defensive missile tone (threat warning).

        Parameters
        ----------
        tone : float/int
            Threat tone level (interpretation depends on your environment).
        """
        self.missile_tone_defence = tone

    def get_missile_tone_defence(self):
        """Return defensive missile tone value."""
        return self.missile_tone_defence

    def get_cones(self):
        """
        Return offensive and defensive cone parameters.

        Returns
        -------
        (attack_cone, defence_cone)
            Each cone is typically [cone_angle_deg, min_dist, max_dist].
        """
        return self.attack_cone, self.defence_cone

    def get_pos(self):
        """Return current aircraft position in world coordinates [m]."""
        return self.physical_model.get_pos()

    def get_distance_from_centroid(self, bases):
        """
        Compute planar (XY) distance from the aircraft to the centroid of base positions.

        Parameters
        ----------
        bases : array-like (N,3)
            World positions for all bases.

        Returns
        -------
        float
            Euclidean distance in the XY plane [m].
        """
        centroid = np.mean(bases, axis=0)[:2]
        return np.linalg.norm(centroid - self.get_pos()[:2])

    def get_absolute_vel(self):
        """
        Return inertial/world-frame velocity vector [m/s].
        """
        return self.physical_model.get_absolute_velocity()

    def get_physics_telemetry(self):
        """
        Return raw telemetry directly from the physics model.

        Notes
        -----
        Physics telemetry is typically recorded every physics integration step,
        which may be higher frequency than agent_telemetry.
        """
        return self.physical_model.getTelemetry()

    def get_agent_telemetry(self):
        """
        Return per-control-step telemetry stored by this wrapper.

        This is typically the stream used for training logs and observation generation.
        """
        return self.agent_telemetry

    def get_own_data(self, max_speed, max_size, bases):
        """
        Build this aircraft's observation vector (normalized features).

        Features included:
          - altitude (z)
          - acceleration (3)
          - body velocity (3)
          - orientation as cos/sin (6)
          - angular velocity (3)
          - AoA as cos/sin (2)
          - sideslip as cos/sin (2)
          - distance to base centroid (1)
          - missile tones (2)

        Parameters
        ----------
        max_speed : float
            Used to normalize body velocity components.
        max_size : float
            Used to normalize spatial distances (e.g., map size).
        bases : array-like
            List/array of base positions.

        Returns
        -------
        list[float]
            Flattened normalized observation vector.
        """

        telemetry = self.get_agent_telemetry()
        current_state = []

        # Altitude: with Z-down convention, larger z means lower altitude.
        current_state.append(telemetry['position'][-1][2] / 10000)

        # Acceleration normalized by ~20 g (rough safety/scale factor for learning).
        current_state.extend(np.array(telemetry["acceleration"][-1]) / (20 * 9.8))

        # Body velocity normalized by scenario max speed.
        current_state.extend(np.array(telemetry["velocity"][-1]) / max_speed)

        # Orientation encoded as cos/sin to avoid wrap-around discontinuities.
        current_state.extend(np.cos(telemetry["orientation"][-1]))
        current_state.extend(np.sin(telemetry["orientation"][-1]))

        # Body rates normalized by a constant scaling factor.
        current_state.extend(np.array(telemetry["angular_velocity"][-1]) / 10)

        # AoA encoded as cos/sin.
        current_state.append(np.cos(telemetry["AoA"][-1]))
        current_state.append(np.sin(telemetry["AoA"][-1]))

        # Sideslip encoded as cos/sin.
        current_state.append(np.cos(telemetry["sideslip"][-1]))
        current_state.append(np.sin(telemetry["sideslip"][-1]))

        # Distance to centroid of bases (XY only), normalized by max_size.
        current_state.append(self.get_distance_from_centroid(bases) / max_size)

        # Missile tone features (attack + defence).
        current_state.append(self.missile_tone_attack)
        current_state.append(self.missile_tone_defence)

        return current_state


class AerialBattle(MultiAgentEnv):
    """
    Multi-agent aerial combat environment (team-vs-team).

    Core responsibilities:
      - Create and manage a set of Aircraft objects (each wraps a physics model + PID control).
      - Handle episode reset: spawn bases, spawn aircraft, select alive agents, configure dummy targets.
      - Produce per-agent observations suitable for RL:
          * own state features (normalized)
          * relative kinematics to other aircraft (in ego/body frame)
          * additional engagement features (closure, track/adverse angles, alive/foe flags)

    This class follows RLlib MultiAgentEnv patterns:
      - self.possible_agents: all agent IDs that could ever appear
      - self.agents: active agent IDs for current episode (in this implementation: same as possible_agents)
      - reset() returns dict[agent_id] -> obs vector, plus info dict
    """

    def __init__(self, env_config, UAV_config, reward_version=1, discretize=True):
        """
        Initialize the environment.

        Parameters
        ----------
        env_config : dict
            Environment-level configuration:
              - simulation rates (physics_frequency, action_frequency)
              - world bounds and spawn settings
              - combat/missile parameters and reward version definitions
              - dummy-agent settings (optional scripted opponents)
        UAV_config : dict
            Aircraft configuration dictionary, indexed by model name.
            Each model entry is passed to Aircraft(...) and its FixedWingAircraft physics model.
        reward_version : int
            Selects which reward configuration to use from env_config['reward_versions'].
        discretize : bool
            Flag reserved for discretized controls (action mapping).
            Action space remains continuous in this snippet; flag may be used later.
        """
        super().__init__()

        # =========================================================
        # Pygame initialization (visualization/support tools)
        # =========================================================
        # Done only once per process to avoid repeated pygame init.
        if not hasattr(self, "_pygame_initialized"):
            pygame.init()
            self._screen = pygame.Surface(env_config['screen_size'])  # off-screen surface for rendering
            self._clock = pygame.time.Clock()
            self._agent_trails = {}                                  # optional: stores historic positions for drawing trails
            self._pygame_initialized = True

            pygame.font.init()
            self.font = pygame.font.SysFont('Arial', 14)

        # =========================================================
        # Core simulation parameters
        # =========================================================
        self.physics_frequency = env_config["physics_frequency"]      # physics integration rate [Hz]
        self.action_frequency = env_config["action_frequency"]        # RL/control rate [Hz]
        self.max_steps = env_config["max_episode_length"]             # episode horizon in control steps

        # ratio between physics and action frequencies (multi-rate integration)
        self.frequency_factor = self.physics_frequency // self.action_frequency

        self.g = env_config['g']

        # reward configuration selection
        self.reward_version = reward_version
        self.Reward_Config = env_config['reward_versions'][reward_version]

        # =========================================================
        # Missile / combat settings
        # =========================================================
        self.stepwise_tone_increment = env_config['stepwise_tone_increment']
        self.tone_threshold = env_config['tone_threshold']
        self.autotrigger = env_config['autotrigger']
        self.trigger_threshold = env_config['trigger_threshold']
        self.collision_distance = env_config["collision_distance"]

        # =========================================================
        # World and agent configuration
        # =========================================================
        self.env_size = env_config["env_size"]                 # world dimensions [x, y, z] (z is treated with your sign convention)
        self.max_size = env_config["max_size"]                 # normalization constant for distances
        self.min_bases_distance = env_config["min_bases_distance"]

        self.num_teams = env_config['team_number']
        self.num_agents_team = env_config["agent_number_team"]
        self.alive_agents_start = env_config['alive_agents_start']    # number of alive agents per team at episode start

        self.discretize = discretize

        # =========================================================
        # Discretization / resolution parameters (if used later)
        # =========================================================
        self.speed_step = env_config['speed_step']
        self.UpAngle_step = env_config['UpAngle_step']
        self.SideAngle_step = env_config['SideAngle_step']

        # =========================================================
        # Dummy agent parameters (scripted behavior)
        # =========================================================
        self.dummy = env_config["dummy"]                         # 'none', 'line', 'curve', 'random', 'mixed', etc.
        self.change_speed = env_config['change_speed']
        self.random_change_period_options = env_config['random_change_period_options']
        self.dummy_turn_options = env_config['dummy_turn_options']
        self.dummy_dir_options = env_config['dummy_dir_options']
        self.dummy_speed_options = env_config['dummy_speed_options']
        self.dummy_kill = env_config['dummy_kill']               # whether dummy aircraft can be killed / removed

        # =========================================================
        # Episode tracking / metrics
        # =========================================================
        self.agent_report = env_config["agent_report_name"]
        self.episode_rewards = {}   # per-agent reward history over episode
        self.episode_returns = {}   # per-agent cumulative return
        self.episode_steps = 0
        self.attack_metric = 0      # custom metric (e.g., number of attack “tone steps” accumulated)
        self.kill_metric = 0        # custom metric (e.g., number of kills)

        # =========================================================
        # Agent / aircraft initialization
        # =========================================================
        self.possible_agents = []          # all agent IDs
        self.Aircrafts = []                # list[Aircraft], aligned with possible_agents indices
        self.observation_spaces = {}       # dict[agent_id] -> gym space
        self.action_spaces = {}            # dict[agent_id] -> gym space

        # Bases are placed per-team; represented as list[np.array([x,y,z])]
        self.bases = []

        # Spawn configuration
        self.spawning_distance = env_config['spawning_distance']
        self.spawning_orientations = env_config['spawning_orientations']
        self.spawning_speeds = env_config['spawning_speeds']
        self.spawning_random = env_config['spawning_random']
        self.initial_multi_offset = env_config['initial_multi_offset']

        # Aircraft model selection per team (e.g., team 0 uses model A, team 1 uses model B)
        self.plane_model = env_config['plane_model']
        self.UAV_config = UAV_config

        # Create all agents and their Aircraft objects, and define spaces
        for i in range(self.num_teams):
            for j in range(self.num_agents_team):
                agent_name = f"agent_{i}_{j}"
                self.possible_agents.append(agent_name)

                # Instantiate the Aircraft wrapper (physics + PID + telemetry + missile state)
                self.Aircrafts.append(
                    Aircraft(
                        self.UAV_config[self.plane_model[i]],  # model config for this team
                        env_config['rho'],
                        env_config['g'],
                        env_config['physics_frequency'],
                        i,  # team ID
                    )
                )

                # Observation space:
                #   - 23 own-state features (see Aircraft.get_own_data())
                #   - For each other agent: 17 features (relative pos/vel + closure + angles + flags)
                obs_dim = 23 + 17 * ((self.num_teams * self.num_agents_team) - 1)
                self.observation_spaces[agent_name] = gym.spaces.Box(
                    low=-1.5, high=1.5, shape=(obs_dim,), dtype=np.float64
                )

                # Action space:
                # 4D continuous action:
                #   [up_angle_norm, side_angle_norm, speed_norm, fire_cmd]
                self.action_spaces[agent_name] = gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(4,), dtype=np.float64
                )

        # Active agents for current episode (here all possible agents are always present,
        # but per-agent alive flags control whether observations are produced).
        self.agents = self.possible_agents.copy()

    def set_plane_model(self, team_id, model_name):
        """
        Change the aircraft model used by a given team.

        Parameters
        ----------
        team_id : int
            Team index.
        model_name : str
            Key into UAV_config for that aircraft model.
        """
        self.plane_model[team_id] = model_name
        print(f"Set plane model for team {team_id} to {model_name}")

    def get_observation_space(self, team_id):
        """
        Return observation space for an agent ID.

        Parameters
        ----------
        team_id : str
            Agent string ID (e.g., "agent_0_1").

        Returns
        -------
        gym.spaces.Box
            Observation space for that agent.
        """
        return self.observation_spaces[team_id]

    def get_action_space(self, team_id):
        """
        Return action space for an agent ID.

        Parameters
        ----------
        team_id : str
            Agent string ID (e.g., "agent_1_2").

        Returns
        -------
        gym.spaces.Box
            Action space for that agent.
        """
        return self.action_spaces[team_id]

    def point_on_circumference(self, x0, y0, r, theta):
        """
        Utility: sample a point on a circle.

        Parameters
        ----------
        x0, y0 : float
            Circle center.
        r : float
            Radius.
        theta : float
            Angle [rad].

        Returns
        -------
        (x, y) : tuple[float, float]
            Point on circumference.
        """
        x = x0 + r * np.cos(theta)
        y = y0 + r * np.sin(theta)
        return (x, y)

    def init_airplane(self, aircraft, alive, testing, team, offset):
        """
        Initialize (or respawn) an aircraft with randomized position/orientation/speed.

        Spawning logic:
          - Alive aircraft spawn around the centroid of bases on a ring,
            with team-specific angular offsets.
          - Dead/inactive aircraft spawn elsewhere (often high/away) so they don't interfere.

        Parameters
        ----------
        aircraft : Aircraft
            Aircraft instance to reset.
        alive : bool/int
            Whether this aircraft should start alive.
        testing : bool
            If True, use more deterministic / evaluation-friendly spawn behavior.
        team : int
            Team index (used to select angular sector/model).
        offset : float
            Additional angle offset [rad] used to vary spawn location.
        """

        # "Centroid" used as a reference center for spawning around the arena.
        centroid = np.mean(self.bases, axis=0)
        centroid[2] = self.env_size[2] / 2

        # Team sectors are distributed evenly around a circle.
        delta = (2 * np.pi) / self.num_teams

        if alive:
            # Alive aircraft: spawn on a ring around the arena.
            max_spawn_distance = self.spawning_distance

            # Choose (x,y) on a ring, then clip to keep within bounds.
            x, y = self.point_on_circumference(
                centroid[0], centroid[1],
                max_spawn_distance,
                offset + (team * delta)
            )
            x = np.clip(x, 100, self.env_size[0] - 100)
            y = np.clip(y, 100, self.env_size[0] - 100)

            # Altitude placement: using your Z convention, this sets aircraft inside the combat volume.
            z = -self.env_size[2] / 2

            rand_pos = [x, y, z]

        else:
            # Inactive/dead aircraft: place randomly such that it is out of normal combat flow.
            rand_pos = [
                np.random.uniform(100, self.env_size[0] - 100),
                np.random.uniform(100, self.env_size[1] - 100),
                np.random.uniform(-(self.env_size[2] - 1000), -1000)
            ]

        # Orient aircraft generally toward the arena centroid.
        rand_orient = self.orientation_to_target(rand_pos, centroid)

        # Apply controlled randomness to roll/pitch/yaw.
        discrete_pitch, discrete_yaw = self.spawning_orientations

        # Small random roll for variety.
        rand_orient[0] = np.deg2rad(np.random.uniform(-10, 10))

        # Pitch is chosen from predefined options (discretized set).
        rand_orient[1] = np.deg2rad(np.random.choice(discrete_pitch))

        # Add yaw offset to avoid identical headings among multiple spawns.
        rand_orient[2] += np.deg2rad(np.random.choice(discrete_yaw))

        # Random initial speed from configured set.
        rand_speed = np.random.choice(self.spawning_speeds)

        # Apply the spawn state to the aircraft.
        aircraft.reset(
            rand_pos,
            rand_orient,
            rand_speed,
            alive,
            self.UAV_config[self.plane_model[team]]
        )

    def reset(self, seed=42, testing=False, options=None):
        """
        Reset the environment and start a new episode.

        Reset pipeline:
          1) Place team bases (with spacing constraints, here on a circle).
          2) Reset per-agent episode reward/return trackers.
          3) Randomly select which agents start alive.
          4) Spawn all aircraft based on alive masks and team sectors.
          5) Optionally convert one aircraft to a scripted dummy actor.
          6) Return initial observations and shared info.

        Parameters
        ----------
        seed : int
            Optional seed parameter for Gym compatibility (not wired into numpy here).
        testing : bool
            If True, use evaluation-style spawning (less stochastic).
        options : dict | None
            Gym API compatibility placeholder.

        Returns
        -------
        (obs, info) : tuple
            obs : dict[agent_id] -> np.ndarray observation
            info : dict with '__common__' metrics
        """
        self.episode_steps = 0
        self.attack_metric = 0
        self.kill_metric = 0

        # Optional random offset applied to spawn angles (adds variety across episodes).
        offset = (self.spawning_random == True) * np.deg2rad(
            np.random.choice([0, 0, 45, -45, 90, -90, 180, -180])
        )

        # Additional offset for multiple alive agents per team (spreads them slightly).
        multi_offset = 0
        if self.alive_agents_start > 1:
            multi_offset = np.deg2rad(self.initial_multi_offset / (self.alive_agents_start - 1))

        # =========================================================
        # Place team bases
        # =========================================================
        # Bases are arranged evenly around the arena to satisfy min distance.
        self.bases.clear()
        for team in range(self.num_teams):
            delta = (2 * np.pi) / self.num_teams
            x, y = self.point_on_circumference(
                self.env_size[0] / 2,
                self.env_size[1] / 2,
                self.min_bases_distance,
                team * delta
            )
            z = 0
            self.bases.append(np.array([x, y, z]))

        # =========================================================
        # Reset per-agent tracking
        # =========================================================
        for agent_id in self.possible_agents:
            self.episode_rewards[agent_id] = []
            self.episode_returns[agent_id] = 0.0

        # =========================================================
        # Choose which agents start alive
        # =========================================================
        # alive_masks[t][a] == 1 indicates aircraft a in team t is active.
        alive_masks = np.zeros((self.num_teams, self.num_agents_team))
        for _ in range(self.alive_agents_start):
            for t in range(self.num_teams):
                r_agent = np.random.randint(0, self.num_agents_team)
                while alive_masks[t][r_agent] == 1:
                    r_agent = np.random.randint(0, self.num_agents_team)
                alive_masks[t][r_agent] = 1

        # =========================================================
        # Spawn all aircraft according to masks
        # =========================================================
        for t in range(self.num_teams):
            for a in range(self.num_agents_team):
                index = t * self.num_agents_team + a
                aircraft = self.Aircrafts[index]

                # Spread multiple alive agents in a team by a small angular offset.
                final_offset = offset + (multi_offset * a)
                self.init_airplane(
                    aircraft,
                    alive_masks[t][a],
                    testing=testing,
                    team=t,
                    offset=final_offset
                )

        # =========================================================
        # Optional dummy aircraft injection
        # =========================================================
        if self.dummy != "none":
            # Choose a non-default team to host a dummy (keeps team 0 for the primary learning agents).
            t = np.random.randint(1, self.num_teams)

            # Pick one of the alive agents in that team.
            alive_indexes = alive_masks[t].nonzero()
            a = alive_indexes[0][np.random.randint(0, len(alive_indexes[0]))]

            index = t * self.num_agents_team + a
            aircraft = self.Aircrafts[index]
            final_offset = offset + (multi_offset * a)

            # Ensure dummy aircraft is alive and properly spawned.
            self.init_airplane(aircraft, alive=True, testing=testing, team=t, offset=final_offset)

            # Configure dummy mode type (optionally randomizing between line and curve).
            dummy_type = self.dummy
            if dummy_type == 'mixed':
                dummy_type = np.random.choice(['line', 'curve'])

            aircraft.set_dummy(
                dummy_type,
                self.change_speed,
                self.random_change_period_options,
                self.dummy_turn_options,
                self.dummy_dir_options,
                self.dummy_speed_options
            )

        # Return initial observations + shared metrics
        return self.get_obs(), {'__common__': {'attack_steps': self.attack_metric, 'kills': self.kill_metric}}

    def get_agent_ids(self):
        """
        Return all agent IDs (including those that may be dead/inactive).

        Returns
        -------
        list[str]
            Agent names (e.g., ["agent_0_0", "agent_1_2", ...]).
        """
        return self.possible_agents

    def orientation_to_target(self, position, target_position):
        """
        Compute the Euler orientation required to face from position -> target_position.

        The result is a simple "point at target" orientation:
          - roll is set to 0 (not needed for pointing)
          - yaw is computed from XY bearing
          - pitch is computed from vertical component and your Z convention

        Parameters
        ----------
        position : array-like (3,)
            Source position [x, y, z].
        target_position : array-like (3,)
            Target position [x, y, z].

        Returns
        -------
        np.ndarray (3,)
            [roll, pitch, yaw] in radians.
        """

        direction = target_position - position
        norm_dir = direction / np.linalg.norm(direction)
        dx, dy, dz = norm_dir

        # Heading in XY plane
        yaw = np.arctan2(dy, dx)

        # Pitch uses dz with your Z-down convention.
        pitch = np.arcsin(-dz)

        roll = 0.0
        return np.array([roll, pitch, yaw])

    def relative_pos(self, i, j, type):
        """
        Relative position vector from aircraft i to a target.

        Parameters
        ----------
        i : int
            Index of reference aircraft.
        j : int
            Index of target aircraft/base (depending on type).
        type : str
            'aircraft' -> target is another aircraft
            'base'     -> target is a base

        Returns
        -------
        np.ndarray (3,)
            Vector from i to j in world coordinates.
        """
        if type == "aircraft":
            rel_pos = np.array(self.Aircrafts[j].get_pos()) - np.array(self.Aircrafts[i].get_pos())
        elif type == "base":
            rel_pos = np.array(self.bases[j]) - np.array(self.Aircrafts[i].get_pos())
        else:
            rel_pos = np.zeros(3)

        # Remove very small numerical noise
        rel_pos = np.where(np.abs(rel_pos) < 1e-6, 0.0, rel_pos)
        return rel_pos

    def relative_polar_pos(self, i, j, type):
        """
        Convert relative position to a polar-like encoding.

        Output format:
          [r_norm, cos(theta), sin(theta), cos(phi), sin(phi)]

        where:
          - r_norm is distance normalized by max_size
          - theta is an elevation-like angle derived from z/r
          - phi is an azimuth angle in the XY plane

        This sin/cos representation avoids discontinuities and is ML-friendly.

        Parameters
        ----------
        i, j : int
            Indices of reference and target.
        type : str
            'aircraft' or 'base'

        Returns
        -------
        list[float]
            Encoded polar relative position.
        """
        rel_pos = self.relative_pos(i, j, type)
        x, y, z = rel_pos

        r = np.linalg.norm(rel_pos)

        if r < 1e-6:
            theta = 0.0
            phi = 0.0
        else:
            theta = np.arccos(z / r)
            phi = np.arctan2(y, x)

        return [r / self.max_size, np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)]

    def relative_vel(self, i, j, type):
        """
        Relative velocity vector from aircraft i to a target.

        Parameters
        ----------
        i : int
            Index of reference aircraft.
        j : int
            Index of target aircraft/base.
        type : str
            'aircraft' -> target velocity minus own velocity
            'base'     -> base assumed stationary

        Returns
        -------
        np.ndarray (3,)
            Relative velocity in world coordinates.
        """
        if type == "aircraft":
            rel_vel = np.array(self.Aircrafts[j].get_absolute_vel()) - np.array(self.Aircrafts[i].get_absolute_vel())
        elif type == "base":
            rel_vel = np.zeros(3) - np.array(self.Aircrafts[i].get_absolute_vel())
        else:
            rel_vel = np.zeros(3)

        rel_vel = np.where(np.abs(rel_vel) < 1e-6, 0.0, rel_vel)
        return rel_vel

    def relative_polar_vel(self, i, j, type):
        """
        Convert relative velocity to a polar-like encoding.

        Output format:
          [speed_norm, cos(theta), sin(theta), cos(phi), sin(phi)]

        where:
          - speed_norm is relative speed normalized by 343 m/s
          - theta and phi encode the velocity direction

        Parameters
        ----------
        i, j : int
            Indices of reference and target.
        type : str
            'aircraft' or 'base'

        Returns
        -------
        list[float]
            Encoded polar relative velocity.
        """
        rel_vel = self.relative_vel(i, j, type)
        x, y, z = rel_vel

        r = np.linalg.norm(rel_vel)

        if r == 0:
            theta = 0.0
            phi = 0.0
        else:
            theta = np.arccos(z / r)
            phi = np.arctan2(y, x)

        return [r / 343, np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)]

    def relative_polar_pos_Body(self, i, j, type):
        """
        Relative polar position of target j w.r.t. aircraft i, expressed in aircraft i body frame.

        This is typically more informative for control/engagement tasks because the
        agent’s action space is also defined in body coordinates.

        Output format:
          [r_norm, cos(theta), sin(theta), cos(phi), sin(phi)]

        Parameters
        ----------
        i : int
            Reference aircraft index (defines the body frame).
        j : int
            Target index (aircraft/base).
        type : str
            'aircraft' or 'base'

        Returns
        -------
        list[float]
            Polar encoding in the ego/body frame of i.
        """
        # Relative position in world coordinates
        rel_pos = self.relative_pos(i, j, type)

        # Ego orientation (Euler angles)
        orientation = self.Aircrafts[i].get_physics_telemetry()['orientation'][-1]
        yaw, pitch, roll = orientation[2], orientation[1], orientation[0]

        # Rotate world vector into body coordinates of aircraft i
        R = self.vehicle_to_body(roll, pitch, yaw)
        rel_body = R @ rel_pos
        x, y, z = rel_body

        # Convert to polar-like representation
        r = np.linalg.norm(rel_body)
        if r < 1e-6:
            theta = 0.0
            phi = 0.0
        else:
            theta = np.arccos(z / r)
            phi = np.arctan2(y, x)

        return [r / self.max_size, np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)]

    def relative_polar_vel_Body(self, i, j, type):
        """
        Relative polar velocity of target j w.r.t. aircraft i, expressed in aircraft i body frame.

        Output format:
          [speed_norm, cos(theta), sin(theta), cos(phi), sin(phi)]

        Parameters
        ----------
        i : int
            Reference aircraft index (defines the body frame).
        j : int
            Target index (aircraft/base).
        type : str
            'aircraft' or 'base'

        Returns
        -------
        list[float]
            Polar encoding in ego/body frame of i.
        """
        # Relative velocity in world coordinates
        rel_vel = self.relative_vel(i, j, type)

        # Ego orientation (Euler angles)
        orientation = self.Aircrafts[i].get_physics_telemetry()['orientation'][-1]
        roll, pitch, yaw = orientation

        # Rotate into body coordinates
        R = self.vehicle_to_body(roll, pitch, yaw)
        rel_body_vel = R @ rel_vel
        x, y, z = rel_body_vel

        speed = np.linalg.norm(rel_body_vel)
        if speed < 1e-6:
            theta = 0.0
            phi = 0.0
        else:
            theta = np.arccos(z / speed)
            phi = np.arctan2(y, x)

        return [speed / 343, np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)]

    def get_obs(self):
        """
        Build observations for all currently alive agents.

        Observation structure per agent i:
          1) Own-state vector (from Aircraft.get_own_data()).
          2) For each other agent j != i:
               - relative position in ego/body frame (polar encoding)
               - relative velocity in ego/body frame (polar encoding)
               - closure rate (normalized)
               - track/adverse angles (cos/sin encoded)
               - alive flag for j
               - foe-or-friend flag (FoF)

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from agent_id -> clipped observation vector in [-1, 1].
        """
        observations = {}

        for i, i_agent in enumerate(self.possible_agents):
            # Only produce observations for alive agents.
            if self.Aircrafts[i].is_alive():
                # -------------------------
                # Own-state features
                # -------------------------
                i_obs = self.Aircrafts[i].get_own_data(
                    max_speed=343,
                    max_size=self.max_size,
                    bases=self.bases
                )

                # -------------------------
                # Other-aircraft features
                # -------------------------
                for j, j_agent in enumerate(self.possible_agents):
                    if j_agent != i_agent:
                        # Relative position and velocity in ego/body frame
                        i_obs.extend(self.relative_polar_pos_Body(i, j, 'aircraft'))
                        i_obs.extend(self.relative_polar_vel_Body(i, j, 'aircraft'))

                        # Closure rate: relative speed projected on line-of-sight
                        i_obs.append(self.get_closure_rate_norm(self.Aircrafts[i], self.Aircrafts[j]))

                        # Track/adverse angles (encoded with sin/cos to avoid wrap-around)
                        track, adverse = self.get_track_adverse_angles_norm(self.Aircrafts[i], self.Aircrafts[j])
                        i_obs.extend([
                            np.cos(track), np.sin(track),
                            np.cos(adverse), np.sin(adverse)
                        ])

                        # Target flags: alive and friend-or-foe
                        i_obs.append(self.Aircrafts[j].is_alive())
                        i_obs.append(self.Aircrafts[i].get_team() != self.Aircrafts[j].get_team())

                # Final cleanup: remove tiny noise and clip for stable learning.
                i_obs = np.where(np.abs(i_obs) < 1e-9, 0.0, i_obs)
                observations[i_agent] = np.array(np.clip(i_obs, -1, 1))

        return observations
    
    def check_collision(self, i):
        """
        Check if the i-th aircraft has collided with the ground or another aircraft.

        Collision modes
        ---------------
        1) Ground collision:
           With a Z-down convention, "ground" is typically Z = 0 and Z > 0 implies
           the aircraft has passed below the ground plane.
        2) Mid-air collision:
           If the distance to any other *alive* aircraft falls below a configured threshold.

        Parameters
        ----------
        i : int
            Index of the aircraft to test.

        Returns
        -------
        bool
            True if any collision condition is detected, otherwise False.
        """
        collided = False
        position = self.Aircrafts[i].get_pos()

        # --- Ground collision check ---
        if position[2] > 0:
            collided = True

        # --- Aircraft-aircraft proximity check ---
        # Only consider collisions with other alive aircraft.
        for j, aircraft in enumerate(self.Aircrafts):
            if j != i and aircraft.is_alive():
                distance = np.linalg.norm(self.relative_pos(i, j, 'aircraft'))
                if distance < self.collision_distance:
                    collided = True

        return collided

    def check_missile_tone(self, agent_index):
        """
        Update missile tone state for a given agent.

        Conceptually:
          - "Attack tone": how good our current lock is on a valid target.
          - "Defence tone": how threatened we are by enemies who have us in their cones.
          - "Target": which enemy we are currently tracking/locking.

        The function:
          1) Evaluates which enemy aircraft lie inside our attack cone (possible_targets).
          2) Evaluates which enemies also have us in their threat cone (defence tone aggregation).
          3) If we keep the same target, attack tone increments stepwise; otherwise target switches.

        Parameters
        ----------
        agent_index : int
            Index of the aircraft acting as "attacker" for tone updates.

        Returns
        -------
        (tone_attack, tone_defence, target_id) : tuple
            tone_attack : float in [0, 1]
            tone_defence: float in [0, 1]
            target_id   : str (agent id) or "none"
        """

        # Retrieve the agent's last stored tone and target (tone acts like "lock memory").
        new_missile_tone_attack, new_missile_target = self.Aircrafts[agent_index].get_missile_tone_attack()
        new_missile_tone_defence = self.Aircrafts[agent_index].get_missile_tone_defence()

        # Attacker state used to build the forward attack cone.
        attack_cone, _ = self.Aircrafts[agent_index].get_cones()
        attack_pos = self.Aircrafts[agent_index].get_pos()
        attack_vel = self.Aircrafts[agent_index].get_absolute_vel()
        team = self.Aircrafts[agent_index].get_team()

        possible_targets = []   # enemy aircraft currently within our engagement geometry
        max_defence_tone = 0    # strongest threat tone we receive from opponents

        # Optional gating: only compute/refresh tones if we are not already threatened strongly.
        if new_missile_tone_defence < 0.2:
            for i, aircraft in enumerate(self.Aircrafts):
                # Only consider enemy aircraft that are alive.
                if i != agent_index and aircraft.is_alive() and aircraft.get_team() != team:
                    _, defence_cone = aircraft.get_cones()
                    defence_pos = aircraft.get_pos()
                    defence_vel = aircraft.get_absolute_vel()

                    # Attacker's cone to defender: can we engage them?
                    intersect = self.check_intersect_cones(
                        attack_cone, attack_pos, attack_vel,
                        defence_cone, defence_pos, defence_vel
                    )

                    # Defender's cone to attacker: can they engage us?
                    intersected = self.check_intersect_cones(
                        defence_cone, defence_pos, defence_vel,
                        attack_cone, attack_pos, attack_vel
                    )

                    if intersect:
                        possible_targets.append(self.possible_agents[i])

                    # Defensive tone is driven by enemies that have a valid engagement on us.
                    defence_tone, _ = aircraft.get_missile_tone_attack()
                    if intersected and defence_tone > max_defence_tone:
                        max_defence_tone = defence_tone

        # --- Update attack tone + target selection ---
        if len(possible_targets) > 0:
            # Continue locking current target if it is still valid.
            if new_missile_target in possible_targets:
                new_missile_tone_attack = np.clip(
                    new_missile_tone_attack + self.stepwise_tone_increment, 0, 1
                )
            else:
                # Otherwise pick a new valid target and reset tone to "initial lock" value.
                new_missile_target = np.random.choice(possible_targets)
                new_missile_tone_attack = self.stepwise_tone_increment
        else:
            # No valid targets: clear target and lock quality.
            new_missile_target = "none"
            new_missile_tone_attack = 0

        # Defensive tone is the strongest enemy lock quality directed toward us.
        new_missile_tone_defence = max_defence_tone

        return new_missile_tone_attack, new_missile_tone_defence, new_missile_target

    def is_within_cone(self, cone_origin, cone_direction, target_position, angle_deg, min_dist, max_dist):
        """
        Check if a target lies within a finite 3D cone region.

        The cone is defined by:
          - origin (cone_origin)
          - axis direction (cone_direction)
          - angular width (angle_deg) interpreted as full angle, so half-angle is used in test
          - a minimum and maximum distance band [min_dist, max_dist]

        Parameters
        ----------
        cone_origin : array-like (3,)
            Cone origin position.
        cone_direction : array-like (3,)
            Cone axis direction (need not be unit length).
        target_position : array-like (3,)
            Target position.
        angle_deg : float
            Full cone angle in degrees.
        min_dist, max_dist : float
            Distance bounds that define a truncated cone.

        Returns
        -------
        bool
            True if target lies inside the truncated cone volume, else False.
        """
        half_angle_deg = angle_deg / 2

        # Vector from origin to target and its distance.
        vector_to_target = np.array(target_position) - np.array(cone_origin)
        distance = np.linalg.norm(vector_to_target)

        # Distance gating.
        if distance < min_dist or distance > max_dist:
            return False

        # Normalize vectors for angular comparison.
        direction_to_target = vector_to_target / distance
        cone_direction = cone_direction / np.linalg.norm(cone_direction)

        # dot(u, v) = cos(angle) for unit vectors.
        dot = np.dot(cone_direction, direction_to_target)

        # Inside cone if angle <= half-angle  <=> cos(angle) >= cos(half-angle)
        return dot >= np.cos(np.deg2rad(half_angle_deg))

    def check_intersect_cones(self, attack_cone, attack_pos, attack_vel, defence_cone, defence_pos, defence_vel):
        """
        Check whether two aircraft are in a mutual engagement geometry.

        This function expresses a simplified "can shoot / can be shot" relationship:
          - Defender must be inside attacker's forward-facing cone.
          - Attacker must be inside defender's rear-facing vulnerability cone.

        Parameters
        ----------
        attack_cone : tuple
            (angle_deg, min_dist, max_dist) for the attacker’s forward cone.
        attack_pos : array-like (3,)
            Attacker position.
        attack_vel : array-like (3,)
            Attacker velocity (used as forward cone axis).
        defence_cone : tuple
            (angle_deg, min_dist, max_dist) for defender’s rear cone.
        defence_pos : array-like (3,)
            Defender position.
        defence_vel : array-like (3,)
            Defender velocity (rear cone axis uses -defence_vel).

        Returns
        -------
        bool
            True if both cone-inclusion tests succeed.
        """

        attack_angle, attack_min_dist, attack_max_dist = attack_cone
        defence_angle, defence_min_dist, defence_max_dist = defence_cone

        # Attacker forward cone test: does attacker "see" defender inside forward cone?
        attacker_check = self.is_within_cone(
            cone_origin=attack_pos,
            cone_direction=attack_vel,
            target_position=defence_pos,
            angle_deg=attack_angle,
            min_dist=attack_min_dist,
            max_dist=attack_max_dist
        )

        # Defender rear cone test: does defender "expose" rear aspect to attacker?
        defender_check = self.is_within_cone(
            cone_origin=defence_pos,
            cone_direction=-defence_vel,   # rear-facing axis
            target_position=attack_pos,
            angle_deg=defence_angle,
            min_dist=defence_min_dist,
            max_dist=defence_max_dist
        )

        return attacker_check and defender_check

    def fire(self, agent_index, missile_target, missile_tone):
        """
        Attempt to fire a missile if lock quality (tone) is above a threshold.

        This models firing as an instantaneous event:
          - If lock is good enough, a probabilistic hit test is performed.
          - On success, returns the victim identifier ("agent id" or "base").
          - On any fire attempt, the attack tone is reset (lock expended / missile fired).

        Parameters
        ----------
        agent_index : int
            Index of firing aircraft.
        missile_target : str
            Target agent ID or 'base' or 'none'.
        missile_tone : float
            Lock quality in [0,1].

        Returns
        -------
        str
            'none' if no kill, otherwise the target ID ('base' or agent id).
        """
        kill = 'none'

        # Fire only if tone suggests a sufficiently good lock.
        if missile_tone > self.tone_threshold:
            if missile_target != 'base':
                # --- Air-to-air shot ---
                target_index = self.possible_agents.index(missile_target)

                hit = self.check_kill_probability(
                    attacker=agent_index,
                    defender=target_index,
                    tone=missile_tone,
                    type='aircraft'
                )

                kill = missile_target if hit else 'none'
                self.Aircrafts[agent_index].set_missile_tone_attack(0, 'none')

            else:
                # --- Base attack shot ---
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
        Probabilistic hit model for missile firing.

        Aircraft case:
          - Builds a hit probability from lock tone and geometry (track angle margin).

        Base case:
          - Can be handled as a simplified probability model derived from tone.

        Parameters
        ----------
        attacker : int | None
            Index of firing aircraft (None for base shots).
        defender : int | None
            Index of target aircraft (None for base shots).
        tone : float
            Lock quality in [0,1].
        type : str
            'aircraft' or 'base'.

        Returns
        -------
        bool
            True if the Bernoulli trial indicates a successful hit.
        """
        is_hit = False
        bernoulli_threshold = 0

        if type == 'aircraft':
            # --- Geometry-dependent probability shaping ---
            att_aircraft = self.Aircrafts[attacker]
            def_aircraft = self.Aircrafts[defender]

            attack_cone, _ = att_aircraft.get_cones()
            _, defence_cone = def_aircraft.get_cones()

            # Track/adverse angles quantify attacker pointing vs LOS and target aspect vs LOS.
            track_angle, adverse_angle = self.get_track_adverse_angles_norm(att_aircraft, def_aircraft)

            # Angle margin term: higher when track_angle is better aligned with the cone center.
            # Combined with tone to define hit probability threshold.
            att_angle_margin = 0.2 + 0.8 * (
                (np.deg2rad(attack_cone[0] / 2) - (np.pi * track_angle)) / np.deg2rad(attack_cone[0] / 2)
            )
            bernoulli_threshold = att_angle_margin * tone

        # --- Bernoulli sample ---
        sample = np.random.uniform(0.0, 1.0)
        if sample < bernoulli_threshold:
            is_hit = True

        return is_hit

    def get_track_adverse_angles_norm(self, aircraft, target_aircraft):
        """
        Compute track and adverse angles between an aircraft and a target.

        Definitions (conceptual):
          - Track angle: angle between our forward direction and line-of-sight to target.
          - Adverse angle: angle between target forward direction and line-of-sight back to us.

        Output is normalized by pi so that each value lies in [0,1].

        Parameters
        ----------
        aircraft : Aircraft
            Reference aircraft (the "agent").
        target_aircraft : Aircraft
            Target aircraft.

        Returns
        -------
        (track_norm, adverse_norm) : tuple[float, float]
            Track and adverse angles normalized by pi.
        """
        # --- Agent forward direction in world frame ---
        agent_pos = np.array(aircraft.get_agent_telemetry()['position'][-1])
        roll, pitch, yaw = aircraft.get_agent_telemetry()['orientation'][-1]
        agent_R = self.body_to_vehicle(roll, pitch, yaw)

        agent_forward = agent_R @ np.array([1, 0, 0])
        agent_forward_unit = agent_forward / np.linalg.norm(agent_forward)

        # --- Target forward direction in world frame ---
        target_pos = np.array(target_aircraft.get_agent_telemetry()['position'][-1])
        roll, pitch, yaw = target_aircraft.get_agent_telemetry()['orientation'][-1]
        target_R = self.body_to_vehicle(roll, pitch, yaw)

        target_forward = target_R @ np.array([1, 0, 0])
        target_forward_unit = target_forward / np.linalg.norm(target_forward)

        # --- Line of sight (agent -> target) ---
        los_vec = target_pos - agent_pos
        los_vec_unit = los_vec / np.linalg.norm(los_vec)

        track_angle = np.arccos(np.clip(agent_forward_unit @ los_vec_unit, -1.0, 1.0))

        # --- Line of sight (target -> agent) ---
        los_vec = agent_pos - target_pos
        los_vec_unit = los_vec / np.linalg.norm(los_vec)

        adverse_angle = np.arccos(np.clip(target_forward_unit @ los_vec_unit, -1.0, 1.0))

        return track_angle / np.pi, adverse_angle / np.pi

    def get_closure_rate_norm(self, aircraft, target_aircraft):
        """
        Compute normalized closure rate between aircraft and target.

        Closure rate is the negative projection of relative velocity onto the
        line-of-sight unit vector (positive closure typically means "closing in").

        The return value is normalized through an arctangent shaping function to:
          - compress extreme values
          - keep the result bounded for learning

        Parameters
        ----------
        aircraft : Aircraft
            Reference aircraft.
        target_aircraft : Aircraft
            Target aircraft.

        Returns
        -------
        float
            Closure rate normalized to approximately [-1, 1].
        """
        agent_pos = np.array(aircraft.get_agent_telemetry()['position'][-1])
        agent_vel = aircraft.get_absolute_vel()

        target_pos = np.array(target_aircraft.get_agent_telemetry()['position'][-1])
        target_vel = target_aircraft.get_absolute_vel()

        # Relative velocity (target wrt agent)
        rel_vel = target_vel - agent_vel

        # LOS direction from agent to target
        los_vec = target_pos - agent_pos
        los_unit = los_vec / (np.linalg.norm(los_vec) + 1e-6)

        # Closure rate: negative dot(rel_vel, LOS) so closing yields positive values
        closure = -np.dot(rel_vel, los_unit)

        # Squash into bounded range using atan-based normalization.
        return np.atan(np.deg2rad(closure)) / np.atan(np.deg2rad(686))

    def get_individual_reward(self, agent_index, action, kill, missile_tone_attack, missile_tone_defence, missile_target):
        """
        Compute reward and termination flags for a single agent.

        Reward components:
          - Flight envelope shaping:
              * AoA penalty, sideslip penalty
              * speed penalty (stall/low-speed discouragement)
              * altitude penalty (stay in desired band)
              * smoothing penalty (discourage large action deltas)
          - Pursuit shaping:
              * pursuit/angle advantage terms
              * closure shaping around an "optimal distance" zone
          - Sparse events:
              * trigger discipline / firing logic shaping
              * attack/defence tone shaping
              * kill bonus
              * termination penalty

        This function also updates:
          - self.attack_metric (tracking occurrences of attack target engagement)
          - self.kill_metric (tracking kills)
          - per-agent episode reward logs and cumulative returns

        Parameters
        ----------
        agent_index : int
            Index of the aircraft being rewarded.
        action : array-like
            The current action for this agent (normalized controls + trigger).
        kill : str
            'none', 'base', or agent id of a killed target.
        missile_tone_attack : float
            Attack lock quality in [0,1].
        missile_tone_defence : float
            Defensive threat tone in [0,1].
        missile_target : str
            Current target agent id or 'none'.

        Returns
        -------
        (reward, terminated, truncated) : tuple[float, bool, bool]
            reward     : scalar float reward for this step
            terminated : whether this agent is terminated due to failure/out-of-bounds
            truncated  : whether this agent is truncated (time-limit style)
        """
        terminated = False
        truncated = False

        # Component dictionaries allow structured logging/debugging of reward terms.
        reward_Flight = {}
        reward_Pursuit = {}
        sparse_reward = {}
        Total_Reward = {}

        aircraft = self.Aircrafts[agent_index]
        team = aircraft.get_team()

        telemetry = aircraft.get_agent_telemetry()
        vel = telemetry['velocity'][-1]
        AoA = np.rad2deg(telemetry['AoA'][-1])
        sideslip = np.rad2deg(telemetry['sideslip'][-1])
        altitude = -telemetry['position'][-1][2]

        # Previous and current commands (used for smoothing penalty).
        [pre_UpAngle_C, pre_SideAngle_C, pre_Speed_C, pre_trigger] = telemetry['commands'][-2]
        [UpAngle_C, SideAngle_C, Speed_C, trigger] = action

        reward_config = self.Reward_Config

        # =========================================================
        # Flight-related shaping rewards (stability / envelope)
        # =========================================================

        # AoA penalty (smooth sigmoid-like penalty beyond a critical region).
        AoA_Norm = abs(AoA) / reward_config['Terminal_AoA']
        AoA_MID = reward_config['Critical_AoA'] / reward_config['Terminal_AoA']
        AoA_ALPHA = 12
        reward_Flight['AoA'] = -reward_config['AoA_W'] * (1 / (1 + np.exp(-AoA_ALPHA * (AoA_Norm - AoA_MID))))

        # Sideslip penalty.
        SS_Norm = abs(sideslip) / reward_config['Terminal_Sideslip']
        SS_MID = reward_config['Critical_Sideslip'] / reward_config['Terminal_Sideslip']
        SS_ALPHA = 12
        reward_Flight['Sideslip'] = -reward_config['Sideslip_W'] * (1 / (1 + np.exp(-SS_ALPHA * (SS_Norm - SS_MID))))

        # Speed penalty (encourage staying above critical speed).
        speed = vel[0]
        Speed_Filtered_Norm = max(reward_config['Critical_Speed'] - speed, 0) / (
            (reward_config['Critical_Speed'] - reward_config['Terminal_Speed'])
        )
        Speed_MID = 0.5
        Speed_ALPHA = 12
        reward_Flight['Speed'] = -reward_config['Speed_W'] * (1 / (1 + np.exp(-Speed_ALPHA * (Speed_Filtered_Norm - Speed_MID))))

        # Altitude penalty (encourage staying near a desired band around mid-altitude).
        Altitude_Norm = abs(self.env_size[2] / 2 - altitude) / (self.env_size[2] / 2)
        Altitude_MID = reward_config['Critical_Altitude'] / (self.env_size[2] / 2)
        Altitude_ALPHA = 12
        reward_Flight['Altitude'] = -reward_config['Altitude_W'] * (1 / (1 + np.exp(-Altitude_ALPHA * (Altitude_Norm - Altitude_MID))))

        # Command smoothing penalty (discourage large changes in steering commands).
        UpAngleDelta = abs(UpAngle_C - pre_UpAngle_C)
        SideAngleDelta = abs(SideAngle_C - pre_SideAngle_C)
        CombinedNorm = (UpAngleDelta + SideAngleDelta) / 2
        Delta_MID = reward_config['Critical_Delta']
        Delta_ALPHA = 12
        reward_Flight['Smoothing'] = -reward_config['Smoothing_W'] * (1 / (1 + np.exp(-Delta_ALPHA * (CombinedNorm - Delta_MID))))

        # =========================================================
        # Pursuit-related shaping rewards (tactical geometry)
        # =========================================================

        # Identify a single reference opponent (e.g., closest enemy) for shaping.
        closest_enemy_plane = None
        dist = 1000000

        for i, enemy_aircraft in enumerate(self.Aircrafts):
            if enemy_aircraft.get_team() != team and enemy_aircraft.is_alive():
                rel_pos = np.linalg.norm(self.relative_pos(agent_index, i, 'aircraft'))
                if rel_pos < dist:
                    dist = rel_pos
                    closest_enemy_plane = enemy_aircraft

        if closest_enemy_plane is not None:
            track_angle, adverse_angle = self.get_track_adverse_angles_norm(aircraft, closest_enemy_plane)
            closure = self.get_closure_rate_norm(aircraft, closest_enemy_plane)

            _, def_cone = closest_enemy_plane.get_cones()
            optimal_zone = reward_config['optimal_zone']
            optimal_distance = (def_cone[1] + def_cone[2]) / 2

            # Angle advantage encourages being behind/inside the opponent's vulnerable region.
            TPAR = reward_config['tan_parameter']
            angle_advantage = adverse_angle - track_angle

            # Pursuit shaping using a tan-based smooth mapping.
            shaped_pursuit = np.tan((angle_advantage) * (np.pi / TPAR)) / np.tan(np.pi / TPAR)
            reward_Pursuit['Pursuit'] = shaped_pursuit * reward_config['AW']

            # Closure shaping varies with distance relative to an optimal intercept zone.
            smooth_weights = [
                np.clip((dist - optimal_distance) / (def_cone[2] - optimal_distance), 0, 1),
                np.clip((optimal_distance - dist) / (optimal_distance - def_cone[1]), 0, 1),
                1 - np.clip(abs(abs(optimal_distance - dist) - optimal_zone) / optimal_zone, 0, 1)
            ]

            closure_dist_norm = (
                smooth_weights[0] * (closure) * (1 - track_angle) +
                smooth_weights[1] * (-closure) +
                smooth_weights[2] * (1 - abs(closure))
            )

            reward_Pursuit['Closure'] = closure_dist_norm * reward_config['CW']

            # Trigger discipline: penalize pulling trigger without proper lock/threshold behavior.
            sparse_reward['Trigger'] = -(
                (missile_tone_attack == 1) *
                abs(np.clip(trigger, -1, self.trigger_threshold) - self.trigger_threshold) *
                reward_config['trigger_penalty']
            )

            # Tone-based sparse shaping (attack encourages lock, defence discourages exposure).
            sparse_reward['Attack'] = reward_config['att_tone_bonus'] * missile_tone_attack * track_angle
            if missile_target != 'none':
                self.attack_metric += 1

            sparse_reward['Defence'] = -reward_config['def_tone_bonus'] * missile_tone_defence * adverse_angle

        else:
            # If no enemies are alive, pursuit/engagement shaping is zeroed.
            reward_Pursuit['Pursuit'] = 0
            reward_Pursuit['Closure'] = 0
            sparse_reward['Attack'] = 0
            sparse_reward['Defence'] = 0
            sparse_reward['Trigger'] = 0

        # Kill bonus (sparse success reward).
        if kill != 'none':
            sparse_reward['Kill'] = reward_config['kill_bonus'] * missile_tone_attack
            self.kill_metric += 1

        # =========================================================
        # Termination checks (safety, bounds, collisions)
        # =========================================================
        if (
            self.check_collision(agent_index)
            or vel[0] < reward_config['Terminal_Speed']
            or abs(AoA) > reward_config['Terminal_AoA']
            or abs(sideslip) > reward_config['Terminal_Sideslip']
            or altitude > self.env_size[2]
            or aircraft.get_distance_from_centroid(self.bases) > self.max_size
        ):
            self.Aircrafts[agent_index].kill()
            terminated = True
            sparse_reward['Termination'] = -reward_config['terminal_penalty']

        # =========================================================
        # Merge reward components
        # =========================================================
        normalized_reward_Pursuit = sum(reward_Pursuit.values())
        normalized_reward_Flight = sum(reward_Flight.values())
        sparse_reward_sum = sum(sparse_reward.values())

        normalized_reward = (
            reward_config['GFW'] * normalized_reward_Flight +
            reward_config['PW'] * normalized_reward_Pursuit
        ) + sparse_reward_sum

        # Store full reward breakdown for reporting/logging.
        Total_Reward.update(reward_Flight)
        Total_Reward.update(reward_Pursuit)
        Total_Reward['Attack'] = sparse_reward['Attack']
        Total_Reward['Defence'] = sparse_reward['Defence']

        # Per-step reward logging (structured).
        self.episode_rewards[self.possible_agents[agent_index]].append(Total_Reward.copy())
        self.episode_returns[self.possible_agents[agent_index]] += normalized_reward

        return normalized_reward, terminated, truncated

    def CLI_report(self, telemetry, action):
        """
        Print a human-readable snapshot of aircraft telemetry and control commands.

        Useful for debugging PID behavior or validating state evolution.

        Parameters
        ----------
        telemetry : dict
            Telemetry dictionary containing time series of state.
            This function prints the latest entry of key fields.
        action : array-like
            Control inputs in the order: (throttle, elevon, aileron, rudder, fire).
        """
        t, e, a, r, f = action
        roll, pitch, yaw = telemetry['orientation'][-1]

        print(f"Position:     {telemetry['position'][-1].round(2)}")
        print(f"Velocity:     {telemetry['velocity'][-1].round(2)}  Acceleration: {telemetry['acceleration'][-1].round(2)}")
        print(f"Orientation:  Roll={np.rad2deg(roll):.2f}°, Pitch={np.rad2deg(pitch):.2f}°, Yaw={np.rad2deg(yaw):.2f}°")
        print(f"AoA:          {telemetry['AoA'][-1]:.3f}   Sideslip: {telemetry['sideslip'][-1]:.3f}")
        print(f"Forces:       {telemetry['force'][-1].round(2)}")
        print(f"Moments:      {telemetry['moment'][-1].round(2)}")
        print(f"Controls:     Throttle={t:.2f}, Elevon={e:.1f}, Aileron={a:.1f}, Rudder={r:.1f}, Fire={f}")
        print()

    def discretizer(self, action):
        """
        Discretize continuous action values into a quantized grid.

        This is useful when:
          - training with discrete-like control,
          - limiting control authority,
          - improving stability by reducing high-frequency action jitter.

        The fire command is passed through unchanged.

        Parameters
        ----------
        action : array-like length 4
            [UpAngle, SideAngle, speed, fire] in normalized ranges.

        Returns
        -------
        np.ndarray
            Discretized [UpAngle, SideAngle, speed, fire].
        """
        UpAngle, SideAngle, speed, fire = action

        def round_to_step(value, step, min_val, max_val):
            # Clip then snap to nearest discrete step.
            value = np.clip(value, min_val, max_val)
            steps = round((value - min_val) / step)
            return min_val + steps * step

        speed_discretized = round_to_step(speed, self.speed_step, 0.0, 1.0)
        UpAngle_discretized = round_to_step(UpAngle, self.UpAngle_step, -1.0, 1.0)
        SideAngle_discretized = round_to_step(SideAngle, self.SideAngle_step, -1.0, 1.0)

        return np.array([
            UpAngle_discretized,
            SideAngle_discretized,
            speed_discretized,
            fire
        ])

    def step(self, action_dict):
        """
        Advance the environment by one control step.

        Per-agent pipeline:
          1) Process action (optional discretization).
          2) Step aircraft dynamics (multi-rate integration).
          3) Update missile tones and determine firing trigger.
          4) Resolve missile firing and potential kills.
          5) Compute reward and termination flags.
          6) Update per-agent missile tone memory.

        Also enforces:
          - global time-limit termination
          - team-elimination termination (no alive aircraft in a team)

        Parameters
        ----------
        action_dict : dict[str, np.ndarray]
            Mapping agent_id -> action vector.

        Returns
        -------
        obs : dict[str, np.ndarray]
        rewards : dict[str, float]
        terminated : dict[str, bool]
        truncated : dict[str, bool]
        info : dict
            Contains '__common__' metrics for reporting.
        """
        self.episode_steps += 1

        rewards = {}
        terminated = {'__all__': True}
        truncated = {'__all__': False}

        # ---------------------------------------------------------
        # Step each agent that provided an action
        # ---------------------------------------------------------
        for player, a in action_dict.items():
            agent_index = self.possible_agents.index(player)

            if self.Aircrafts[agent_index].is_alive():
                # Optional action discretization
                action = self.discretizer(a) if self.discretize else a

                # Step aircraft (internally handles frequency_factor physics updates)
                self.Aircrafts[agent_index].step(action, self.frequency_factor)

                # Evaluate missile tones before firing
                missile_tone_attack, missile_tone_defence, missile_target = self.check_missile_tone(agent_index)

                # Determine firing trigger (dummy may use special trigger rules)
                kill = 'none'
                if self.Aircrafts[agent_index].is_dummy():
                    trigger = self.dummy_kill
                else:
                    trigger = action[-1] > self.trigger_threshold or self.autotrigger

                # Fire if triggered, then refresh tones after shot resolution
                if trigger:
                    kill = self.fire(agent_index, missile_target, missile_tone_attack)
                    missile_tone_attack, missile_tone_defence, missile_target = self.check_missile_tone(agent_index)

                # -------------------------------------------------
                # Resolve kill effects
                # -------------------------------------------------
                if kill != 'none':
                    if kill == 'base':
                        # Base destruction ends the episode.
                        terminated['__all__'] = True
                    else:
                        # Mark victim killed and apply victim penalty.
                        victim_index = self.possible_agents.index(missile_target)
                        self.Aircrafts[victim_index].kill()

                        victim_id = self.possible_agents[victim_index]
                        rewards[victim_id] = -self.Reward_Config['killed_penalty']
                        self.episode_rewards[victim_id].append({'Killed': -self.Reward_Config['killed_penalty']})
                        self.episode_returns[victim_id] += -self.Reward_Config['killed_penalty']

                # Compute this agent's reward + termination flags
                rewards[player], terminated[player], truncated[player] = self.get_individual_reward(
                    agent_index, action, kill,
                    missile_tone_attack, missile_tone_defence,
                    missile_target
                )

                # If at least one agent is not terminated, episode continues.
                if not terminated[player]:
                    terminated['__all__'] = False

                # Persist tone state to the Aircraft object (used as memory into next step)
                self.Aircrafts[agent_index].set_missile_tone_attack(missile_tone_attack, missile_target)
                self.Aircrafts[agent_index].set_missile_tone_defence(missile_tone_defence)

            else:
                # Dead agents: no reward, marked terminated for this step.
                rewards[player] = 0.0
                terminated[player] = True
                truncated[player] = False

        # ---------------------------------------------------------
        # Global termination: time limit
        # ---------------------------------------------------------
        if self.episode_steps > self.max_steps:
            terminated['__all__'] = True

        # ---------------------------------------------------------
        # Global termination: team elimination
        # ---------------------------------------------------------
        for team in range(self.num_teams):
            at_least_one_alive = 0
            for a in range(self.num_agents_team):
                if self.Aircrafts[team * self.num_agents_team + a].is_alive():
                    at_least_one_alive += 1

            # If a team is fully eliminated, end the episode for all.
            if at_least_one_alive == 0:
                for k in terminated.keys():
                    terminated[k] = True
                    terminated['__all__'] = True

        return (
            self.get_obs(),
            rewards,
            terminated,
            truncated,
            {'__common__': {'attack_steps': self.attack_metric, 'kills': self.kill_metric}}
        )

    def get_winning_team(self):
        """
        Determine the winning team at the end of an episode based on team return.

        Team score is computed as the sum of individual episode returns for all agents
        belonging to that team. A "draw" is declared if the top team is within a small
        margin of at least one other team.

        Returns
        -------
        str
            "team_k" for the winning team, or "draw" if the outcome is considered tied.
        """
        score = {}

        # --- Aggregate per-agent returns into per-team totals ---
        for team in range(self.num_teams):
            score[f'team_{team}'] = 0
            for air in range(self.num_agents_team):
                agent_id = self.possible_agents[team * self.num_agents_team + air]
                score[f'team_{team}'] += self.episode_returns[agent_id]

        # --- Determine winner with a "close margin → draw" rule ---
        teams = np.array(list(score.keys()))
        scores = np.array(list(score.values()))

        m = scores.max()               # best score
        mask = scores != m             # all non-best teams
        tied = np.any((m - scores[mask]) <= 100)  # within 100 points of the leader

        return "draw" if tied else teams[np.argmax(scores)]

    def body_to_vehicle(self, roll, pitch, yaw):
        """
        Rotation matrix from body frame to world frame.

        This is the transpose (inverse) of the world->body matrix because rotation
        matrices are orthonormal (R^{-1} = R^T).

        Parameters
        ----------
        roll, pitch, yaw : float
            Euler angles in radians using aerospace ZYX convention.

        Returns
        -------
        np.ndarray shape (3,3)
            Rotation matrix mapping body vectors into world coordinates.
        """
        return self.vehicle_to_body(roll, pitch, yaw).T

    def vehicle_to_body(self, roll, pitch, yaw):
        """
        Rotation matrix from world frame to body frame.

        Uses a ZYX (yaw -> pitch -> roll) Euler rotation convention, typical in aerospace:
          1) rotate about Z by yaw
          2) rotate about Y by pitch
          3) rotate about X by roll

        Parameters
        ----------
        roll, pitch, yaw : float
            Euler angles in radians.

        Returns
        -------
        np.ndarray shape (3,3)
            Rotation matrix mapping world vectors into body coordinates.
        """
        # Precompute trig for readability and speed.
        cr = np.cos(roll);  sr = np.sin(roll)
        cp = np.cos(pitch); sp = np.sin(pitch)
        cy = np.cos(yaw);   sy = np.sin(yaw)

        # ZYX rotation matrix (world -> body).
        R = np.array([
            [cp * cy,                 cp * sy,                 -sp],
            [sr * sp * cy - cr * sy,  sr * sp * sy + cr * cy,  sr * cp],
            [cr * sp * cy + sr * sy,  cr * sp * sy - sr * cy,  cr * cp]
        ])
        return R

    def render(self, screen_size=(800, 800), mode='rgb_array', altitude_range=(0, 100)):
        """
        Render a 2D visualization of the environment using pygame.

        The renderer:
          - Draws a background and bases/aircraft in a scaled top-down view (XY plane).
          - Encodes altitude into alpha (transparency) so height differences are visible.
          - Draws per-aircraft attack and defence cones as semi-transparent sectors.
          - Can return an RGB array (for video logging) or display a window (human mode).

        Parameters
        ----------
        screen_size : tuple(int,int)
            Output render size (width, height).
        mode : str
            'rgb_array' returns a NumPy image array.
            'human' opens a window and blits the current frame.
        altitude_range : tuple(float,float)
            Range of Z values mapped to alpha. Z is "positive down" in this convention.

        Returns
        -------
        np.ndarray | None
            RGB array (H,W,3) when mode='rgb_array', else None.
        """
        SEA_BLUE = (0, 105, 148)
        self._screen.fill(SEA_BLUE)

        # Color palette used to distinguish teams visually.
        TEAM_COLORS = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]

        def altitude_to_alpha(z, z_min, z_max):
            """
            Map altitude (Z-down) to alpha so higher altitude appears more opaque.
            """
            z = np.clip(z, z_min, z_max)
            norm = 1 - (z - z_min) / (z_max - z_min)
            return int(norm * 255)

        def rotate_point(x, y, angle_rad):
            """
            Rotate a 2D point about the origin by angle_rad (CCW).
            Used to rotate the aircraft triangle by yaw.
            """
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)

        # If bases haven't been set up yet, return a blank image.
        if not self.bases:
            return np.zeros((screen_size[1], screen_size[0], 3), dtype=np.uint8)

        # ---------------------------------------------------------
        # Compute scaling so the full scene fits on screen.
        # Uses bases + current aircraft positions to determine bounds.
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # Draw aircraft as triangles (alive) or dots (dead),
        # plus cones and labels for debugging.
        # ---------------------------------------------------------
        for idx, aircraft in enumerate(self.Aircrafts):
            telemetry = aircraft.get_physics_telemetry()
            if not telemetry["position"]:
                continue

            pos = telemetry["position"][-1]
            ori = telemetry["orientation"][-1]
            x, y, z = pos
            yaw = ori[2]

            # Project world XY into screen coordinates (Y is inverted for display).
            screen_x = int((x - center_x) * scale + screen_size[0] / 2)
            screen_y = int(screen_size[1] / 2 - (y - center_y) * scale)

            if aircraft.is_alive():
                # Team color selection based on index -> team mapping.
                team_id = idx // self.num_agents_team
                color = TEAM_COLORS[team_id % len(TEAM_COLORS)]

                # Encode altitude into transparency for easier height perception.
                alpha = altitude_to_alpha(z, *altitude_range)

                # --- Draw aircraft triangle ---
                shape = [(7, 0), (-4, 3), (-4, -3)]
                rotated = [rotate_point(px, py, -yaw) for px, py in shape]
                points = [(screen_x + px, screen_y + py) for px, py in rotated]
                pygame.draw.polygon(self._screen, (*color, alpha), points)

                # --- Draw attack/defence cones as filled sectors ---
                attack_cone_params, defence_cone_params = aircraft.get_cones()
                if attack_cone_params is not None:
                    a_cone_angle_deg, a_cone_min_dist, a_cone_max_dist = attack_cone_params
                    d_cone_angle_deg, d_cone_min_dist, d_cone_max_dist = defence_cone_params

                    num_segments = 15  # polygon resolution for the arc
                    a_min_len_px = a_cone_min_dist * scale
                    a_max_len_px = a_cone_max_dist * scale
                    d_min_len_px = d_cone_min_dist * scale
                    d_max_len_px = d_cone_max_dist * scale

                    # Draw two cones: forward attack (angle_offset=0) and rear defence (angle_offset=pi).
                    for label, angle_offset, a in [
                        ("attack", 0, 60),
                        ("defence", np.pi, 50)
                    ]:
                        cone_angle = yaw + angle_offset
                        cone_pts = []

                        if label == "attack":
                            # Outer arc boundary (max distance)
                            for i in range(num_segments + 1):
                                ang = cone_angle - np.radians(a_cone_angle_deg) / 2 + i * np.radians(a_cone_angle_deg) / num_segments
                                dx = a_max_len_px * np.cos(ang)
                                dy = -a_max_len_px * np.sin(ang)
                                cone_pts.append((screen_x + dx, screen_y + dy))

                            # Inner arc boundary (min distance), reversed so polygon closes properly
                            for i in reversed(range(num_segments + 1)):
                                ang = cone_angle - np.radians(a_cone_angle_deg) / 2 + i * np.radians(a_cone_angle_deg) / num_segments
                                dx = a_min_len_px * np.cos(ang)
                                dy = -a_min_len_px * np.sin(ang)
                                cone_pts.append((screen_x + dx, screen_y + dy))
                        else:
                            # Defence cone drawn behind aircraft
                            for i in range(num_segments + 1):
                                ang = cone_angle - np.radians(d_cone_angle_deg) / 2 + i * np.radians(d_cone_angle_deg) / num_segments
                                dx = d_max_len_px * np.cos(ang)
                                dy = -d_max_len_px * np.sin(ang)
                                cone_pts.append((screen_x + dx, screen_y + dy))

                            for i in reversed(range(num_segments + 1)):
                                ang = cone_angle - np.radians(d_cone_angle_deg) / 2 + i * np.radians(d_cone_angle_deg) / num_segments
                                dx = d_min_len_px * np.cos(ang)
                                dy = -d_min_len_px * np.sin(ang)
                                cone_pts.append((screen_x + dx, screen_y + dy))

                        # Draw cone polygon on a transparent surface so alpha is respected.
                        cone_surface = pygame.Surface(screen_size, pygame.SRCALPHA)
                        pygame.draw.polygon(cone_surface, (*color, a), cone_pts)
                        self._screen.blit(cone_surface, (0, 0))

                # --- Debug label: id, altitude, forward speed, alive flag, tone and target ---
                missile_tone, missile_target = aircraft.get_missile_tone_attack()
                target_str = str(missile_target) if missile_target is not None else "None"

                label_text = (
                    f"{idx} - alt:{z:.2f} - vel:{telemetry['velocity'][-1][0]:.2f} - "
                    f"{aircraft.is_alive()}: {missile_tone:.2f} → {target_str}"
                )
                label = self.font.render(label_text, True, color)
                self._screen.blit(label, (screen_x + 10, screen_y - 10))

            else:
                # Dead aircraft are drawn as a simple translucent dot.
                radius = 8
                dead_color = (0, 0, 0, 100)
                s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, dead_color, (radius, radius), radius)
                self._screen.blit(s, (screen_x - radius, screen_y - radius))

                label_text = f"{idx} - alt:{z:.2f}"
                label = self.font.render(label_text, True, dead_color)
                self._screen.blit(label, (screen_x + 10, screen_y - 10))

        # ---------------------------------------------------------
        # Output: either return an RGB array or display interactively
        # ---------------------------------------------------------
        if mode == 'rgb_array':
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
        Save a 3D visualization of all agents' trajectories with velocity and orientation cues.

        This is intended for offline debugging and evaluation:
          - plots 3D path (X, Y, altitude up)
          - draws velocity vectors at intervals
          - draws body-frame axes (nose/wing/tail) at intervals
          - saves both a static PNG and an interactive Plotly HTML

        Parameters
        ----------
        save_folder : str
            Output directory.
        every_n : int
            Plot arrows every N samples to reduce clutter.
        scale : float
            Visual length scaling for quiver arrows.
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

            # Plot trajectory (flip Z to altitude-up for human readability).
            ax.plot(
                positions[:, 0], positions[:, 1], -positions[:, 2],
                color=color, linewidth=2, label=f'{label_prefix} Trajectory'
            )

            all_positions.append(positions)

            # Plot velocity vectors and body axes at sparse intervals.
            for i in range(0, len(positions), every_n):
                pos = [positions[i][0], positions[i][1], -positions[i][2]]
                roll, pitch, yaw = orientations[i]

                # Convert body velocity into world frame before drawing.
                R = self.body_to_vehicle(roll, pitch, yaw)
                vel = R @ velocities[i]
                vel[2] = -vel[2]  # flip for altitude-up plotting

                ax.quiver(*pos, *vel, length=scale, normalize=True, color=color)

                # Body-frame axes rendered in world frame (for attitude intuition).
                body_axes = {
                    'Nose': (R @ np.array([1, 0, 0]), 'red'),
                    'Wing': (R @ np.array([0, -1, 0]), 'green'),
                    'Tail': (R @ np.array([0, 0, 1]), 'magenta')
                }

                for label, (vec, axis_color) in body_axes.items():
                    vec[2] = -vec[2]
                    ax.quiver(*pos, *vec, length=scale * 0.5, normalize=True, color=axis_color)

            # Track global span for consistent axis scaling.
            max_range = max(max_range, np.ptp(positions, axis=0).max())

        # Compute a global center and apply symmetric bounds for equal scaling.
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

        # Save static PNG.
        path = os.path.join(save_folder, "all_agents_trajectory.png")
        plt.savefig(path)

        # Save interactive Plotly version.
        plotly_fig = self.mpl3d_to_plotly(fig)
        html_path = os.path.join(save_folder, "all_agents_trajectory.html")
        plotly_fig.write_html(html_path)

        plt.close()

    def mpl3d_to_plotly(self, fig):
        """
        Convert a Matplotlib 3D figure into a Plotly 3D figure.

        This is used to preserve interactive exploration (rotate/zoom) of the same
        trajectories that were drawn in Matplotlib.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            A figure containing one or more 3D axes.

        Returns
        -------
        plotly.graph_objects.Figure
            Plotly figure with 3D line traces for each Matplotlib 3D line.
        """
        def mpl_rgba_to_plotly(rgba):
            """
            Convert Matplotlib RGBA (floats 0..1) into Plotly rgba() CSS string.
            """
            r, g, b, a = rgba
            return f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {a})"

        plotly_fig = go.Figure()

        # Matplotlib 3D axes can be identified by projection method presence.
        axes_3d = [ax for ax in fig.get_axes() if hasattr(ax, 'get_proj')]

        for ax in axes_3d:
            # Recreate each trajectory line as a Plotly 3D line trace.
            for line in ax.lines:
                x, y, z = line.get_data_3d()
                label = line.get_label()

                plotly_fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    name=label if label != '_nolegend_' else None,
                    line=dict(
                        color=mpl_rgba_to_plotly(line.get_color()),
                        width=line.get_linewidth()
                    )
                ))

            # Copy axis labels for parity between Matplotlib and Plotly views.
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
        Export telemetry for each agent to CSV and save a multi-panel diagnostic plot.

        For each agent:
          - Write a CSV with time series for position/velocity/accel/orientation/forces/moments/commands/AoA/sideslip
          - Save an 8-panel PNG with the same signals for quick inspection

        Parameters
        ----------
        save_folder : str
            Output directory for CSV and PNG files.
        """
        for idx, agent in enumerate(self.possible_agents):
            telemetry = self.Aircrafts[idx].get_agent_telemetry()

            # Convert telemetry lists into arrays for indexing and plotting.
            pos = np.array(telemetry['position'])
            vel = np.array(telemetry['velocity'])
            accel = np.array(telemetry['acceleration'])
            eulers = np.rad2deg(np.array(telemetry['orientation']))
            force = np.array(telemetry['force'])
            moment = np.array(telemetry['moment'])
            cmds = np.array(telemetry['commands'])
            AoA = np.rad2deg(np.array(telemetry['AoA']))
            sideslip = np.rad2deg(np.array(telemetry['sideslip']))

            # --- Save CSV for offline analysis ---
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

            os.makedirs(save_folder, exist_ok=True)
            csv_path = os.path.join(save_folder, f"{agent}_telemetry.csv")
            df.to_csv(csv_path, index=False)

            # --- Plot: 8 stacked panels for quick debugging ---
            fig, axs = plt.subplots(8, 1, figsize=(18, 18), sharex=True)

            axs[0].plot(pos[:, 0], label='X')
            axs[0].plot(pos[:, 1], label='Y')
            axs[0].plot(pos[:, 2], label='Z')
            axs[0].set_title("Position (m)")
            axs[0].legend()

            axs[1].plot(vel[:, 0], label='Vx')
            axs[1].plot(vel[:, 1], label='Vy')
            axs[1].plot(vel[:, 2], label='Vz')
            axs[1].set_title("Velocity (m/s)")
            axs[1].legend()

            axs[2].plot(accel[:, 0], label='Ax')
            axs[2].plot(accel[:, 1], label='Ay')
            axs[2].plot(accel[:, 2], label='Az')
            axs[2].set_title("Acceleration (m/s²)")
            axs[2].legend()

            axs[3].plot(eulers[:, 0], label='Roll (°)')
            axs[3].plot(eulers[:, 1], label='Pitch (°)')
            axs[3].plot(eulers[:, 2], label='Yaw (°)')
            axs[3].set_title("Euler Angles (degrees)")
            axs[3].legend()

            axs[4].plot(force[:, 0], label='Fx')
            axs[4].plot(force[:, 1], label='Fy')
            axs[4].plot(force[:, 2], label='Fz')
            axs[4].set_title("Forces (N)")
            axs[4].legend()

            axs[5].plot(moment[:, 0], label='Mx')
            axs[5].plot(moment[:, 1], label='My')
            axs[5].plot(moment[:, 2], label='Mz')
            axs[5].set_title("Moments (Nm)")
            axs[5].legend()

            axs[6].plot(cmds[:, 0], label='UpAngle')
            axs[6].plot(cmds[:, 1], label='SideAngle')
            axs[6].plot(cmds[:, 2], label='Speed')
            axs[6].plot(cmds[:, 3], label='Fire')
            axs[6].set_title("Control Inputs")
            axs[6].legend()

            axs[7].plot(AoA, label='AoA (°)')
            axs[7].plot(sideslip, label='Sideslip (°)')
            axs[7].set_title("Wind Angles")
            axs[7].legend()

            plt.tight_layout()

            fig_path = os.path.join(save_folder, f"{agent}_telemetry_plot.png")
            plt.savefig(fig_path)
            plt.close()

    def plot_rewards(self, save_folder):
        """
        Plot per-agent reward components over time and save to disk.

        Rewards are stored per agent as a list of dictionaries, where each dictionary
        contains named reward components for that timestep. This plot:
          - builds a union of all keys used during the episode
          - plots each component as its own line

        Parameters
        ----------
        save_folder : str
            Output directory for the plot image.
        """
        os.makedirs(save_folder, exist_ok=True)

        num_agents = len(self.episode_rewards)
        fig, axs = plt.subplots(num_agents, 1, figsize=(12, 4 * num_agents), constrained_layout=True)

        # If only one agent exists, wrap axes so code stays consistent.
        if num_agents == 1:
            axs = [axs]

        for idx, (agent_name, rewards_list) in enumerate(self.episode_rewards.items()):
            if not rewards_list:
                continue

            # All components that appeared at least once during the episode.
            all_keys = set().union(*(r.keys() for r in rewards_list))
            timesteps = list(range(len(rewards_list)))

            # Draw one line per component (missing values default to 0).
            for key in sorted(all_keys):
                values = [r.get(key, 0.0) for r in rewards_list]
                axs[idx].plot(timesteps, values, label=key)

            axs[idx].set_title(f"Agent: {agent_name}")
            axs[idx].set_xlabel("Timestep")
            axs[idx].set_ylabel("Reward Value")
            axs[idx].legend()
            axs[idx].grid(True)

        plt.suptitle("Per-Agent Reward Components Over Time", fontsize=16)
        save_path = os.path.join(save_folder, "agent_rewards.png")
        plt.savefig(save_path)
        plt.close()


def Test_env():
    """
    Run a short scripted rollout in the multi-agent AerialBattle environment and save outputs.

    This is a developer-facing integration test (not a unit test). It validates that:
      - YAML configuration loads correctly
      - Environment reset/step loops run without crashing
      - Rendering produces frames suitable for video logging
      - Trajectory/telemetry/reward exporters produce artifacts on disk

    Outputs written to disk:
      - ENV_TEST/test_1.mp4                 : rendered episode video
      - ENV_TEST/all_agents_trajectory.*    : 3D trajectory plot (PNG + interactive HTML)
      - ENV_TEST/<agent>_telemetry.csv/png  : per-agent telemetry dumps
      - ENV_TEST/agent_rewards.png          : per-agent reward component curves
    """

    # ---------------------------------------------------------------------
    # 1) Load configuration
    # ---------------------------------------------------------------------
    # The YAML is expected to provide the environment parameters plus UAV model config.
    with open("Train_Run_config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # ---------------------------------------------------------------------
    # 2) Create the environment instance
    # ---------------------------------------------------------------------
    # env_config: physics/action rates, map size, reward config, etc.
    # uav_config: aircraft parameters, gains, cones, aero coefficients, etc.
    env = AerialBattle(config["env_config"], config['uav_config'])

    # List to accumulate frames for video export.
    images = []

    # Optional: override team 0's aircraft model before starting the episode.
    env.set_plane_model(0, 5)

    # ---------------------------------------------------------------------
    # 3) Reset environment and capture initial frame
    # ---------------------------------------------------------------------
    observations = env.reset()          # initial observations (multi-agent dict)
    images.append(env.render())         # capture first frame for the video

    # ---------------------------------------------------------------------
    # 4) Define scripted action schedule
    # ---------------------------------------------------------------------
    # Action format: [UpAngle, SideAngle, Speed, Fire]
    # Values are normalized (typically in [-1, 1], with Speed often clipped internally).
    predefined_actions = [
        [0,   0,   1, 0],   # all agents: level flight, max speed, no fire
        [0,   0.5, 1, 0],   # introduce some lateral command (turn) for alive agents
        [0,   0,   1, 0],
        [0,   0,   1, 0],
        [0,   0,   1, 0]
    ]

    # Index of which scripted action is currently active.
    a = 0

    # ---------------------------------------------------------------------
    # 5) Step loop: hold each action for a fixed number of steps
    # ---------------------------------------------------------------------
    # We hold each predefined action for 50 environment steps to produce visible motion.
    for step in range(len(predefined_actions) * 50):

        # Switch to the next scripted action every 50 steps.
        if step % 50 == 0 and step != 0:
            a += 1

        # Build an action dict only for agents that are currently alive.
        # This mirrors typical MARL control loops where dead agents stop acting.
        actions = {}
        for i, agent_id in enumerate(env.get_agent_ids()):
            if env.Aircrafts[i].is_alive():
                actions[agent_id] = predefined_actions[a]

        # Advance the simulation by one environment step.
        obs, rewards, terminated, truncated, infos = env.step(actions)

        # Capture a render frame after stepping for video export.
        images.append(env.render())

        # Debug print: termination flags per agent + "__all__" (if present).
        print("Dones:", terminated)

        # Stop early if the episode ended for all agents.
        if all(terminated.values()):
            print("All agents are done.")
            break

    # ---------------------------------------------------------------------
    # 6) Save video and analysis artifacts
    # ---------------------------------------------------------------------
    # Convert frames to numpy arrays and write an MP4.
    imageio.mimsave(
        "ENV_TEST/test_1.mp4",
        [np.array(img) for img in images],
        fps=10
    )

    # Export additional diagnostics for post-run inspection.
    env.render_trajectory("ENV_TEST")
    env.plot_telemetry("ENV_TEST")
    env.plot_rewards("ENV_TEST")

    # ---------------------------------------------------------------------
    # 7) Cleanup
    # ---------------------------------------------------------------------
    # Close the environment to release resources (rendering, file handles, etc.).
    env.close()


# Execute the test rollout when running this file directly (or in a notebook cell).
#Test_env()

