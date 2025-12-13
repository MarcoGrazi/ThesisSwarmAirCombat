import numpy as np


class FixedWingAircraft:
    """
    Fixed-wing aircraft 6DOF rigid-body simulator (Euler-angle attitude).

    High-level model:
      - State is stored as position (world frame), velocity (body frame),
        Euler orientation (world->body angles), and body angular rates.
      - At each step() we:
          1) Compute forces & moments in body frame (aero + thrust + gravity).
          2) Integrate translational dynamics in body coordinates.
          3) Transform body velocity to world frame to integrate position.
          4) Integrate rotational dynamics using rigid-body equations.
          5) Log telemetry.

    Frames (as implied by your code/comments):
      - World / vehicle frame: Z-down convention is assumed for gravity.
      - Body frame: [u, v, w] = [forward, lateral, vertical] (vertical consistent with Z-down).
      - Wind frame: aerodynamic frame aligned with incoming airflow for aero forces.

    Note:
      - This is an explicit-Euler integrator. It is simple and fast, but can become
        numerically unstable with large time steps or aggressive maneuvers.
    """

    def __init__(self, config, rho, g, frequency):
        """
        Initialize the aircraft model with environment and aircraft parameters.

        Parameters
        ----------
        config : dict
            Configuration dictionary loaded from a YAML file (UAV_config section).
            Contains mass/inertia, aero coefficient polynomials, geometry, thrust limits, etc.
        rho : float
            Air density [kg/m^3].
        g : float
            Gravitational acceleration [m/s^2].
        frequency : float
            Simulation update frequency [Hz]. The time step is dt = 1/frequency.

        Side effects
        ------------
        - Loads all aircraft parameters from config into instance fields.
        - Initializes state vectors and telemetry buffers.
        """

        # --- Environment constants ---
        # These are external parameters that affect aerodynamic forces (rho) and weight (g).
        self.rho = rho
        self.g = g

        # --- Aircraft physical properties ---
        # These define basic mass/inertia limits and simple kinematic safety constraints.
        self.m = config['mass']                               # Mass [kg]
        self.max_speed = config['max_speed']                  # Max allowable airspeed magnitude [m/s]
        self.max_acc = config['max_acc']                      # Max allowable acceleration "multiplier" (later used as max_acc*g)
        self.bounding_box = np.array(config['bounding_box'])  # Dimensions for collision/physics [m]

        # --- Inertia tensor ---
        # Defines resistance to rotation about principal axes (in body frame).
        inertia_tensor = np.array(config['inertia_tensor'], dtype=np.float64)
        self.I = inertia_tensor
        self.I_inv = np.linalg.inv(inertia_tensor)  # Precompute inverse for efficient torque->angular-acc conversion

        # --- Aerodynamic parameters (polynomial coefficient vectors) ---
        # Airframe coefficients:
        #   - CL vs AoA   (lift)
        #   - CDL vs AoA  (drag along wind)
        #   - CY vs beta  (side force)
        #   - CDY vs beta (side drag; in this code it is evaluated but not used further)
        self.a_CL_c = np.array(config['airframe']['coefficients']['front']['lift'])
        self.a_CDL_c = np.array(config['airframe']['coefficients']['front']['drag'])
        self.a_CY_c = np.array(config['airframe']['coefficients']['side']['lateral'])
        self.a_CDY_c = np.array(config['airframe']['coefficients']['side']['drag'])

        # Center of force (lever arm) and reference surface for the main airframe.
        self.a_COF = np.array(config['airframe']['COF'])   # Force application point relative to CG [m]
        self.a_surface = config['airframe']['surface']     # Reference surface area [m^2]

        # --- Ailerons (roll control) ---
        # Coefficients map control deflection to CL/CD contributions for each aileron side.
        self.al_CL_c = np.array(config['aleiron']['coefficients']['lift'])
        self.al_CD_c = np.array(config['aleiron']['coefficients']['drag'])
        self.al_COF = np.array(config['aleiron']['COF'])       # Lever arm for roll moment estimation [m]
        self.al_surface = config['aleiron']['surface']         # Surface area [m^2]

        # --- Elevons / elevators (pitch control) ---
        self.el_CL_c = np.array(config['elevons']['coefficients']['lift'])
        self.el_CD_c = np.array(config['elevons']['coefficients']['drag'])
        self.el_COF = np.array(config['elevons']['COF'])
        self.el_surface = config['elevons']['surface']

        # --- Rudders (yaw control) ---
        self.r_CY_c = np.array(config['rudders']['coefficients']['lateral'])
        self.r_CD_c = np.array(config['rudders']['coefficients']['drag'])
        self.r_COF = np.array(config['rudders']['COF'])
        self.r_surface = config['rudders']['surface']

        # --- Propulsion and braking ---
        self.max_thrust = config['max_thrust']              # Max thrust [N]
        self.aerobrake_CD = config['aerobrake_CD']          # Aerobrake drag coefficient multiplier
        self.aerobrake_surface = config['aerobrake_surface']# Aerobrake reference area [m^2]

        # --- Flight condition flags ---
        # Set during angle computations to indicate extreme aerodynamic angles.
        self.stall = False

        # --- Simulation time step ---
        # All integration uses a fixed dt.
        self.dt = 1 / frequency

        # =========================
        # State vectors (6DOF)
        # =========================

        # Position in world frame [x, y, z] [m]
        self.p = np.zeros(3)

        # Linear velocity in body frame [u, v, w] [m/s]
        self.v = np.zeros(3)

        # Orientation in Euler angles [roll, pitch, yaw] [rad]
        self.o = np.zeros(3)

        # Angular velocity (body rates) [p, q, r] [rad/s]
        self.w = np.zeros(3)

        # =========================
        # Telemetry buffers
        # =========================
        # Lists are appended each step to support plotting and debugging.
        self.telemetry = {
            'position': [],
            'orientation': [],
            'velocity': [],
            'angular_velocity': [],
            'acceleration': [],
            'AoA': [],             # Angle of Attack (alpha) history
            'sideslip': [],        # Sideslip angle (beta) history
            'force': [],           # Net body-frame force history
            'moment': [],          # Net body-frame moment history
            'commands': []         # Stored control inputs per step (action vector)
        }

    def reset(self, position, orientation, speed, config):
        """
        Reset the aircraft state (and reload configuration parameters).

        Parameters
        ----------
        position : array-like (3,)
            Initial position in world frame [m].
        orientation : array-like (3,)
            Initial Euler angles [roll, pitch, yaw] in radians.
        speed : float
            Initial forward speed in body frame along +X (u component) [m/s].
        config : dict
            Configuration dictionary; allows reloading aircraft parameters at reset time.

        Side effects
        ------------
        - Overwrites parameters (mass, inertia, aero coeffs, surfaces, etc.) from config.
        - Resets state (p, v, o, w) and clears telemetry.
        - Logs an initial (t=0) telemetry snapshot.
        """

        # =========================
        # Reload aircraft parameters
        # =========================
        # This mirrors __init__ so you can reset with a new aircraft model.

        self.m = config['mass']
        self.max_speed = config['max_speed']
        self.max_acc = config['max_acc']
        self.bounding_box = np.array(config['bounding_box'])

        inertia_tensor = np.array(config['inertia_tensor'], dtype=np.float64)
        self.I = inertia_tensor
        self.I_inv = np.linalg.inv(inertia_tensor)

        self.a_CL_c = np.array(config['airframe']['coefficients']['front']['lift'])
        self.a_CDL_c = np.array(config['airframe']['coefficients']['front']['drag'])
        self.a_CY_c = np.array(config['airframe']['coefficients']['side']['lateral'])
        self.a_CDY_c = np.array(config['airframe']['coefficients']['side']['drag'])
        self.a_COF = np.array(config['airframe']['COF'])
        self.a_surface = config['airframe']['surface']

        self.al_CL_c = np.array(config['aleiron']['coefficients']['lift'])
        self.al_CD_c = np.array(config['aleiron']['coefficients']['drag'])
        self.al_COF = np.array(config['aleiron']['COF'])
        self.al_surface = config['aleiron']['surface']

        self.el_CL_c = np.array(config['elevons']['coefficients']['lift'])
        self.el_CD_c = np.array(config['elevons']['coefficients']['drag'])
        self.el_COF = np.array(config['elevons']['COF'])
        self.el_surface = config['elevons']['surface']

        self.r_CY_c = np.array(config['rudders']['coefficients']['lateral'])
        self.r_CD_c = np.array(config['rudders']['coefficients']['drag'])
        self.r_COF = np.array(config['rudders']['COF'])
        self.r_surface = config['rudders']['surface']

        self.max_thrust = config['max_thrust']
        self.aerobrake_CD = config['aerobrake_CD']
        self.aerobrake_surface = config['aerobrake_surface']

        # =========================
        # Reset dynamic state
        # =========================

        # World position [m]
        self.p = position

        # Body-frame velocity [m/s]: start with forward velocity only.
        self.v = np.array([speed, 0.0, 0.0])

        # Euler angles [rad]
        self.o = orientation

        # Body rates [rad/s]
        self.w = np.array([0.0, 0.0, 0.0])

        # Reset stall indicator
        self.stall = False

        # =========================
        # Reset telemetry
        # =========================
        self.telemetry = {
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

        # Log baseline state (t=0)
        self.telemetry['position'].append(self.p.copy())
        self.telemetry['orientation'].append(self.o.copy())
        self.telemetry['velocity'].append(self.v.copy())
        self.telemetry['angular_velocity'].append(self.w.copy())

        # At reset we assume no computed acceleration/forces yet.
        self.telemetry['acceleration'].append([0, 0, 0])
        self.telemetry['AoA'].append(0)
        self.telemetry['sideslip'].append(0)
        self.telemetry['force'].append([0, 0, 0])
        self.telemetry['moment'].append([0, 0, 0])

        # Commands log placeholder for consistency with step()
        self.telemetry['commands'].append([0, 0, 0, 0])

    def dummy_step(self, dummy_type, turn_radius, direction, speed):
        """
        Advance the model using a simplified kinematic motion rule (non-physical).

        This function is useful when you want deterministic motion without forces:
          - "line": constant forward motion along current orientation
          - "curve": constant-speed turning in yaw with a given turn radius
          - "fixed": no motion (hold position)

        Parameters
        ----------
        dummy_type : str
            One of {'line', 'curve', 'fixed'}.
        turn_radius : float
            Turn radius [m] used only for 'curve'.
        direction : int
            Turn direction: +1 right, -1 left, 0 none.
        speed : float
            Forward speed [m/s].
        """

        if dummy_type == 'line':
            # Constant forward velocity in body coordinates.
            self.v = np.array([speed, 0.0, 0.0])

            # Convert body velocity to world frame and integrate position.
            R = self.body_to_vehicle(self.o[0], self.o[1], self.o[2])
            self.p = self.p.copy() + (R @ self.v) * self.dt

        elif dummy_type == 'curve':
            # Flat turn assumption: heading changes at rate v/R.
            turn_rate = speed / turn_radius

            # Update yaw (heading) only; roll/pitch remain unchanged.
            self.o[2] += direction * turn_rate * self.dt

            # Forward velocity remains constant in body frame.
            self.v = np.array([speed, 0.0, 0.0])

            # Integrate position in world frame.
            R = self.body_to_vehicle(self.o[0], self.o[1], self.o[2])
            self.p += (R @ self.v) * self.dt

            # Body rates: only yaw rate is non-zero in this simplified motion.
            self.w = np.array([0.0, 0.0, direction * turn_rate])

        elif dummy_type == 'fixed':
            # Explicit "do nothing" case. Copy keeps semantics consistent with other branches.
            self.p = self.p.copy()

        # =========================
        # Telemetry logging (dummy mode)
        # =========================
        # We log zeros for force/moment/acceleration since no physics are computed here.
        self.telemetry['position'].append(self.p.copy())
        self.telemetry['orientation'].append(self.o.copy())
        self.telemetry['velocity'].append(self.v.copy())
        self.telemetry['angular_velocity'].append(self.w.copy())

        self.telemetry['acceleration'].append(np.zeros(3))
        self.telemetry['AoA'].append(0)
        self.telemetry['sideslip'].append(0)
        self.telemetry['force'].append(np.zeros(3))
        self.telemetry['moment'].append(np.zeros(3))
        self.telemetry['commands'].append(np.zeros(4))

    def step(self, throttle, elevon_angle, aleiron_angle, rudder_angle, action):
        """
        Advance the aircraft by one physics time step using rigid-body 6DOF equations.

        Inputs are interpreted as "commands" that affect aerodynamic forces and thrust.

        Parameters
        ----------
        throttle : float
            Engine throttle command (scalar). In compute_forces_moment you clamp thrust to be non-negative.
        elevon_angle : float
            Elevon deflection [deg].
        aleiron_angle : float
            Aileron deflection [deg].
        rudder_angle : float
            Rudder deflection [deg].
        action : array-like
            Full command vector saved into telemetry (e.g. RL action). Only logged here.

        Notes
        -----
        - Translational integration is done in body coordinates, which requires adding ω x v
          (the "rotating frame" term) to get correct body-frame acceleration.
        - Position is updated in world coordinates using the body->world rotation matrix.
        - Rotational dynamics use Euler's rigid-body equation with inertia tensor I.
        """

        # Compute net force/moment in BODY frame, and aerodynamic angles for logging.
        force_body, moment_body, AoA, sideslip = self.compute_forces_moment(
            throttle, elevon_angle, aleiron_angle, rudder_angle
        )

        # =========================
        # Translational dynamics
        # =========================
        # When expressing velocity in the rotating body frame:
        #   v_dot_body = (F/m) - ω×v   (depending on sign convention).
        # Here you're effectively adding ω×v as "rot_a"; consistent with your derived equations.
        rot_a = np.array([
            self.w[2] * self.v[1] - self.w[1] * self.v[2],
            self.w[0] * self.v[2] - self.w[2] * self.v[0],
            self.w[1] * self.v[0] - self.w[0] * self.v[1]
        ])

        # Net body acceleration [m/s^2]
        a = rot_a + force_body / self.m

        # Safety cap on acceleration magnitude (helps stability / limits extreme dynamics)
        acc = np.linalg.norm(a)
        if acc > self.max_acc * self.g:
            a = (a / acc) * (self.max_acc * self.g)

        # Integrate body velocity (explicit Euler)
        self.v += a * self.dt

        # Cap overall speed magnitude
        speed = np.linalg.norm(self.v)
        speed = np.where(speed < 1e-3, 0.0, speed)
        if speed > self.max_speed:
            self.v = (self.v / speed) * self.max_speed

        # =========================
        # Position update (world frame)
        # =========================
        # Convert body velocity to world velocity and integrate.
        R = self.body_to_vehicle(self.o[0], self.o[1], self.o[2])
        self.p += (R @ self.v) * self.dt

        # =========================
        # Rotational dynamics
        # =========================
        # Euler rigid-body equation:
        #   ω_dot = I^{-1} ( M - ω×(Iω) )
        wa = self.I_inv @ (moment_body - np.cross(self.w, np.dot(self.I, self.w)))

        # Optional cap on angular acceleration to prevent numerical blow-up.
        wa_norm = np.linalg.norm(wa)
        if wa_norm > 1000:
            wa = (wa / wa_norm) * 1000

        # =========================
        # Euler angle kinematics
        # =========================
        # Convert body rates [p,q,r] into Euler angle rates [roll_dot, pitch_dot, yaw_dot].
        att = np.array([
            [1, np.sin(self.o[0]) * np.tan(self.o[1]), np.cos(self.o[0]) * np.tan(self.o[1])],
            [0, np.cos(self.o[0]), -np.sin(self.o[0])],
            [0, np.sin(self.o[0]) / np.cos(self.o[1]), np.cos(self.o[0]) / np.cos(self.o[1])]
        ])

        # Integrate attitude and body rates (explicit Euler)
        self.o += (att @ self.w) * self.dt
        self.w += wa * self.dt

        # Normalize angles to [-pi, pi] for numerical hygiene and easier downstream usage.
        self.o[0] = (self.o[0] + np.pi) % (2 * np.pi) - np.pi
        self.o[1] = (self.o[1] + np.pi) % (2 * np.pi) - np.pi
        self.o[2] = (self.o[2] + np.pi) % (2 * np.pi) - np.pi

        # =========================
        # Telemetry logging
        # =========================
        self.telemetry['position'].append(self.p.copy())
        self.telemetry['orientation'].append(self.o.copy())
        self.telemetry['velocity'].append(self.v.copy())
        self.telemetry['angular_velocity'].append(self.w.copy())
        self.telemetry['acceleration'].append(np.array(a))
        self.telemetry['AoA'].append(AoA)
        self.telemetry['sideslip'].append(sideslip)
        self.telemetry['force'].append(np.array(force_body))
        self.telemetry['moment'].append(np.array(moment_body))
        self.telemetry['commands'].append(np.array(action))

    def compute_forces_moment(self, throttle, elevon_angle, aleiron_angle, rudder_angle):
        """
        Compute total external force and moment acting on the aircraft in BODY frame.

        Components included:
          - Aerodynamic forces and moments (airframe + control surfaces + aerobrake)
          - Weight projected into body axes
          - Thrust along body +X

        Parameters
        ----------
        throttle : float
            Throttle command. Negative throttle is interpreted here as aerobrake deployment
            (via aerobrake_deploy) while thrust is clamped to >= 0.
        elevon_angle : float
            Elevon deflection [deg].
        aleiron_angle : float
            Aileron deflection [deg].
        rudder_angle : float
            Rudder deflection [deg].

        Returns
        -------
        force : ndarray (3,)
            Net force vector in body frame [N].
        moment : ndarray (3,)
            Net moment vector in body frame [N*m].
        AoA : float
            Angle of attack returned from aero calc (units as produced there; you use rad internally).
        sideslip : float
            Sideslip returned from aero calc (same).
        """

        # Aerobrake deployment is derived from negative throttle magnitude.
        aerobrake_deploy = abs(min(throttle, 0))

        # Aerodynamic forces and moments
        F_aero, M_aero, AoA, sideslip = self.compute_aero_forces_and_moment(
            elevon_angle, aleiron_angle, rudder_angle, aerobrake_deploy
        )

        # Weight in world frame under Z-down convention is +mg along +Z_world.
        # Project to body frame using world->body rotation.
        F_weight = self.vehicle_to_body(self.o[0], self.o[1], self.o[2]) @ np.array([0.0, 0.0, self.m * self.g])

        # Thrust acts along +X_body. Clamped so negative throttle does not create reverse thrust here.
        F_thrust = np.array([max(self.max_thrust * throttle, 0), 0.0, 0.0])

        # Sum of all external forces and moments (all expressed in body frame).
        force = F_thrust + F_weight + F_aero
        moment = M_aero

        return force, moment, AoA, sideslip

    def compute_aero_forces_and_moment(self, elevon_angle, aleiron_angle, rudder_angle, aerobrake_deploy):
        """
        Compute aerodynamic forces and moments based on current velocity and control deflections.

        This function:
          1) Derives aerodynamic angles (AoA, sideslip) from body velocity.
          2) Evaluates airframe/control-surface force models in wind frame.
          3) Rotates forces wind -> body.
          4) Computes aerodynamic moments.

        Parameters
        ----------
        elevon_angle : float
            Elevon deflection [deg].
        aleiron_angle : float
            Aileron deflection [deg].
        rudder_angle : float
            Rudder deflection [deg].
        aerobrake_deploy : float
            Aerobrake deployment factor (0..1).

        Returns
        -------
        F_aero : ndarray (3,)
            Total aerodynamic force in body frame [N].
        M_aero : ndarray (3,)
            Total aerodynamic moment in body frame [N*m].
        AoA : float
            Angle of attack [rad].
        sideslip : float
            Sideslip [rad].
        """

        # Airspeed magnitude; guard small values to avoid division by zero.
        V = max(np.linalg.norm(self.v), 1e-3)

        # Body velocity components
        u, v, w = self.v

        # Aerodynamic angles derived from velocity direction in body frame.
        AoA = np.arctan2(w, u)
        sideslip = np.arcsin(v / V)

        # Flag extreme angles and clamp to avoid invalid trig / rotations.
        if (AoA > np.deg2rad(89) or AoA < -np.deg2rad(89) or
            sideslip > np.deg2rad(89) or sideslip < -np.deg2rad(89)):
            AoA = np.clip(AoA, -np.deg2rad(89), np.deg2rad(89))
            sideslip = np.clip(sideslip, -np.deg2rad(89), np.deg2rad(89))
            self.stall = True
        else:
            self.stall = False

        # --- Aerodynamic forces in wind frame ---
        # Each component model returns a force vector aligned with the wind coordinate system.
        f_airframe_wind = self.Airframe(np.rad2deg(AoA), np.rad2deg(sideslip), V)
        f_aleirons_wind, roll_moment_aleirons = self.Ailerons(aleiron_angle, V)
        f_elevons_wind = self.Elevators(elevon_angle, V)
        f_rudders_wind = self.Rudders(rudder_angle, V)

        # Wind->body rotation matrix for converting aerodynamic forces.
        R_w2b = self.wind_to_body(AoA, sideslip)

        # Convert aerodynamic forces into body frame so they can be summed with thrust/weight.
        f_airframe_body = R_w2b @ f_airframe_wind
        f_elevons_body = R_w2b @ f_elevons_wind
        f_aleirons_body = R_w2b @ f_aleirons_wind
        f_rudders_body = R_w2b @ f_rudders_wind

        # Aerobrake force is modeled directly as a drag vector (as returned by Aerobrake()).
        f_aerobrake = self.Aerobrake(V, aerobrake_deploy)

        # Total aerodynamic force in body axes.
        F_aero = f_airframe_body + f_elevons_body + f_aleirons_body + f_rudders_body + f_aerobrake

        # --- Aerodynamic moments ---
        # Compute moments about CG from force application points + any directly modeled moments.
        # (This keeps the “torque = r × F” approach consistent across surfaces.)
        M_airframe = np.cross(self.a_COF, f_airframe_wind)
        M_aleirons = roll_moment_aleirons
        M_elevons = np.cross(self.el_COF, f_elevons_wind)
        M_rudders = np.cross(self.r_COF, f_rudders_wind)

        M_aero = M_airframe + M_aleirons + M_elevons + M_rudders

        return F_aero, M_aero, AoA, sideslip

    def Airframe(self, AoA, sideslip, V):
        """
        Airframe aerodynamic force model in wind frame.

        Parameters
        ----------
        AoA : float
            Angle of attack [deg] used to evaluate lift and drag coefficient polynomials.
        sideslip : float
            Sideslip angle [deg] used to evaluate lateral force coefficients.
        V : float
            Airspeed magnitude [m/s].

        Returns
        -------
        ndarray (3,)
            Wind-frame aerodynamic force [N]:
              X_w: drag (negative, opposes motion)
              Y_w: lateral (negative by convention used here)
              Z_w: lift (negative so that positive lift corresponds to -Z_w here)

        Implementation notes
        --------------------
        - Coefficients are evaluated with np.polyval(), so the arrays are expected
          in decreasing polynomial order.
        - Dynamic pressure q = 0.5 * rho * V^2.
        """

        CL = np.polyval(self.a_CL_c, np.clip(AoA, -60, 60))
        CDL = np.polyval(self.a_CDL_c, AoA)
        CDY = np.polyval(self.a_CDY_c, sideslip)  # currently computed but not used further
        CY  = np.polyval(self.a_CY_c, np.clip(sideslip, -60, 60))

        q = 0.5 * self.rho * V**2

        lift = CL * q * self.a_surface
        drag = CDL * q * self.a_surface
        lateral = CY * q * self.a_surface

        return np.array([-drag, -lateral, -lift])

    def Ailerons(self, angle, V):
        """
        Aileron aerodynamic force (wind frame) and roll moment (body frame).

        The aileron model assumes two symmetric surfaces with opposite deflection.
        It computes:
          - Total lift/drag contribution from both surfaces
          - A roll moment proportional to lift and lateral lever arm (y-offset)

        Parameters
        ----------
        angle : float
            Aileron deflection [deg]. Sign convention described in the comment.
        V : float
            Airspeed magnitude [m/s].

        Returns
        -------
        force : ndarray (3,)
            Total force in wind frame [N].
        roll_moment : ndarray (3,)
            Roll moment vector [N*m] (Mx, My, Mz). Only Mx is non-zero here.
        """

        # Evaluate polynomial fits for CL and CD. Scaling "angle * 40" matches your coefficient domain.
        CL_1 = np.polyval(self.al_CL_c, angle)
        CD_1 = np.polyval(self.al_CD_c, angle * 40)

        CL_2 = np.polyval(self.al_CL_c, -angle)
        CD_2 = np.polyval(self.al_CD_c, -angle * 40)

        q = 0.5 * self.rho * V**2

        lift_1 = CL_1 * q * self.al_surface
        lift_2 = CL_2 * q * self.al_surface
        drag = (CD_1 + CD_2) * q * self.al_surface

        force = np.array([
            -drag,
            0.0,                 # no lateral force modeled for ailerons
            -(lift_1 + lift_2)
        ])

        # Roll moment: lift times lateral lever arm (uses y-component of COF).
        roll_moment = 2 * (lift_1 * self.al_COF[1])
        return force, np.array([roll_moment, 0, 0])

    def Elevators(self, angle, V):
        """
        Elevator / elevon aerodynamic force model in wind frame.

        Parameters
        ----------
        angle : float
            Elevator deflection [deg].
        V : float
            Airspeed magnitude [m/s].

        Returns
        -------
        ndarray (3,)
            Wind-frame force [N] with drag in -X_w and lift in +Z_w (per your convention).
        """

        CL = np.polyval(self.el_CL_c, -angle)
        CD = np.polyval(self.el_CD_c, -angle * 40)

        q = 0.5 * self.rho * V**2

        # Two elevator surfaces are assumed; hence multiply by 2.
        lift = CL * q * self.el_surface * 2
        drag = CD * q * self.el_surface * 2

        return np.array([-drag, 0.0, lift])

    def Rudders(self, angle, V):
        """
        Rudder aerodynamic force model in wind frame.

        Parameters
        ----------
        angle : float
            Rudder deflection [deg].
        V : float
            Airspeed magnitude [m/s].

        Returns
        -------
        ndarray (3,)
            Wind-frame force [N] with:
              - drag in -X_w
              - side force in -Y_w (per your sign convention)
              - no lift (Z component is zero)
        """

        CY = np.polyval(self.r_CY_c, angle)
        CD = np.polyval(self.r_CD_c, angle * 40)

        q = 0.5 * self.rho * V**2

        drag = CD * q * self.r_surface * 2
        lateral = CY * q * self.r_surface * 2

        return np.array([-drag, -lateral, 0.0])

    def Aerobrake(self, V, deploy):
        """
        Aerobrake drag model.

        Parameters
        ----------
        V : float
            Airspeed magnitude [m/s]
        deploy : float
            Deployment factor (typically 0..1)

        Returns
        -------
        ndarray (3,)
            Drag force vector [N]. This model produces drag along the local X axis only.
        """

        q = 0.5 * self.rho * V**2
        CD = self.aerobrake_CD * deploy
        drag = CD * q * self.aerobrake_surface
        return np.array([-drag, 0, 0])

    def body_to_vehicle(self, roll, pitch, yaw):
        """
        Body -> world rotation matrix.

        This is computed as the transpose of vehicle_to_body because rotation
        matrices are orthonormal (inverse equals transpose).

        Returns
        -------
        ndarray (3,3)
            Rotation matrix mapping body-frame vectors into world frame.
        """
        return self.vehicle_to_body(roll, pitch, yaw).T

    def vehicle_to_body(self, roll, pitch, yaw):
        """
        World -> body rotation matrix using ZYX Euler convention (yaw, pitch, roll).

        Parameters
        ----------
        roll : float
            Roll angle [rad]
        pitch : float
            Pitch angle [rad]
        yaw : float
            Yaw angle [rad]

        Returns
        -------
        ndarray (3,3)
            Rotation matrix mapping world-frame vectors into body frame.
        """

        cr = np.cos(roll);  sr = np.sin(roll)
        cp = np.cos(pitch); sp = np.sin(pitch)
        cy = np.cos(yaw);   sy = np.sin(yaw)

        R = np.array([
            [cp * cy, cp * sy, -sp],
            [sy * sp * cy - cr * sy, sr * sp * sy + cr * cy, sr * cp],
            [cr * sp * cy + sr * sy, cy * sp * sy - sr * cy, cr * cp]
        ])
        return R

    def body_to_wind(self, AoA, sideslip):
        """
        Body -> wind rotation matrix.

        The wind frame is aligned with airflow direction:
          - X_w aligned with relative wind direction
          - Y_w lateral axis
          - Z_w completes right-handed triad

        Parameters
        ----------
        AoA : float
            Angle of attack [rad]
        sideslip : float
            Sideslip angle [rad]

        Returns
        -------
        ndarray (3,3)
            Rotation matrix mapping body-frame vectors into wind frame.
        """

        ca = np.cos(AoA);      sa = np.sin(AoA)
        cb = np.cos(sideslip); sb = np.sin(sideslip)

        R = np.array([
            [cb * ca, sb, cb * sa],
            [-sb * ca, cb, -sb * sa],
            [-sa,     0,   ca]
        ])
        return R

    def wind_to_body(self, AoA, sideslip):
        """
        Wind -> body rotation matrix.

        Because body_to_wind is a pure rotation, the inverse is the transpose.

        Returns
        -------
        ndarray (3,3)
            Rotation matrix mapping wind-frame vectors into body frame.
        """
        return self.body_to_wind(AoA, sideslip).T

    def getTelemetry(self):
        """
        Return collected telemetry.

        Returns
        -------
        dict
            Dictionary of lists: each key corresponds to a time history array.
        """
        return self.telemetry

    def get_absolute_velocity(self):
        """
        Convert current body-frame velocity into world-frame velocity.

        Returns
        -------
        ndarray (3,)
            Velocity vector in world frame [m/s].
        """
        return self.body_to_vehicle(self.o[0], self.o[1], self.o[2]) @ self.v

    def get_pos(self):
        """
        Current world position.

        Returns
        -------
        ndarray (3,)
            Position in world frame [m].
        """
        return self.p

