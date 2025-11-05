import numpy as np

class FixedWingAircraft:
    def __init__(self, config, rho, g, frequency):
        """
        Initialize the 6DOF aircraft physics model.
        
        :param config: Configuration dictionary loaded from UAV_config section of a YAML file.
        :param rho: Air density in kg/m^3 (environmental parameter).
        :param g: Gravitational acceleration (typically 9.81 m/s^2).
        :param frequency: Simulation update frequency in Hz.
        """

        # Environmental and physical constants
        self.rho = rho
        self.g = g

        # Aircraft physical properties
        self.m = config['mass']                            # Aircraft mass [kg]
        self.max_speed = config['max_speed']              # Maximum allowable airspeed [m/s]
        self.max_acc = config['max_acc']                  # Maximum allowable acceleration [m/s^2]
        self.bounding_box = np.array(config['bounding_box'])  # Aircraft bounding dimensions for collision/physics

        # Inertia tensor and its inverse for rotational dynamics
        inertia_tensor = np.array(config['inertia_tensor'], dtype=np.float64)
        self.I = inertia_tensor
        self.I_inv = np.linalg.inv(inertia_tensor)         # Pre-compute inverse for torque calculations

        # === Aerodynamic Coefficients (from config) ===
        # Front section (main wing/body)
        self.a_CL_c = np.array(config['airframe']['coefficients']['front']['lift'])     # Lift coeffs
        self.a_CDL_c = np.array(config['airframe']['coefficients']['front']['drag'])    # Drag coeffs
        self.a_CY_c = np.array(config['airframe']['coefficients']['side']['lateral'])   # Side force coeffs
        self.a_CDY_c = np.array(config['airframe']['coefficients']['side']['drag'])     # Side drag coeffs
        self.a_COF = np.array(config['airframe']['COF'])                                 # Center of force [x, y, z]
        self.a_surface = config['airframe']['surface']                                   # Surface area [m²]

        # Ailerons (roll control surfaces)
        self.al_CL_c = np.array(config['aleiron']['coefficients']['lift'])
        self.al_CD_c = np.array(config['aleiron']['coefficients']['drag'])
        self.al_COF = np.array(config['aleiron']['COF'])
        self.al_surface = config['aleiron']['surface']

        # Elevons (pitch control surfaces)
        self.el_CL_c = np.array(config['elevons']['coefficients']['lift'])
        self.el_CD_c = np.array(config['elevons']['coefficients']['drag'])
        self.el_COF = np.array(config['elevons']['COF'])
        self.el_surface = config['elevons']['surface']

        # Rudders (yaw control surfaces)
        self.r_CY_c = np.array(config['rudders']['coefficients']['lateral'])
        self.r_CD_c = np.array(config['rudders']['coefficients']['drag'])
        self.r_COF = np.array(config['rudders']['COF'])
        self.r_surface = config['rudders']['surface']

        # Maximum available engine thrust [N]
        self.max_thrust = config['max_thrust']
        self.aerobrake_CD = config['aerobrake_CD']
        self.aerobrake_surface = config['aerobrake_surface']

        # Stall flag (to be triggered during AoA evaluation)
        self.stall = False

        # Simulation timestep (dt = 1/frequency)
        self.dt = 1 / frequency

        # === State Vectors ===
        # Position in world frame [x, y, z]
        self.p = np.zeros(3)

        # Linear velocity in body frame [u, v, w]
        self.v = np.zeros(3)

        # Orientation (Euler angles) [roll, pitch, yaw]
        self.o = np.zeros(3)

        # Angular velocity in body frame [p, q, r]
        self.w = np.zeros(3)

        # === Telemetry Logging ===
        # Used for post-simulation analysis or visualization
        self.telemetry = {
            'position': [],
            'orientation': [],
            'velocity': [],
            'angular_velocity': [],
            'acceleration': [],
            'AoA': [],             # Angle of Attack over time
            'sideslip': [],        # Sideslip angle over time
            'force': [],
            'moment': [],
            'commands': []         # Control input history
        }

    def reset(self, position, orientation, speed):
        """
        Reset the aircraft's dynamic state to initial conditions.

        :param position: Initial position in world frame (3D vector)
        :param orientation: Initial Euler angles (roll, pitch, yaw) in radians (3D vector)
        :param speed: Initial forward speed in body frame along X-axis (scalar)
        """

        # === Reset dynamic state variables ===

        self.p = position                     # Position in world frame [x, y, z]
        self.v = np.array([speed, 0.0, 0.0])  # Initial velocity in body frame [u, v, w]
        self.o = orientation                  # Orientation (Euler angles) [roll, pitch, yaw]
        self.w = np.array([0.0, 0.0, 0.0])    # Angular velocity in body frame [p, q, r]

        # Reset stall condition
        self.stall = False

        # === Reset telemetry log ===
        # Clears and initializes the log used to track state evolution during the simulation

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

        # === Log initial values ===
        # These provide a baseline (t=0) state for the simulation

        self.telemetry['position'].append(self.p.copy())
        self.telemetry['orientation'].append(self.o.copy())
        self.telemetry['velocity'].append(self.v.copy())
        self.telemetry['angular_velocity'].append(self.w.copy())
        self.telemetry['acceleration'].append([0, 0, 0])    # Assuming rest + gravity in Z by default
        self.telemetry['AoA'].append(0)                       # Angle of Attack (deg or rad)
        self.telemetry['sideslip'].append(0)                  # Sideslip angle (deg or rad)
        self.telemetry['force'].append([0, 0, 0])             # Net aerodynamic + thrust forces
        self.telemetry['moment'].append([0, 0, 0])            # Net torques/moments
        self.telemetry['commands'].append([0, 0, 0, 0])       # Control inputs: [UpAngle, SideAngle, Speed, Fire]

    def dummy_step(self, dummy_type, turn_radius, direction, speed):
        """
        Advance the aircraft one timestep using a simplified (non-physical) motion model.

        :param dummy_type: Type of trajectory ['line', 'curve', 'fixed']
        :param turn_radius: Radius of turn (used only for 'curve' type)
        :param direction: Direction of turn (+1 for right, -1 for left, 0 for no turn)
        """

        if dummy_type == 'line':
            # === Straight line motion ===
            # Moves forward in the direction of current orientation at constant velocity
            self.v = np.array([speed, 0.0, 0.0])
            R = self.body_to_vehicle(self.o[0], self.o[1], self.o[2])  # Rotation matrix: body → world
            self.p = self.p.copy() + (R @ self.v) * self.dt            # Update position using v_b transformed to world frame

        elif dummy_type == 'curve':
            # === Turning motion ===
            # Simplified 2D yaw turn, constant altitude and speed

            turn_rate = speed / turn_radius              # Yaw rate [rad/s] from speed and radius

            self.o[2] += direction * turn_rate * self.dt # Update yaw angle (only heading changes)

            self.v = np.array([speed, 0.0, 0.0])          # Velocity remains forward in body frame

            R = self.body_to_vehicle(self.o[0], self.o[1], self.o[2])  # Update orientation matrix
            self.p += (R @ self.v) * self.dt             # Update position in world frame

            self.w = np.array([0.0, 0.0, direction * turn_rate])  # Simulated angular velocity (yaw only)

        elif dummy_type == 'fixed':
            # === Static case ===
            # No movement at all, useful for testing or holding position

            self.p = self.p.copy()  # Ensure immutability

        # === Log dummy step telemetry ===

        self.telemetry['position'].append(self.p.copy())
        self.telemetry['orientation'].append(self.o.copy())
        self.telemetry['velocity'].append(self.v.copy())
        self.telemetry['angular_velocity'].append(self.w.copy())

        self.telemetry['acceleration'].append(np.zeros(3))      # No acceleration in dummy motion
        self.telemetry['AoA'].append(0)                          # Angle of attack = 0 by assumption
        self.telemetry['sideslip'].append(0)                     # No sideslip modeled
        self.telemetry['force'].append(np.zeros(3))              # No forces computed
        self.telemetry['moment'].append(np.zeros(3))             # No torques computed
        self.telemetry['commands'].append(np.zeros(4))           # No control input simulated

    def step(self, throttle, elevon_angle, aleiron_angle, rudder_angle, action):
        """
        Perform a single physics timestep update using the 6DOF rigid-body equations.

        :param throttle: Engine throttle command (scalar)
        :param elevon_angle: Elevon deflection [degrees]
        :param aleiron_angle: Aileron deflection [degrees]
        :param rudder_angle: Rudder deflection [degrees]
        :param action: Full control action array [throttle, aileron, elevon, rudder, ...]
        """

        # === Compute total force and moment in body frame ===
        # Includes aerodynamic, control surface, and engine effects
        force_body, moment_body, AoA, sideslip = self.compute_forces_moment(
            throttle, elevon_angle, aleiron_angle, rudder_angle
        )

        # === Translational Acceleration (Newton's 2nd Law) ===
        # a = (F/m) + ω × v (in body frame)
        rot_a = np.array([
            self.w[2]*self.v[1] - self.w[1]*self.v[2],
            self.w[0]*self.v[2] - self.w[2]*self.v[0],
            self.w[1]*self.v[0] - self.w[0]*self.v[1]
        ])  # Coriolis acceleration

        a = rot_a + force_body / self.m  # Total acceleration in body frame

        # === Cap excessive linear acceleration ===
        acc = np.linalg.norm(a)
        if acc > self.max_acc * self.g:
            a = (a / acc) * (self.max_acc * self.g)
        
        self.v += a * self.dt  # Integrate linear velocity in body frame

        # === Cap maximum forward speed ===
        speed = np.linalg.norm(self.v)
        speed = np.where(speed < 1e-3, 0.0, speed)
        if speed > self.max_speed:
            self.v = (self.v / speed) * self.max_speed

        # === Update position in world frame ===
        # Use rotation matrix to convert body velocity to inertial frame
        R = self.body_to_vehicle(self.o[0], self.o[1], self.o[2])
        self.p += (R @ self.v) * self.dt

        # === Rotational Dynamics (Euler's Equation) ===
        # Angular acceleration: I⁻¹ * (M - ω × (Iω))
        wa = self.I_inv @ (moment_body - np.cross(self.w, np.dot(self.I, self.w)))

        # Optional cap to prevent instability (very large angular acceleration)
        wa_norm = np.linalg.norm(wa)
        if wa_norm > 1000:
            wa = (wa / wa_norm) * 1000

        # === Angular Integration (Euler angle rates from body rates) ===
        # Transformation matrix from body angular rates to Euler angle rates
        att = np.array([
            [1, np.sin(self.o[0]) * np.tan(self.o[1]), np.cos(self.o[0]) * np.tan(self.o[1])],
            [0, np.cos(self.o[0]), -np.sin(self.o[0])],
            [0, np.sin(self.o[0]) / np.cos(self.o[1]), np.cos(self.o[0]) / np.cos(self.o[1])]
        ])

        self.o += (att @ self.w) * self.dt  # Integrate orientation
        self.w += wa * self.dt              # Integrate angular velocity

        # === Normalize Euler angles to [-π, π] ===
        self.o[0] = (self.o[0] + np.pi) % (2 * np.pi) - np.pi  # Roll
        self.o[1] = (self.o[1] + np.pi) % (2 * np.pi) - np.pi  # Pitch
        self.o[2] = (self.o[2] + np.pi) % (2 * np.pi) - np.pi  # Yaw

        # === Log Telemetry ===
        self.telemetry['position'].append(self.p.copy())
        self.telemetry['orientation'].append(self.o.copy())
        self.telemetry['velocity'].append(self.v.copy())
        self.telemetry['angular_velocity'].append(self.w.copy())
        self.telemetry['acceleration'].append(np.array(a))
        self.telemetry['AoA'].append(AoA)
        self.telemetry['sideslip'].append(sideslip)
        self.telemetry['force'].append(np.array(force_body))
        self.telemetry['moment'].append(np.array(moment_body))
        self.telemetry['commands'].append(np.array(action))  # Last element of action is fire command, unused in physics

    def compute_forces_moment(self, throttle, elevon_angle, aleiron_angle, rudder_angle):
        """
        Compute the total external force and moment acting on the aircraft in body frame.

        Includes contributions from:
        - Aerodynamic forces and moments (control surfaces)
        - Gravity (transformed into body frame)
        - Engine thrust

        :param throttle: Throttle input (0 to 1), scalar
        :param elevon_angle: Elevon deflection in degrees
        :param aleiron_angle: Aileron deflection in degrees
        :param rudder_angle: Rudder deflection in degrees

        :return:
            force:     Net external force in body frame [N]
            moment:    Net external moment (torque) in body frame [Nm]
            AoA:       Angle of attack [rad or deg] (for telemetry)
            sideslip:  Sideslip angle [rad or deg] (for telemetry)
        """
        aerobrake_deploy = abs(min(throttle, 0))

        # === Aerodynamic forces and moments from control surfaces ===
        F_aero, M_aero, AoA, sideslip = self.compute_aero_forces_and_moment(
            elevon_angle, aleiron_angle, rudder_angle, aerobrake_deploy
        )

        # === Gravitational force in body frame ===
        # Gravity vector in world frame: [0, 0, +mg] with Z-down convention
        # Convert to body frame by applying inverse rotation
        F_weight = self.vehicle_to_body(self.o[0], self.o[1], self.o[2]) @ np.array([0.0, 0.0, self.m * self.g])

        # === Engine thrust force ===
        # Acts along the positive X-axis of the body frame
        F_thrust = np.array([self.max_thrust * throttle, 0.0, 0.0])

        # === Total external force and moment in body frame ===
        force = F_thrust + F_weight + F_aero
        moment = M_aero  # No propulsion-induced moment modeled here

        return force, moment, AoA, sideslip

    def compute_aero_forces_and_moment(self, elevon_angle, aleiron_angle, rudder_angle, aerobrake_deploy):
        """
        Compute aerodynamic forces and moments in the body frame based on current flight condition
        and control surface deflections.

        :param elevon_angle: Elevon deflection [degrees]
        :param aleiron_angle: Aileron deflection [degrees]
        :param rudder_angle: Rudder deflection [degrees]

        :return:
            F_aero:    Total aerodynamic force in body frame [N]
            M_aero:    Total aerodynamic moment in body frame [Nm]
            AoA:       Angle of Attack [rad]
            sideslip:  Sideslip angle [rad]
        """

        # === Body-frame airspeed magnitude ===
        V = max(np.linalg.norm(self.v), 1e-3)  # Prevent division by zero

        # Decompose velocity into components
        u, v, w = self.v  # u = forward, v = lateral, w = vertical in body frame

        # === Compute aerodynamic angles ===
        AoA = np.arctan2(w, u)        # Angle of Attack (alpha)
        sideslip = np.arcsin(v / V)   # Sideslip angle (beta)

        # === Stall check and clipping ===
        # Prevent extreme angles which may destabilize computation
        if (AoA > np.deg2rad(89) or AoA < -np.deg2rad(89) or
            sideslip > np.deg2rad(89) or sideslip < -np.deg2rad(89)):
            AoA = np.clip(AoA, -np.deg2rad(89), np.deg2rad(89))
            sideslip = np.clip(sideslip, -np.deg2rad(89), np.deg2rad(89))
            self.stall = True
        else:
            self.stall = False

        # === Aerodynamic forces (wind frame) ===
        # These functions return force vectors (in wind frame) and any associated moments
        f_airframe_wind = self.Airframe(np.rad2deg(AoA), np.rad2deg(sideslip), V)
        f_aleirons_wind, roll_moment_aleirons = self.Ailerons(aleiron_angle, V)
        f_elevons_wind = self.Elevators(elevon_angle, V)
        f_rudders_wind = self.Rudders(rudder_angle, V)

        # === Rotate aerodynamic forces into body frame ===
        # Each surface's force is in wind frame → transform it to body frame
        R_w2b = self.wind_to_body(AoA, sideslip)
        f_airframe_body = R_w2b @ f_airframe_wind
        f_elevons_body = R_w2b @ f_elevons_wind
        f_aleirons_body = R_w2b @ f_aleirons_wind
        f_rudders_body = R_w2b @ f_rudders_wind

        f_aerobrake = self.Aerobrake(aerobrake_deploy, V)

        # === Total aerodynamic force (body frame) ===
        F_aero = f_airframe_body + f_elevons_body + f_aleirons_body + f_rudders_body + f_aerobrake

        # === Aerodynamic moments (body frame) ===
        # Moments from aerodynamic force offset from CG
        M_airframe = np.cross(self.a_COF, f_airframe_wind)
        M_aleirons = roll_moment_aleirons  # Direct moment from aileron differential lift
        M_elevons = np.cross(self.el_COF, f_elevons_wind)
        M_rudders = np.cross(self.r_COF, f_rudders_wind)

        # Total moment
        M_aero = M_airframe + M_aleirons + M_elevons + M_rudders

        return F_aero, M_aero, AoA, sideslip

    def Airframe(self, AoA, sideslip, V):
        """
        Compute aerodynamic forces generated by the main airframe (fuselage + wing surfaces),
        expressed in the wind frame.

        :param AoA: Angle of Attack [degrees]
        :param sideslip: Sideslip angle [degrees]
        :param V: Airspeed magnitude [m/s]

        :return: 3D aerodynamic force vector in wind frame [N]
                - X_w: -drag (along relative wind)
                - Y_w: -lateral (to the right wing)
                - Z_w: -lift (downward in wind frame)
        """

        # === Aerodynamic coefficients from polynomials ===
        CL = np.polyval(self.a_CL_c, np.clip(AoA, -60, 60))     # Lift coefficient vs AoA
        CDL = np.polyval(self.a_CDL_c, AoA)                     # Longitudinal drag vs AoA
        CDY = np.polyval(self.a_CDY_c, sideslip)                # Lateral drag (usually negligible)
        CY  = np.polyval(self.a_CY_c, np.clip(sideslip, -60, 60))  # Side force vs sideslip

        # === Dynamic pressure ===
        q = 0.5 * self.rho * V**2  # Bernoulli dynamic pressure [Pa]

        # === Compute forces in wind frame ===
        lift = CL * q * self.a_surface      # Z_w: Lift (acts perpendicular to wind)
        drag = CDL * q * self.a_surface     # X_w: Drag (acts along wind vector)
        lateral = CY * q * self.a_surface   # Y_w: Side force (from sideslip)

        # === Return wind-frame aerodynamic force vector ===
        # Wind frame axes:
        #   X_w: along airspeed vector (forward)
        #   Y_w: right wing
        #   Z_w: down when AoA is positive
        return np.array([
            -drag,    # Negative because drag opposes motion
            -lateral, # Negative: rightward sideslip causes leftward force
            -lift     # Negative: positive lift acts upward in body frame
        ])

    def Ailerons(self, angle, V):
        """
        Compute aerodynamic forces and rolling moment generated by the ailerons.

        Models differential lift on left and right ailerons based on input deflection.

        :param angle: Aileron deflection angle [scalar, degrees], 
                    positive = right aileron down / left aileron up (right roll)
        :param V: Airspeed magnitude [m/s]

        :return:
            force: Total aerodynamic force vector in wind frame [N]
            roll_moment: Rolling moment due to differential lift [Nm]
        """

        # === Simplified lift model for left and right ailerons ===
        # CL is approximated linearly instead of using polynomials for now

        # Right aileron (CL increases with positive angle)
        CL_1 = 0.5 * angle
        CD_1 = np.polyval(self.al_CD_c, angle * 40)  # Adjust scaling to match lookup shape

        # Left aileron (opposite deflection)
        CL_2 = 0.5 * -angle
        CD_2 = np.polyval(self.al_CD_c, -angle * 40)

        # === Dynamic pressure ===
        q = 0.5 * self.rho * V**2

        # === Forces in wind frame ===
        lift_1 = CL_1 * q * self.al_surface
        lift_2 = CL_2 * q * self.al_surface
        drag = (CD_1 + CD_2) * q * self.al_surface
        lateral = 0  # Ailerons don't generate side force in this model

        # === Total aerodynamic force ===
        # Assumes lift acts in -Z_w, drag in -X_w
        force = np.array([
            -drag,               # Drag (along wind direction)
            -lateral,            # Side force (zero in this model)
            -(lift_1 + lift_2)   # Total lift from both ailerons
        ])

        # === Rolling moment (about X_b) ===
        # Uses vertical force * lateral offset (y-position of ailerons in body frame)
        roll_moment = 2 * (lift_1 * self.al_COF[1])  # Both ailerons assumed symmetric

        return force, np.array([roll_moment, 0, 0])

    def Elevators(self, angle, V):
        """
        Compute aerodynamic forces generated by the elevators (or elevons),
        returned in the wind frame.

        :param angle: Elevator deflection angle [degrees]
                    Positive angle = trailing edge down = nose-up moment
        :param V: Airspeed magnitude [m/s]

        :return:
            force: Aerodynamic force vector in wind frame [N]
                - X_w: -drag
                - Y_w: (0, no lateral force assumed)
                - Z_w: +lift (positive = up in body frame)
        """

        # === Aerodynamic coefficients ===
        # Lift is assumed linear in angle; you use a simplified model here.
        CL = 0.6 * -angle  # Negate for consistent convention: positive deflection → nose-up
        CD = np.polyval(self.el_CD_c, -angle * 40)  # Drag increases with deflection

        # === Dynamic pressure ===
        q = 0.5 * self.rho * V**2

        # === Forces in wind frame ===
        lift = CL * q * self.el_surface * 2   # Two elevators assumed
        drag = CD * q * self.el_surface * 2
        lateral = 0  # Elevators typically generate no side force

        return np.array([
            -drag,     # -X_w: drag resists motion
            lateral,   #  Y_w: assumed zero
            lift       #  Z_w: lift acts upward in body frame (positive here)
        ])

    def Rudders(self, angle, V):
        """
        Compute aerodynamic forces generated by the rudders, expressed in the wind frame.

        :param angle: Rudder deflection angle [degrees]
                    Positive = rudder right → nose yaw right (standard convention)
        :param V: Airspeed magnitude [m/s]

        :return:
            force: Aerodynamic force vector in wind frame [N]
                - X_w: -drag
                - Y_w: -side force (acts left for positive yaw)
                - Z_w: 0 (no lift from rudders)
        """

        # === Aerodynamic coefficients ===
        # You use a linearized side force model and a polynomial drag model.
        CY = 0.2 * angle  # Approximate linear relation (or use poly if needed)
        CD = np.polyval(self.r_CD_c, angle * 40)  # Scaled input for polynomial domain

        # === Dynamic pressure ===
        q = 0.5 * self.rho * V**2

        # === Forces in wind frame ===
        lift = 0  # Rudders do not produce vertical lift (aligned with vertical fin)
        drag = CD * q * self.r_surface * 2        # Total drag from both rudders
        lateral = CY * q * self.r_surface * 2     # Lateral force causes yawing moment

        return np.array([
            -drag,     # -X_w: drag opposes motion
            -lateral,  # -Y_w: side force opposes yaw (standard sign convention)
            -lift      # -Z_w: 0 here
        ])
    
    def Aerobrake(self, V, deploy):
        # === Dynamic pressure ===
        q = 0.5 * self.rho * V**2
        CD = self.aerobrake_CD * deploy

        drag = CD * q * self.aerobrake_surface

        return np.array([-drag, 0, 0])

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
            [cp * cy, cp * sy, -sp],
            [sy * sp * cy - cr * sy, sr * sp * sy + cr * cy, sr * cp],
            [cr * sp * cy + sr * sy, cy * sp * sy - sr * cy, cr * cp]
        ])
        return R

    def body_to_wind(self, AoA, sideslip):
        """
        Compute the rotation matrix from the body frame to the wind frame.

        The wind frame is aligned with the relative airflow:
        - X_w: direction of incoming air (opposite to velocity vector)
        - Y_w: lateral direction (perpendicular to X_w in horizontal plane)
        - Z_w: completes right-handed system (typically points down if AoA > 0)

        :param AoA: Angle of Attack [rad]
        :param sideslip: Sideslip angle [rad]

        :return: 3x3 rotation matrix R such that:
                v_wind = R @ v_body
        """

        # === Trig shorthands ===
        ca = np.cos(AoA)
        sa = np.sin(AoA)
        cb = np.cos(sideslip)
        sb = np.sin(sideslip)

        # === Rotation matrix: Body → Wind ===
        # This transforms a vector from body coordinates to wind-aligned coordinates.
        # Order: sideslip (Y-axis), then angle of attack (Z-axis)
        R = np.array([
            [cb * ca, sb, cb * sa],
            [-sb * ca, cb, -sb * sa],
            [-sa,     0,   ca]
        ])

        return R

    def wind_to_body(self, AoA, sideslip):
        """
        Compute the rotation matrix from the wind frame to the body frame.

        This is the inverse of the body-to-wind transformation, and is used
        to rotate aerodynamic force vectors (defined in wind coordinates)
        into the aircraft body frame.

        :param AoA: Angle of Attack [rad]
        :param sideslip: Sideslip angle [rad]

        :return: 3x3 rotation matrix R such that:
                v_body = R @ v_wind
        """
        R = self.body_to_wind(AoA, sideslip).T  # Inverse = transpose for rotation matrices
        return R

    def getTelemetry(self):
        """
        Return the full telemetry dictionary collected during simulation.

        :return: dict of lists containing logged state variables
        """
        return self.telemetry

    def get_absolute_velocity(self):
        """
        Get the velocity vector in the world (inertial) frame by transforming
        the body-frame velocity using the current orientation.

        :return: 3D velocity vector in inertial frame [m/s]
        """
        absolute_vel = self.body_to_vehicle(self.o[0], self.o[1], self.o[2]) @ self.v
        return absolute_vel

    def get_pos(self):
        """
        Get the current position of the aircraft in the world (inertial) frame.

        :return: 3D position vector [m]
        """
        return self.p

    def get_max_acc(self):
        """
        Estimate the maximum vertical acceleration (approximate lift limit)
        at 35° angle of attack and current speed.

        Used to determine acceleration limits for physics capping or safety checks.

        :return: Max lift-based acceleration in +Z_body direction [m/s²]
        """
        V = np.linalg.norm(self.v)  # Airspeed magnitude
        _, _, lift = self.Airframe(np.deg2rad(35), 0, V)  # Max lift estimate at AoA = 35°
        return -lift / self.m  # Acceleration = Force / Mass, sign reflects body-frame axis


