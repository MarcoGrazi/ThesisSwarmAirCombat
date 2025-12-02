from Physics import FixedWingAircraft
import yaml
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_telemetry(telemetry, save_folder):
    """
    Plot and save telemetry for each agent as separate figures and CSV files.

    Args:
    - save_folder: Directory to save the plots and data
    """
    os.makedirs(save_folder, exist_ok=True)

    pos = np.array(telemetry['position'])
    vel = np.array(telemetry['velocity'])
    accel = np.array(telemetry['acceleration'])
    eulers = np.array(np.rad2deg(telemetry['orientation']))
    force = np.array(telemetry['force'])
    moment = np.array(telemetry['moment'])
    cmds = np.array(telemetry['commands'])
    AoA = np.array(np.rad2deg(telemetry['AoA']))
    sideslip = np.array(np.rad2deg(telemetry['sideslip']))

    # Create plots
    fig, axs = plt.subplots(8, 1, figsize=(18, 18))

    axs[0].plot(pos[:, 0], label='X')
    axs[0].plot(pos[:, 1], label='Y')
    axs[0].plot(pos[:, 2], label='Z')
    axs[0].set_title("Position")
    axs[0].legend()

    axs[1].plot(vel[:, 0], label='Vx')
    axs[1].plot(vel[:, 1], label='Vy')
    axs[1].plot(vel[:, 2], label='Vz')
    axs[1].set_title("Velocity")
    axs[1].legend()

    axs[2].plot(accel[:, 0], label='Ax')
    axs[2].plot(accel[:, 1], label='Ay')
    axs[2].plot(accel[:, 2], label='Az')
    axs[2].set_title("Acceleration")
    axs[2].legend()

    axs[3].plot(eulers[:, 0], label='Roll (°)')
    axs[3].plot(eulers[:, 1], label='Pitch (°)')
    axs[3].plot(eulers[:, 2], label='Yaw (°)')
    axs[3].set_title("Euler Angles (Global Frame)")
    axs[3].legend()

    axs[4].plot(force[:, 0], label='Fx')
    axs[4].plot(force[:, 1], label='Fy')
    axs[4].plot(force[:, 2], label='Fz')
    axs[4].set_title("Force")
    axs[4].legend()

    axs[5].plot(moment[:, 0], label='Mx')
    axs[5].plot(moment[:, 1], label='My')
    axs[5].plot(moment[:, 2], label='Mz')
    axs[5].set_title("Moment")
    axs[5].legend()

    axs[6].plot(cmds[:, 0], label='Throttle')
    axs[6].plot(cmds[:, 1], label='Elevon')
    axs[6].plot(cmds[:, 2], label='Aileron')
    axs[6].plot(cmds[:, 3], label='Rudder')
    axs[6].set_title("Control Inputs")
    axs[6].legend()

    axs[7].plot(AoA[:], label='AoA')
    axs[7].plot(sideslip[:], label='sideslip')
    axs[7].set_title("Wind Angles")
    axs[7].legend()

    plt.tight_layout()
    fig_path = os.path.join(save_folder, f"PID_telemetry_plot.png")
    plt.savefig(fig_path)
    plt.close()


# Load config
with open("Train_Run_config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# PID state variables (initialize globally or later wrap in a class)
prev_errors = {'AoA': 0, 'sideslip': 0, 'roll': 0, 'speed': 0}
integrals = {'AoA': 0, 'sideslip': 0, 'roll': 0, 'speed': 0}
last_cmd = {'elevator': 0.0, 'aileron' : 0.0, 'rudder'  : 0.0, 'throttle': 0.0}

# Max change per second in normalized units ([-1,1] for surfaces, [0,1] for throttle)
rate_limits = {
    'elevator': 20,   # tune
    'aileron' : 20,   # usually fastest
    'rudder'  : 20,   # often the slowest
    'throttle': 20,   # smoother power changes
}


# PID gains (these are arbitrary starting points — you'll need to tune them)
gains = {
    'AoA':     {'kp': 0.8, 'ki': 0, 'kd': 0.3},
    'sideslip':{'kp': 3, 'ki': 0.1, 'kd': 0.8},
    'roll':    {'kp': 3, 'ki': 0.1, 'kd': 0.8},
    'speed':   {'kp': 20, 'ki': 0, 'kd': 0}
}

dt = 1.0 / config['env_config']['physics_frequency']  # Time step


def rate_limit(name, cmd, dt):
    """
    Per-actuator slew-rate limiter.
    name: 'elevator' | 'aileron' | 'rudder' | 'throttle'
    cmd : desired normalized command after clipping
    dt  : seconds
    """
    rl = rate_limits[name]
    if rl <= 0 or dt <= 0:
        last_cmd[name] = cmd
        return cmd

    prev = last_cmd[name]
    max_step = rl * dt
    delta = cmd - prev

    if   delta >  max_step: cmd = prev + max_step
    elif delta < -max_step: cmd = prev - max_step

    last_cmd[name] = cmd
    return cmd



def PID_Control(action):
    """
    PID controller that outputs values in the correct range:
    - Throttle ∈ [0, 1]
    - Elevator, Aileron, Rudder ∈ [-1, 1]
    """
    global prev_commands, damping

    # Desired values
    target_AoA, target_sideslip, target_roll, target_speed = action

    #scale to degrees
    target_AoA *= 40
    target_sideslip *= 40
    target_roll *= 180
    target_speed *= 343
    print(f"targets: {target_AoA}, {target_sideslip}, {target_roll}, {target_speed}")


    # Get current state
    telemetry = aircraft.getTelemetry()
    current_AoA = np.rad2deg(telemetry['AoA'][-1])
    current_sideslip = np.rad2deg(telemetry['sideslip'][-1])
    current_roll = np.rad2deg(telemetry['orientation'][-1][0])  # roll in radians
    current_speed = np.linalg.norm(telemetry['velocity'][-1])

    # Compute errors (convert radians to degrees for clarity)
    errors = {
        'AoA': target_AoA - current_AoA,
        'sideslip': target_sideslip - current_sideslip,
        'roll': ((target_roll - current_roll) + 180) % 360 - 180,
        'speed': target_speed - current_speed
    }
    print(f"errors: {errors["AoA"]}, {errors["sideslip"]}, {errors["roll"]}, {errors["speed"]}")
    outputs = {}

    for key in errors:
        integrals[key] += errors[key] * dt
        derivative = (errors[key] - prev_errors[key]) / (dt)
        gains_k = gains[key]
        outputs[key] = (
            gains_k['kp'] * errors[key] +
            gains_k['ki'] * integrals[key] +
            gains_k['kd'] * derivative
        )
        prev_errors[key] = errors[key]

    # Normalize control outputs
    commands = {
        'elevator' : -np.clip(outputs['AoA']/40, -1, 1),                       # [-1, 1]
        'rudder' : -np.clip(outputs['sideslip'] / 40, -1, 1),                  # [-1, 1]
        'aileron' : np.clip(outputs['roll']/40, -1, 1),                        # [-1, 1]
        'throttle' : np.clip(outputs['speed'] / 343, -1, 1.0)          # [0, 1]
    }

    # Apply per-surface rate limits
    commands['elevator'] = rate_limit('elevator', commands['elevator'], dt)
    commands['aileron']  = rate_limit('aileron',  commands['aileron'],  dt)
    commands['rudder']   = rate_limit('rudder',   commands['rudder'],   dt)
    commands['throttle'] = rate_limit('throttle', commands['throttle'], dt)

    # (Optional safety) clip again in case numerical drift occurred
    commands['elevator'] = float(np.clip(commands['elevator'], -1.0, 1.0))
    commands['aileron']  = float(np.clip(commands['aileron'],  -1.0, 1.0))
    commands['rudder']   = float(np.clip(commands['rudder'],   -1.0, 1.0))
    commands['throttle'] = float(np.clip(commands['throttle'],  -1, 1.0))

    print(f"surfaces: {commands['elevator']}, {commands['rudder']}, {commands['aileron']}, {commands['throttle']}")
    return commands['throttle'], commands['elevator'], commands['aileron'], commands['rudder']


aircraft = FixedWingAircraft(config['uav_config'][2], config['env_config']['rho'],
                             config['env_config']['g'], config['env_config']['physics_frequency'])

aircraft.reset([0,0,0], [0,0,0], 200, config['uav_config'][3])

commands_sequence = [
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0.9, 0, 0, 1],
    [0.9, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
]
c = 0
action = commands_sequence[c]
for i in range(1200):
    if i%120 == 0 and i != 0:
        c += 1
        print(c)
        action = commands_sequence[c]
        print(command)

    command = PID_Control(action)
    t, e, a, r = command
    aircraft.step(t, e, a, r, command)

print(config['uav_config'][0]['gains'])
plot_telemetry(aircraft.getTelemetry(), 'PID_TEST')
