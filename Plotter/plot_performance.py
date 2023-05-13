import matplotlib.pyplot as plt
import json

if __name__ == '__main__':
    with open(b"log.json", "r") as read_file:
        log = json.load(read_file)

    # --------------------------------------------------------------
    # Plotting the cross and along track errors and control commands
    # --------------------------------------------------------------
    acc_cte = round(log['acc_cross_track_error'], 1)
    acc_ate = round(log['acc_along_track_error'], 1)
    lap_time = round(log['lap_time'], 1)


    t = range(1, len(log['cross_track_error']) + 1)

    fig, axs = plt.subplots(2)
    fig.suptitle(f'Trajectory Errors and Control Commands\nAcc. Cross-Track error = {acc_cte} [m], '
                 f'Acc. Along-Track error = {acc_ate} [m]\nLap time = {lap_time} [sec]', fontweight='bold')
    axs[0].plot(t, log['cross_track_error'], color='b', label='Cross Track Error')
    axs[0].set_ylabel('Cross Track Error [m]', color='b')
    twin = axs[0].twinx()
    twin.plot(t, log['steering_command'], color='r', label='Steering Command')
    twin.set_ylabel('Steering Command [rad]', color='r')
    # axs[0].set_xlabel('Simulation step')
    axs[0].set_title('Cross Track Control')
    axs[0].grid()

    axs[1].plot(t, log['along_track_error'], color='b', label='Along Track Error')
    axs[1].set_ylabel('Along Track Error [m]', color='b')
    twin = axs[1].twinx()
    twin.plot(t, log['speed_command'], color='r', label='Speed Command')
    twin.set_ylabel('Speed Command [m/s]', color='r')
    axs[1].set_xlabel('Simulation step')
    axs[1].set_title('Along Track Control')
    axs[1].grid()

    plt.show()

    # --------------------------------------------------------------
    # Plotting robot's path in respect to the trajectory path
    # --------------------------------------------------------------
    fig = plt.figure()
    plt.plot(log['traj_x'], log['traj_y'], 'go--', label='Trajectory', linewidth=2, markersize=3)
    plt.plot(log['robot_x'], log['robot_y'], 'ro-', label='Robot', linewidth=2,  markersize=3)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.legend()
    plt.title('Robot Actual vs Desired Trajectory', fontweight='bold')
    plt.grid()
    plt.show()

