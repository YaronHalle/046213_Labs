import matplotlib.pyplot as plt
import json

if __name__ == '__main__':
    with open(b"C:\Users\YARONHA\Downloads\log2.json", "r") as read_file:
        log = json.load(read_file)

    t = range(1, len(log['cross_track_error']) + 1)

    fig, axs = plt.subplots(2)
    fig.suptitle('Trajectory Errors and Control Commands', fontweight='bold')
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

