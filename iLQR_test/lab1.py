import time
import os
import json

import numpy as np

def get_ref_pos(self):
    # get the next waypoint in the reference trajectory based on the current time
    waypoint = self.ref_traj[self.waypoint_index % len(self.ref_traj)]
    self.waypoint_index += 1
    return waypoint

class iLQR():
    def __init__(self, ref_traj):
        self.wheelbase = 0.3302
        self.log = {}
        self.ref_traj = ref_traj
        self.dt = 0.5

    def ilqr_control(self, pose, Xinit=None, Uinit=None):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Defining Q,R cost matrices after optimization
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        N_states = 3
        N_inputs = 2
        Q = np.zeros((N_states, N_states))
        R = np.zeros((N_inputs, N_inputs))

        # State vector penalties
        Q[0, 0] = 100  # x
        Q[1, 1] = 100  # y
        Q[2, 2] = 10  # theta

        rho = 1
        rho_delta = 0.1
        convergence_improv_ratio = 0.01

        # Input vector penalties
        R[0, 0] = 1  # velocity
        R[1, 1] = 0.1  # steering angle

        N_max_iterations = 100

        # Creating the reference X and U using the differential flat model
        Xref, Uref = self.get_diff_flat_X_and_U(self.ref_traj)

        traj_len = Xref.shape[1]

        # Initializing the best cost so far to be infinity
        prev_cost = float('inf')

        # Main for loop of iLQR
        for current_iter in range(0, N_max_iterations):
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Setting the reference trajectory for the
            # current iteration
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if current_iter == 0:
                # Using the initial trajectory on first iteration
                X, U = Xinit, Uinit
            else:
                # Otherwise, using the trajectory from previous iteration
                X, U = Xnext, Unext

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Creating a system model representing the deviations
            # from the desired reference trajectory
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Populating A and B matrices representing the linearized system dynamics
            velocity = U[0, :]
            steering_angle = U[1, :]
            theta = X[2, :]

            A = np.zeros((N_states, N_states, traj_len))
            A[0, 0, :] = -velocity * np.sin(theta)
            A[0, 1, :] = velocity * np.cos(theta)

            B = np.zeros((N_states, N_inputs, traj_len))
            B[0, 0, :] = np.cos(theta)
            B[1, 0, :] = np.sin(theta)
            B[2, 0, :] = np.tan(steering_angle) / self.wheelbase
            B[2, 1, :] = velocity / self.wheelbase / np.power(np.cos(steering_angle), 2)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Backward pass for computing d_t and K_t (feedback
            # gain and bias) for computing the Unext
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            d = np.zeros((N_inputs, traj_len))
            K = np.zeros((N_inputs, N_states, traj_len))

            # Computing the s and S (first and second derivatives of the future cost function)
            s = 2 * (X[:, -1] - Xref[:, -1]).transpose() @ Q
            S = Q.transpose() + rho * np.eye(N_states)

            # Backwards loop
            for i in range(traj_len - 2, -1, -1):
                d[:, i] = -np.linalg.pinv(2 * R + B[:, :, i].transpose() @ S @ B[:, :, i]) @ \
                          (2 * U[:, i] @ R - 2 * Uref[:, i].transpose() @ R + s @ B[:, :, i])
                K[:, :, i] = -np.linalg.pinv(2 * R + B[:, :, i].transpose() @ S @ B[:, :, i]) @ B[:, :, i].transpose() @ S @ A[:, :, i]

                s += 2 * (X[:, i] - Xref[:, i]).transpose() @ Q
                S += Q.transpose()

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Forward pass for computing the next iteration's
            # X and U vectors
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            Xnext = np.zeros((N_states, traj_len))
            Unext = np.zeros((N_inputs, traj_len))

            # Initializing for t=0
            Xnext[:, 0] = np.array(pose)

            # For-looping forwards
            for i in range(0, traj_len - 1):
                # Computing control action
                Unext[:, i] = U[:, i] + K[:, :, i] @ (Xnext[:, i] - X[:, i]) + d[:, i]

                # Applying dynamics for computing the next state vector
                Xnext[:, i + 1] = Xnext[:, i] + self.dt * self.ackerman_non_linear(Xnext[:, i], Unext[:, i])

            # Computing the new trajectory cost
            new_cost = self.compute_cost(Xnext, Unext, Q, R, Xref, Uref)
            change_ratio = (prev_cost - new_cost) / prev_cost

            # DEBUG
            print(f'iLQR iteration {current_iter} : prev_cost = {prev_cost}, new_cost = {new_cost}')

            # Logging iteration outputs
            self.log[current_iter] = {}
            self.log[current_iter]['x'] = Xnext[0, :]
            self.log[current_iter]['y'] = Xnext[1, :]

            if new_cost < prev_cost:
                rho = rho * (1 - rho_delta)

                if change_ratio < convergence_improv_ratio:
                    break
            else:
                rho = rho * (1 + rho_delta)

            prev_cost = new_cost

        # return np.array([steering_angle, speed])

    def compute_cost(self, X, U, Q, R, Xref, Uref):
        '''
        Helper function for computing the total cost of a trajectory control.
        :param X: State matrix of size (N_states) x (N_trajectory_steps)
        :param U: Control input matrix of size (N_control_inputs) x (N_trajectory_steps)
        :param Q: Q matrix penalizing for state deviations
        :param R: R matrix penalizing for input commands deviations
        :param Xref: Reference state matrix of size (N_states) x (N_trajectory_steps)
        :param Uref: Reference control input matrix of size (N_control_inputs) x (N_trajectory_steps)
        :return: Computed cost (scalar value)
        '''
        cost = 0
        N_steps = Xref.shape[1]
        for i in range(0, N_steps - 1):
            # C(X,U) = (X-Xref)T * Q * (X-Xref) + (U-Uref)T * R * (U-Uref)
            cost += (X[:, i] - Xref[:, i]).transpose() @ Q @ (X[:, i] - Xref[:, i]) + \
                    (U[:, i] - Uref[:, i]).transpose() @ R @ (U[:, i] - Uref[:, i])

        cost += (X[:, -1] - Xref[:, -1]).transpose() @ Q @ (X[:, -1] - Xref[:, -1])

        return cost

    def ackerman_non_linear(self, X, U):
        '''
        Helper function for computing the next X state vector of the non-linear Ackerman car model
        using the current X state vector and U input commands vector.
        Inputs:
        @param X current state vector comprised of (x,y,theta)
        @param U current input vector comprised of (velocity, steering_angle)
        Outputs:
        @param X_next next state vector after computing the non-linear system dynamics
        '''
        theta = X[2]
        velocity = U[0]
        steering_angle = U[1]

        X_next = np.zeros(3)
        X_next[0] = velocity * np.cos(theta)
        X_next[1] = velocity * np.sin(theta)
        X_next[2] = velocity / self.wheelbase * np.tan(steering_angle)

        return X_next

    def get_diff_flat_X_and_U(self, ref_traj):
        '''
        Helper function for computing the X (state) and U (input commands) vector
        using the differential flat Ackerman model taugth in the tutorials.
        Input:
        @param ref_traj Reference trajctory comprised of [N,2] array of (x,y) coordinates.
        Output:
        @param X matrix of 3 rows (= 3 states) spread over number of steps of reference trajectory (columns)
        @param U matrix of 2 rows (= 2 inputs) spread over number of steps of reference trajectory (columns)

        '''
        ref_traj = np.array(ref_traj)
        N_steps = ref_traj.shape[0]
        x = ref_traj[:, 0]
        y = ref_traj[:, 1]
        X = np.zeros((3,N_steps)) # X state vector comprised of x,y,theta
        U = np.zeros((2,N_steps)) # U input vector comprised of velocity and steering angle

        # Computing the required derivatives of x and y
        xdot = np.diff(x) / self.dt
        ydot = np.diff(y) / self.dt

        # Duplicating the first element so xdot,ydot will have the same shape as x,y
        xdot = np.insert(xdot, 0, xdot[1])
        ydot = np.insert(ydot, 0, ydot[1])

        # Populating X matrix
        theta = np.arctan2(ydot, xdot)
        X[0, :] = x
        X[1, :] = y
        X[2, :] = theta

        velocity = xdot / np.cos(np.arctan2(ydot, xdot))
        # Using v=ydot/sin(theta) if xdot is zero
        for i in range(0, len(velocity)):
            if velocity[i] == 0:
                velocity[i] = ydot[i] / np.sin(np.arctan2(ydot[i], xdot[i]))
        thetadot = np.diff(theta) / self.dt
        thetadot = np.insert(thetadot, 0, thetadot[1])

        # Populating U matrix
        U[0, :] = velocity
        U[1, :] = np.arctan(self.wheelbase * thetadot / velocity)

        return X, U

def export_log_to_json(log):
    with open("log.json", "w") as write_file:
        json.dump(log, write_file)


if __name__ == '__main__':
    ref_traj = np.load('ref_traj.npy')

    with open(b"pp_traj.json", "r") as read_file:
        log = json.load(read_file)

    N_samples = len(log['cross_track_error'][0:-1])
    Xinit = np.zeros((3, N_samples))
    Uinit = np.zeros((2, N_samples))

    Xinit[0, :] = log['robot_x'][0:-1]
    Xinit[1, :] = log['robot_y'][0:-1]
    Xinit[2, :] = log['robot_theta'][0:-1]

    Uinit[0, :] = log['speed_command'][0:-1]
    Uinit[1, :] = log['steering_command'][0:-1]

    regulator = iLQR(ref_traj)
    pose = np.array([0, 0, 0])
    regulator.ilqr_control(pose, Xinit, Uinit)
    regulator.log['ref_traj'] = ref_traj
    # export_log_to_json(regulator.log)