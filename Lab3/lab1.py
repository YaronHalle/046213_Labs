import rclpy
from rclpy.node import Node
import sys
import time
import os
import json

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Transform
from geometry_msgs.msg import Quaternion
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import TransformBroadcaster

from ament_index_python.packages import get_package_share_directory
import gym
import numpy as np
from transforms3d import euler


class FilterData(object):
    '''
    Helper class for storing previous EKF filter status
    '''
    def __init__(self, time, speed_cmd, steering_cmd, pose, cov):
        self.time = time
        self.speed_cmd = speed_cmd
        self.steering_cmd = steering_cmd
        self.pose = pose
        self.cov = cov

# this node should get the current pose from odom and get a reference trajectory from a yaml file
# and publish ackermann drive commands to the car based on one of 4 controllers selected with a parameter
# the controllers are PID, Pure Pursuit, iLQR, and an optimal controller 
class Lab1(Node):
    def __init__(self, controller_type: str = 'pid', traj_type: str = 'ref'):
        super().__init__('lab1')
        self.get_logger().info("Lab 1 Node has been started")

        self.odom_sub_for_lap = self.create_subscription(Odometry, '/ego_racecar/odom', self.lap_callback, 10)
        
        # get parameters
        self.controller = self.declare_parameter('controller', controller_type).value
        self.get_logger().info("Controller: " + self.controller)
        # to set the parameter, run the following command in a terminal when running a different controller
        # ros2 run f1tenth_gym_ros lab1.py --ros-args -p controller:=<controller_type>

        # get the current pose
        self.get_logger().info("Subscribing to Odometry")
        self.odom_sub = self.create_subscription(PoseWithCovarianceStamped, '/ekf_pose', self.odom_callback, 10)
        self.odom_sub # prevent unused variable warning
        
        self.get_logger().info("Publishing to Ackermann Drive")
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.cmd_pub # prevent unused variable warning
        
        # get the reference trajectory
        self.get_logger().info("Loading Reference Trajectory")
        self.ref_traj = np.load(os.path.join(get_package_share_directory('f1tenth_gym_ros'),
                                            'resource',
                                            traj_type + '_traj.npy'))
        self.ref_traj # prevent unused variable warning
        
        # create a timer to publish the control input every 20ms
        self.dt = 0.5
        self.get_logger().info("Creating Timer")
        self.timer = self.create_timer(self.dt, self.timer_callback)
        self.timer # prevent unused variable warning
        
        self.pose = np.zeros(3)

        self.current_cross_track_error = 0
        self.current_along_track_error = 0
        self.cross_track_accumulated_error = 0
        self.along_track_accumulated_error = 0
        self.along_track_error_integral = 0
        self.cross_track_error_integral = 0
        self.waypoint_index = 0
        self.last_steering_command = 0
        self.last_speed_command = 0
        self.theta_ref = 0
        self.velocity = 0
        self.last_time = self.get_clock().now().nanoseconds * 1e-9
        self.new_ekf_data = False
        self.last_ekf_pose = np.zeros(3)
        self.last_ekf_covariance = self.compute_R_matrix()
        self.measured_pose = np.zeros(3)
        self.measured_covariance = np.zeros((3,3))
        self.measurement_time = 0
        self.moved = False
        self.is_odom_clock_synced = False
        self.odom_to_ekf_time_shift = 0
        self.filter_hist = []

    def get_ref_pos(self):
        # get the next waypoint in the reference trajectory based on the current time
        waypoint = self.ref_traj[self.waypoint_index % len(self.ref_traj)]
        self.waypoint_index += 1
        return waypoint

    def log_accumulated_error(self):
        ref_pos = np.array(self.get_ref_pos())
        next_ref_pos = np.array(self.get_ref_pos())
        self.waypoint_index -= 1
        x = self.pose[0]
        y = self.pose[1]
        theta = self.pose[2]
        x_ref = next_ref_pos[0]
        y_ref = next_ref_pos[1]

        # compute the trajectory vector between previous and next points for getting theta_ref
        self.theta_ref = np.arctan2(next_ref_pos[1] - ref_pos[1], next_ref_pos[0] - ref_pos[0])

        # compute the cross track and along track errors
        cross_track_error = -np.sin(self.theta_ref) * (x - x_ref) + np.cos(self.theta_ref) * (y - y_ref)
        along_track_error =  np.cos(self.theta_ref) * (x - x_ref) + np.sin(self.theta_ref) * (y - y_ref)

        self.current_cross_track_error = cross_track_error
        self.current_along_track_error = along_track_error

        # update accumulators needed for controllers
        self.along_track_error_integral += along_track_error
        self.cross_track_error_integral += cross_track_error
        if self.waypoint_index > 1:  # don't calculate derivatives on the first waypoint
            # Computing along and cross track errors derivatives
            self.along_track_error_derivative = \
                (along_track_error - self.prev_along_track_error) / self.dt  # length of timestep
            self.cross_track_error_derivative = \
                (cross_track_error - self.prev_cross_track_error) / self.dt  # length of timestep

            # Computing velocity by differentiating the current and previous poses
            robot_translation = self.pose[0:2] - self.prev_pose[0:2]
            self.velocity = np.linalg.norm(robot_translation) / self.dt

        self.prev_along_track_error = along_track_error
        self.prev_cross_track_error = cross_track_error
        self.prev_pose = self.pose

        print(f'Waypoint = {self.waypoint_index}')

        # log the accumulated error to screen and internally to be printed at the end of the run
        self.get_logger().info("Cross Track Error: " + str(cross_track_error))
        self.get_logger().info("Along Track Error: " + str(along_track_error))
        self.cross_track_accumulated_error += abs(cross_track_error)
        self.along_track_accumulated_error += abs(along_track_error)
        
    def lap_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        if not self.moved and (x < -1 and y > 3):
            self.moved = True
        elif self.moved and x > 0:
            raise EndLap
    
    def odom_callback(self, msg):
        # Reading measurement time
        self.measurement_time = msg.pose.covariance[2]

        # get the current pose
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = euler.quat2euler([q.w, q.x, q.y, q.z])

        self.pose = np.array([x, y, yaw])
        self.measured_pose = np.copy(self.pose)
        self.measured_covariance[0, 0] = msg.pose.covariance[0]
        self.measured_covariance[0, 1] = msg.pose.covariance[1]
        self.measured_covariance[0, 2] = msg.pose.covariance[5]
        self.measured_covariance[1, 0] = msg.pose.covariance[6]
        self.measured_covariance[1, 1] = msg.pose.covariance[7]
        self.measured_covariance[1, 2] = msg.pose.covariance[11]
        self.measured_covariance[2, 0] = msg.pose.covariance[30]
        self.measured_covariance[2, 1] = msg.pose.covariance[31]
        self.measured_covariance[2, 2] = msg.pose.covariance[35]
        self.new_ekf_data = True

    def forward_simulation_of_kineamtic_model(self, x, y, theta, v, delta, dt):
        d = 0.3302
        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = theta + (v / d) * np.tan(delta) * dt
        return np.array([x_new, y_new, theta_new])

    def compute_G_matrix(self, velocity, theta):
        G = np.zeros((3, 3))
        G[0, 2] = -velocity * np.sin(theta)
        G[1, 2] =  velocity * np.cos(theta)
        return G

    def compute_R_matrix(self):
        return np.diag((0.1, 0.1, 0.1))

    def ekf_predict(self, delta_time):
        Xprev = self.last_ekf_pose

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Matrices definitions
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        x = Xprev[0]
        y = Xprev[1]
        theta = Xprev[2]
        velocity = self.last_speed_command
        delta = self.last_steering_command

        G = self.compute_G_matrix(velocity, theta)
        R = self.compute_R_matrix()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Prediction Step
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Xpredict = self.forward_simulation_of_kineamtic_model(x, y, theta, velocity, delta, delta_time)
        Ppredict = G @ self.last_ekf_covariance @ G.T + R

        return Xpredict, Ppredict

    def ekf_predict_and_update(self, current_time, meas_time, measured_pose, measured_covariance):
        # Scanning filter status for the last prediction step before measurement time
        for i in range(len(self.filter_hist)):
            if self.filter_hist[i].time > meas_time:
                last_ekf_ind_before_meas = i-1
                break

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Matrices definitions
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        time_prev = self.filter_hist[last_ekf_ind_before_meas].time
        Xprev = self.filter_hist[last_ekf_ind_before_meas].pose
        Pprev = self.filter_hist[last_ekf_ind_before_meas].cov
        velocity = self.filter_hist[last_ekf_ind_before_meas].speed_cmd
        delta = self.filter_hist[last_ekf_ind_before_meas].steering_cmd

        x = Xprev[0]
        y = Xprev[1]
        theta = Xprev[2]

        G = self.compute_G_matrix(velocity, theta)
        H = np.eye(3)
        R = self.compute_R_matrix()
        z = measured_pose
        Q = measured_covariance

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # First prediction step (from last predict before measurement -> measurement time)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        dt_before_measurement = meas_time - time_prev
        Xpredict = self.forward_simulation_of_kineamtic_model(x, y, theta, velocity, delta, dt_before_measurement)
        Ppredict = G @ Pprev @ G.T + R

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Computing the Kalman gain
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        K = Ppredict @ H.T @ np.linalg.pinv(H @ Ppredict @ H.T + Q)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update step in respect to the measurement time
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Xnext = Xpredict + K @ (z - Xpredict)
        Pnext = (np.eye(3) - K @ H) @ Ppredict

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Second prediction step (from measurement time -> last ekf filter time)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        x = Xnext[0]
        y = Xnext[1]
        theta = Xnext[2]

        # Predicting from measurement time to the following filter time
        dt_after_measurement = self.filter_hist[last_ekf_ind_before_meas + 1].time - meas_time
        Xpredict = self.forward_simulation_of_kineamtic_model(x, y, theta, velocity, delta, dt_after_measurement)
        Ppredict = G @ Pnext @ G.T + R

        # Predicting from the following filter time all the way until the last stored filter
        for i in range(last_ekf_ind_before_meas + 1, len(self.filter_hist) - 1):
            x = Xpredict[0]
            y = Xpredict[1]
            theta = Xpredict[2]
            velocity = self.filter_hist[i].speed_cmd
            delta = self.filter_hist[i].steering_cmd

            # Predicting from measurement time to the following filter time
            delta_time = self.filter_hist[i + 1].time - self.filter_hist[i].time
            Xpredict = self.forward_simulation_of_kineamtic_model(x, y, theta, velocity, delta, delta_time)
            G = self.compute_G_matrix(velocity, theta)
            Ppredict = G @ Ppredict @ G.T + R

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Third prediction step (from last ekf filter time -> current time)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        x = Xpredict[0]
        y = Xpredict[1]
        theta = Xpredict[2]
        delta_time = current_time - self.last_time
        velocity = self.last_speed_command
        delta = self.last_steering_command
        G = self.compute_G_matrix(velocity, theta)
        Xpredict = self.forward_simulation_of_kineamtic_model(x, y, theta, velocity, delta, delta_time)
        Ppredict = G @ Ppredict @ G.T + R

        return Xpredict, Ppredict

    def timer_callback(self):
        # Updating time variables
        current_time = self.get_clock().now().nanoseconds * 1e-9
        delta_time = current_time - self.last_time

        # compute the control input
        if self.controller == "pid_unicycle":
            u = self.pid_unicycle_control(self.pose)
        elif self.controller == "pid":
            u = self.pid_control(self.pose)
        elif self.controller == "pure_pursuit":
            u = self.pure_pursuit_control(self.pose)
        elif self.controller == "ilqr":
            u = self.ilqr_control(self.pose)
        elif self.controller == "optimal":
            if self.new_ekf_data and len(self.filter_hist) > 0:
                # EKF Prediction and Update Steps
                self.last_ekf_pose, self.last_ekf_covariance = \
                    self.ekf_predict_and_update(current_time, self.measurement_time, self.measured_pose, self.measured_covariance)
                self.new_ekf_data = False
            else:
                # EKF Prediction only step
                self.last_ekf_pose, self.last_ekf_covariance = self.ekf_predict(delta_time)

            u = self.optimal_control(self.last_ekf_pose)
            self.log_accumulated_error()
            self.last_time = current_time

            # Adding current EKF status to history array
            filter_status = FilterData(current_time, u[0], u[1], self.last_ekf_pose, self.last_ekf_covariance)
            self.filter_hist.append(filter_status)
        else:
            self.get_logger().info("Unknown controller")
            return
        
        # publish the control input
        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"
        cmd.drive.steering_angle = u[0]
        cmd.drive.speed = u[1]
        self.cmd_pub.publish(cmd)

        # Keeping record of the last commands
        self.last_steering_command = u[0]
        self.last_speed_command = u[1]

    def pid_control(self, pose):
        #### YOUR CODE HERE ####

        # return np.array([steering_angle, speed])
        #### END OF YOUR CODE ####
        raise NotImplementedError
    
    def pid_unicycle_control(self, pose):
        #### YOUR CODE HERE ####
        
        
        # return np.array([steering_angle, speed])
        #### END OF YOUR CODE ####
        raise NotImplementedError
    
    def pure_pursuit_control(self, pose):
        #### YOUR CODE HERE ####
        
        
        # return np.array([steering_angle, speed])
        #### END OF YOUR CODE ####
        raise NotImplementedError
        
    def ilqr_control(self, pose):
        #### YOUR CODE HERE ####
        
        
        # return np.array([steering_angle, speed])
        #### END OF YOUR CODE ####
        raise NotImplementedError
        
    def optimal_control(self, pose):
        theta_error = pose[2] - self.theta_ref

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Along-track PID control
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        K_at_p = 2
        K_at_d = 0.2
        K_at_i = 0.4
        speed = -(K_at_p * self.current_along_track_error +
                  K_at_d * self.velocity * np.cos(theta_error) +
                  K_at_i * self.along_track_error_integral)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Cross-track Bang-Bang + Pure Pursuit control
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        lookahead = 7
        d = 0.3302  # wheelbase as defined in .xacro file

        lookahead_waypoint = self.ref_traj[(self.waypoint_index + lookahead) % len(self.ref_traj)]
        x_ref = lookahead_waypoint[0]
        y_ref = lookahead_waypoint[1]

        x = pose[0]
        y = pose[1]
        heading_angle = pose[2]
        waypoint_angle = np.arctan2(y_ref - y, x_ref - x)

        alpha = (waypoint_angle - heading_angle)
        L = np.sqrt((x - x_ref) ** 2 + (y - y_ref) ** 2)

        steering_angle = np.arctan((2 * d * np.sin(alpha)) / L)

        # Enforcing commands cut-off
        max_speed = 0.5  # [m/s]
        max_angle = 0.2  # [rad]

        if speed > max_speed:
            speed = max_speed
        elif speed < -max_speed:
            speed = -max_speed

        # Steering angle is determined by a "BANG-BANG" policy
        if steering_angle > 0:
            steering_angle = max_angle
        elif steering_angle < 0:
            steering_angle = -max_angle

        return np.array([steering_angle, speed])

class EndLap(Exception):
    # this exception is raised when the car crosses the finish line
    pass

def main(args=None):
    rclpy.init()

    if len(sys.argv) < 3:
        lab1 = Lab1(controller_type=sys.argv[1])
    else:
        lab1 = Lab1(controller_type=sys.argv[1], traj_type=sys.argv[2])

    tick = time.time()
    try:
        rclpy.spin(lab1)
    except NotImplementedError:
        rclpy.logging.get_logger('lab1').info("You havn't implemented this controller yet!")
    except EndLap:
        tock = time.time()
        rclpy.logging.get_logger('lab1').info("Finished lap")
        rclpy.logging.get_logger('lab1').info("Cross Track Error: " + str(lab1.cross_track_accumulated_error))
        rclpy.logging.get_logger('lab1').info("Along Track Error: " + str(lab1.along_track_accumulated_error))
        rclpy.logging.get_logger('lab1').info("Lap Time: " + str(tock - tick))
        print("Cross Track Error: " + str(lab1.cross_track_accumulated_error))
        print("Along Track Error: " + str(lab1.along_track_accumulated_error))
        print("Lap Time: " + str(tock - tick))

    lab1.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main(sys.argv)
    
    
        