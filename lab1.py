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

# this node should get the current pose from odom and get a reference trajectory from a yaml file
# and publish ackermann drive commands to the car based on one of 4 controllers selected with a parameter
# the controllers are PID, Pure Pursuit, iLQR, and an optimal controller 
class Lab1(Node):
    def __init__(self, controller_type: str = 'pid'):
        super().__init__('lab1')
        self.get_logger().info("Lab 1 Node has been started")

        # get parameters
        self.controller = self.declare_parameter('controller', controller_type).value
        self.get_logger().info("Controller: " + self.controller)
        # to set the parameter, run the following command in a terminal when running a different controller
        # ros2 run f1tenth_gym_ros lab1.py --ros-args -p controller:=<controller_type>

        # get the current pose
        self.get_logger().info("Subscribing to Odometry")
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.odom_sub # prevent unused variable warning
        
        self.get_logger().info("Publishing to Ackermann Drive")
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.cmd_pub # prevent unused variable warning
        
        # get the reference trajectory
        self.get_logger().info("Loading Reference Trajectory")
        self.ref_traj = np.load(os.path.join(get_package_share_directory('f1tenth_gym_ros'),
                                            'resource',
                                            'ref_traj.npy'))
        self.ref_traj # prevent unused variable warning
        
        # create a timer to publish the control input every 20ms
        self.get_logger().info("Creating Timer")
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.timer # prevent unused variable warning
        
        self.pose = np.zeros(3)

        self.current_cross_track_error = 0
        self.current_along_track_error = 0
        self.cross_track_accumulated_error = 0
        self.along_track_accumulated_error = 0
        self.waypoint_index = 0

        self.moved = False

        # New code by students
        # Creating a dictionary log for later plotting the performance
        self.log = {}
        self.log['cross_track_error'] = []
        self.log['along_track_error'] = []
        self.log['steering_command'] = []
        self.log['speed_command'] = []
        self.log['robot_x'] = []
        self.log['robot_y'] = []
        self.log['traj_x'] = []
        self.log['traj_y'] = []

        self.along_track_error_integral = 0
        self.cross_track_error_integral = 0
        self.along_track_error_derivative = 0
        self.cross_track_error_derivative = 0
        self.prev_along_track_error = 0
        self.prev_cross_track_error = 0
        self.dt = 0.5
        self.theta_ref = 0
        self.prev_pose = np.zeros(3)
        self.velocity = 0
        self.last_steering_command = 0
        # End of New code by students
    
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

        #### DEBUG PRINTS ####
        print(f'\nx = {x}, y = {y}, theta = {np.rad2deg(theta)}')
        print(f'velocity = {self.velocity}')
        print(f'x_ref = {x_ref}, y_ref = {y_ref}, theta_ref = {np.rad2deg(self.theta_ref)}')
        print(f'at_der =  {self.along_track_error_derivative}, eat_int = {self.along_track_error_integral}')
        print(f'ect_der =  {self.cross_track_error_derivative}, ect_int = {self.cross_track_error_integral}')

        # log the accumulated error to screen and internally to be printed at the end of the run
        self.get_logger().info("Cross Track Error: " + str(cross_track_error))
        self.get_logger().info("Along Track Error: " + str(along_track_error))
        self.cross_track_accumulated_error += abs(cross_track_error)
        self.along_track_accumulated_error += abs(along_track_error)


    def odom_callback(self, msg):
        # get the current pose
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = euler.quat2euler([q.w, q.x, q.y, q.z])

        if not self.moved and (x < -1 and y > 3):
            self.moved = True
        elif self.moved and x > 0:
            raise EndLap

        self.pose = np.array([x, y, yaw])
        
    def timer_callback(self):
        self.log_accumulated_error()
        
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
            u = self.optimal_control(self.pose)
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

        #### DEBUG PRINTS ####
        print(f'steering_command = {u[0]}, speed_command = {u[1]}')

        # Update log
        self.last_steering_command = u[0]
        self.log['cross_track_error'].append(self.current_cross_track_error)
        self.log['along_track_error'].append(self.current_along_track_error)
        self.log['steering_command'].append(u[0])
        self.log['speed_command'].append(u[1])
        self.log['robot_x'].append(self.pose[0])
        self.log['robot_y'].append(self.pose[1])
        self.log['traj_x'].append(self.ref_traj[self.waypoint_index % len(self.ref_traj)][0])
        self.log['traj_y'].append(self.ref_traj[self.waypoint_index % len(self.ref_traj)][1])

    def pid_control(self, pose):
        theta_measured = pose[2]
        wheelbase = 0.3302
        theta_next = theta_measured + self.velocity / wheelbase * np.tan(self.last_steering_command) * self.dt
        theta_error = theta_next - self.theta_ref

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Cross-track PID control
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        K_ct_p = 1
        K_ct_d = 2
        K_ct_i = 0
        steering_angle = -(K_ct_p * self.current_cross_track_error +
                           K_ct_d * self.velocity * np.sin(theta_error) +
                           K_ct_i * self.cross_track_accumulated_error)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Along-track PID control
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        K_at_p = 5
        K_at_d = 0
        K_at_i = 0
        speed = -(K_at_p * self.current_along_track_error +
                          K_at_d * self.velocity * np.cos(theta_error) +
                          K_at_i * self.along_track_accumulated_error)

        # Limiting the speed commands
        speed_cut_off = 0.5
        if speed > speed_cut_off:
            speed = speed_cut_off
        elif speed < -speed_cut_off:
            speed = -speed_cut_off

        return np.array([steering_angle, speed])

    def pid_unicycle_control(self, pose):
        theta_measured = pose[2]
        theta_next = theta_measured + self.last_steering_command * self.dt
        theta_error = theta_next - self.theta_ref

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Cross-track PID control
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        K_ct_p = 1
        K_ct_d = 2
        K_ct_i = 0
        steering_angle = -(K_ct_p * self.current_cross_track_error +
                                   K_ct_d * self.velocity * np.sin(theta_error) +
                                   K_ct_i * self.cross_track_accumulated_error)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Along-track PID control
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        K_at_p = 5
        K_at_d = 0
        K_at_i = 0
        speed = -(K_at_p * self.current_along_track_error +
                          K_at_d * self.velocity * np.cos(theta_error) +
                          K_at_i * self.along_track_accumulated_error)

        # Limiting the speed commands
        speed_cut_off = 0.5
        if speed > speed_cut_off:
            speed = speed_cut_off
        elif speed < -speed_cut_off:
            speed = -speed_cut_off

        return np.array([steering_angle, speed])
        #### END OF YOUR CODE ####

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
        #### YOUR CODE HERE ####
        
        
        # return np.array([steering_angle, speed])
        #### END OF YOUR CODE ####
        raise NotImplementedError
        

class EndLap(Exception):
    # this exception is raised when the car crosses the finish line
    pass

def export_log_to_json(log):
    with open("log.json", "w") as write_file:
        json.dump(log, write_file)

def main(args=None):
    rclpy.init()

    lab1 = Lab1(controller_type=sys.argv[1])

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

        # Logging the final accumulated errors
        lab1.log['acc_cross_track_error'] = lab1.cross_track_accumulated_error
        lab1.log['acc_along_track_error'] = lab1.along_track_accumulated_error
        lab1.log['lap_time'] = tock - tick
        export_log_to_json(lab1.log)

    lab1.destroy_node()
    rclpy.shutdown()
    
    
if __name__ == '__main__':
    controller_type = sys.argv[1]
    main(controller_type)