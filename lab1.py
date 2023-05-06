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
        self.total_velocity = 0.0
        
        self.current_cross_track_error = 0
        self.current_along_track_error = 0
        self.cross_track_accumulated_error = 0
        self.along_track_accumulated_error = 0
        self.waypoint_index = 0

        # Creating a dictionary log for later plotting the performance
        self.log = {}
        self.log['cross_track_error'] = []
        self.log['along_track_error'] = []
        self.log['steering_command'] = []
        self.log['speed_command'] = []
        
        self.moved = False
    
    def get_ref_pos(self):
        # get the next waypoint in the reference trajectory based on the current time
        self.waypoint_index += 1
        waypoint = self.ref_traj[self.waypoint_index % len(self.ref_traj)]
        return waypoint

    def log_accumulated_error(self):
        next_ref_pos = np.array(self.get_ref_pos())
        prev_ref_pos = np.array(self.ref_traj[(self.waypoint_index - 1) % len(self.ref_traj)])
        x = self.pose[0]
        y = self.pose[1]
        theta = self.pose[2]
        x_ref = next_ref_pos[0]
        y_ref = next_ref_pos[1]

        # compute the cross track and along track error depending on the current pose and the next waypoint
        relative_trajectory_vec = next_ref_pos - prev_ref_pos
        theta_ref = self.compute_angle_of_vector(relative_trajectory_vec)
        cross_track_error = -np.sin(theta_ref) * (x - x_ref) + np.cos(theta_ref) * (y - y_ref)
        along_track_error =  np.cos(theta_ref) * (x - x_ref) + np.sin(theta_ref) * (y - y_ref)

        self.current_cross_track_error = cross_track_error
        self.current_along_track_error = along_track_error
        self.theta_error = (theta - theta_ref)

        # DEBUG
        print(f'Current pose   = (X = {x}, Y = {y})')
        print(f'Reference pose = (X = {x_ref}, Y = {y_ref})')
        print(f'Waypoint index = {self.waypoint_index}')
        print(f'Theta = {theta} , Theta_ref = {theta_ref}')

        # log the accumulated error to screen and internally to be printed at the end of the run
        self.get_logger().info("Cross Track Error: " + str(cross_track_error))
        self.get_logger().info("Along Track Error: " + str(along_track_error))
        self.cross_track_accumulated_error += abs(cross_track_error)
        self.along_track_accumulated_error += abs(along_track_error)

    def compute_angle_of_vector(self, vec):
        '''
        A utility function to compute the angle of a vector.
        @param vec Vector from previous to next trajectory point
        '''
        vec_norm = np.linalg.norm(vec)
        if vec_norm == 0:
            print('Warning! in compute_angle_of_vector: division by zero')
        vec = vec / vec_norm
        if vec[1] > 0:
            return np.arccos(vec[0])
        else:  # vec[1] <= 0
            return -np.arccos(vec[0])

    def odom_callback(self, msg):
        # get the current pose
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = euler.quat2euler([q.w, q.x, q.y, q.z])

        # get the current total linear velocity
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y

        if not self.moved and (x < -1 and y > 3):
            self.moved = True
        elif self.moved and x > 0:
            raise EndLap

        self.total_velocity = np.linalg.norm([vx, vy])
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

        # Update log
        self.log['cross_track_error'].append(self.current_cross_track_error)
        self.log['along_track_error'].append(self.current_along_track_error)
        self.log['steering_command'].append(u[0])
        self.log['speed_command'].append(u[1])

    def pid_control(self, pose):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Cross-track PID control
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        K_ct_p = 1
        K_ct_d = 3
        K_ct_i = 0
        steering_angle_command = -(K_ct_p * self.current_cross_track_error +
                                   K_ct_d * self.total_velocity * np.sin(self.theta_error) +
                                   K_ct_i * self.cross_track_accumulated_error)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Along-track PID control
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        K_at_p = 5
        K_at_d = 0
        K_at_i = 0
        speed_command = -(K_at_p * self.current_along_track_error +
                          K_at_d * self.total_velocity * np.cos(self.theta_error) +
                          K_at_i * self.along_track_accumulated_error)

        # Limiting the speed commands
        speed_cut_off = 0.5
        if speed_command > speed_cut_off :
            speed_command = speed_cut_off
        elif speed_command < -speed_cut_off :
            speed_command = -speed_cut_off

        print(f'PID_CONTROL: ThetaError = {self.theta_error}')
        print(f'PID_CONTROL: SteeringCommand = {steering_angle_command} , SpeedCommand = {speed_command}')
        return np.array([steering_angle_command, speed_command])

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

        export_log_to_json(lab1.log)

    lab1.destroy_node()
    rclpy.shutdown()
    
    
if __name__ == '__main__':
    controller_type = sys.argv[1]
    main(controller_type)