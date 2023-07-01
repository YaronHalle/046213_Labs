import rclpy
from rclpy.node import Node
import sys
import os
from PIL import Image
import yaml
import time

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Quaternion
from ackermann_msgs.msg import AckermannDriveStamped

from ament_index_python.packages import get_package_share_directory

import numpy as np
from numpy import cos, sin, tan, pi
from transforms3d import euler


def load_map_and_metadata(map_file, only_borders=False):
    # load the map from the map_file
    map_img = Image.open(map_file).transpose(Image.FLIP_TOP_BOTTOM)
    map_arr = np.array(map_img, dtype=np.uint8)
    threshold = 220
    if only_borders:
        threshold = 1
    map_arr[map_arr >= threshold] = 1
    map_arr[map_arr < threshold] = 0
    mask = np.ones(map_arr.shape, dtype=bool)
    map_arr = map_arr ^ mask
    map_arr = map_arr.astype(bool)
    map_hight = map_arr.shape[0]
    map_width = map_arr.shape[1]
    # load the map dimentions and resolution from yaml file
    with open(map_file.replace('.png', '.yaml'), 'r') as f:
        try:
            map_metadata = yaml.safe_load(f)
            map_resolution = map_metadata['resolution']
            map_origin = map_metadata['origin']
        except yaml.YAMLError as exc:
            print(exc)
    origin_x = map_origin[0]
    origin_y = map_origin[1]
        
    return map_arr, map_hight, map_width, map_resolution, origin_x, origin_y


def pose2map_coordinates(map_resolution, origin_x, origin_y, x, y):
    if isinstance(x, np.ndarray):
        x_map = ((x - origin_x) / map_resolution).astype(int)
        y_map = ((y - origin_y) / map_resolution).astype(int)
        return y_map, x_map
    else:
        x_map = int((x - origin_x) / map_resolution)
        y_map = int((y - origin_y) / map_resolution)
        return y_map, x_map


def map2pose_coordinates(map_resolution, origin_x, origin_y, x_map, y_map):
    x = x_map * map_resolution + origin_x
    y = y_map * map_resolution + origin_y
    return x, y


def forward_simulation_of_kineamtic_model(x, y, theta, v, delta, dt):
    d = 0.3302
    x_new = x + v * cos(theta) * dt
    y_new = y + v * sin(theta) * dt
    theta_new = theta + (v/d) * tan(delta) * dt
    return x_new, y_new, theta_new


# this node: 
# 1. creates odomotry data from laser scan data
# 2. uses the odometry data in a EKF to estimate the pose of the car and publish it on the /ekf_pose topic
class Lab3(Node):
    def __init__(self, map_file: str = 'levine.png'):
        super().__init__('lab3')
        self.get_logger().info("Lab 3 Node has been started")
        
        # subscribe to the scan topic
        self.get_logger().info("Subscribing to Scan")
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.scan_sub # prevent unused variable warning
        
        self.curr_scan = None
        self.prev_scan = None
        
        self.get_logger().info("Subscribing to Odom")
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.odom_sub # prevent unused variable warning
        
        self.tmp_pose = np.zeros(3)
        
        self.laser_pose = np.zeros(3)
        self.laser_covariance = np.zeros((3,3))
        
        # subscribe to the drive topic for the commands
        self.get_logger().info("Subscribing to Ackermann Drive")
        self.cmd_sub = self.create_subscription(AckermannDriveStamped, '/drive', self.cmd_callback, 10)
        self.cmd_sub # prevent unused variable warning
        self.cmd = np.zeros(2)
        
        # publish the estimated pose on the /ekf_pose topic
        self.get_logger().info("Publishing to EKF Pose")
        self.ekf_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/ekf_pose', 10)
        self.ekf_pose_pub # prevent unused variable warning
        
        # create timer to run the EKF every 10ms
        self.get_logger().info("Creating Timer")
        self.timer = self.create_timer(0.01, self.timer_callback)
        self.timer # prevent unused variable warning
        self.dt = 0.01

        # load map and metadata
        self.get_logger().info("Loading Map")
        map_path = os.path.join(get_package_share_directory('f1tenth_gym_ros'), 'maps', map_file)
        self.map_arr, self.map_hight, self.map_width, self.map_resolution, self.origin_x, self.origin_y = load_map_and_metadata(map_path, only_borders=True)
        self.top_y = np.where(self.map_arr == 1)[0].max()
        self.top_x = np.where(self.map_arr == 1)[1].max()
        self.bottom_x = np.where(self.map_arr == 1)[1].min()
        self.bottom_y = np.where(self.map_arr == 1)[0].min()
        
        self.pose = np.zeros(3)
        self.P = np.eye(3)*0.1

        self.is_first = True
        self.est_velocity_cmd = 0
        self.est_steering_angle_cmd = 0
        self.last_measrued_pose = np.zeros(3)
        self.last_time = self.get_clock().now().nanoseconds * 1e-9
        
    def scan_callback(self, msg):
        # if self.prev_scan != None and self.curr_scan != None and self.prev_scan.ranges != self.curr_scan.ranges:
        #     print("new scan, who dis?")
        self.prev_scan = self.curr_scan
        self.curr_scan = msg
        
    def cmd_callback(self, msg):
        self.cmd[0] = msg.drive.speed
        self.cmd[1] = msg.drive.steering_angle

    def odom_callback(self, msg):
        self.tmp_pose[0] = msg.pose.pose.position.x
        self.tmp_pose[1] = msg.pose.pose.position.y
        q = np.array([msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z])
        self.tmp_pose[2] = euler.quat2euler(q)[2]
        
    def _scan_to_odom(self, scan):
        if scan is None:
            self.get_logger().info("scan not ready")
            return
        # transform scan ranges to x,y points in the lidar frame
        xs,ys = [],[]
        for i, r in enumerate(scan.ranges):
            if r > scan.range_min and r < scan.range_max:
                theta = scan.angle_min + i * scan.angle_increment
                xs.append(r * np.cos(theta))
                ys.append(r * np.sin(theta))
        xs = np.array(xs)
        ys = np.array(ys)
        
        # transform x,y points in the lidar frame to x,y points in the base_link frame
        scan_locations = np.array([xs, ys]).T
        
        pose = self.tmp_pose
        
        # create a 3d histogram of all possible x,y,theta values and fill with the number of points in the scan data matching the map in that configuration of the car
        histogram = np.zeros((self.top_x - self.bottom_x, self.top_y - self.bottom_y, 360), dtype=np.uint16)
        y_in_map, x_in_map = pose2map_coordinates(self.map_resolution, self.origin_x, self.origin_y, self.pose[0], self.pose[1])
        theta_deg = int(self.pose[2] * 180 / np.pi) + 180
        for x in range(x_in_map-10, x_in_map+10):
            for y in range(y_in_map-10, y_in_map+10):
                for theta in range(theta_deg-20, theta_deg+20):
                    # count the number of points in the scan data that match the map in the current configuration of the car
                    histogram[x-self.bottom_x,y-self.bottom_y,theta%360] = self._count_matching_points_in_scan(scan_locations, x, y, theta-180)

        # find the covariance of the x,y,theta configuration based on the histogram data that corresponds to the maximum value in the histogram and return it as the covariance of the EKF
        covariance = np.zeros((3,3))
        xs, ys, thetas = [], [], []
        for x in range(self.top_x - self.bottom_x):
            xs.append(np.sum(histogram[x,:,:]))
        for y in range(self.top_y - self.bottom_y):
            ys.append(np.sum(histogram[:,y,:]))
        for theta in range(360):
            thetas.append(np.sum(histogram[:,:,theta]))
            
        # normalize to get probability distribution
        xs = np.array(xs) / np.sum(xs)
        ys = np.array(ys) / np.sum(ys)
        thetas = np.array(thetas) / np.sum(thetas)
        
        max_len = max(len(xs), len(ys), len(thetas))
        padded_xs = np.pad(xs, ((max_len - len(xs))//2, (max_len - len(xs))//2 + (max_len - len(xs))%2), 'constant', constant_values=(0))
        padded_ys = np.pad(ys, ((max_len - len(ys))//2, (max_len - len(ys))//2 + (max_len - len(ys))%2), 'constant', constant_values=(0))
        padded_thetas = np.pad(thetas, ((max_len - len(thetas))//2, (max_len - len(thetas))//2 + (max_len - len(thetas))%2), 'constant', constant_values=(0))
        
        covariance[0,0] = np.var(xs)
        covariance[1,1] = np.var(ys)
        covariance[2,2] = np.var(thetas)
        covariance[0,1] = np.cov(padded_xs, padded_ys)[0,1]
        covariance[0,2] = np.cov(padded_xs, padded_thetas)[0,1]
        covariance[1,2] = np.cov(padded_ys, padded_thetas)[0,1]
        covariance[1,0] = covariance[0,1]
        covariance[2,0] = covariance[0,2]
        covariance[2,1] = covariance[1,2]
                    
        self.laser_pose = pose
        self.laser_covariance = covariance
        
    def _count_matching_points_in_scan(self, scan_locations_in_base_link, x, y, theta):
        x_in_pose, y_in_pose = map2pose_coordinates(self.map_resolution, self.origin_x, self.origin_y, x, y)
        theta_rad = theta * np.pi / 180
        rotated_points = scan_locations_in_base_link @ np.array([[cos(theta_rad), -sin(theta_rad)],
                                                                 [sin(theta_rad), cos(theta_rad)]])
        moved_points = rotated_points + np.array([x_in_pose, y_in_pose])
        map_p_ys, map_p_xs = pose2map_coordinates(self.map_resolution, self.origin_x, self.origin_y, moved_points[:,0], moved_points[:,1])
        valid_idxs = np.where((map_p_xs >= 0) & (map_p_xs < self.map_width) & (map_p_ys >= 0) & (map_p_ys < self.map_hight))
        return np.sum(self.map_arr[map_p_ys[valid_idxs], map_p_xs[valid_idxs]])

    def compute_G_matrix(self, velocity, theta):
        G = np.zeros((3, 3))
        G[0, 2] = -velocity * np.sin(theta)
        G[1, 2] =  velocity * np.cos(theta)
        return G

    def compute_R_matrix(self):
        return np.diag((5, 5, 1))

    def estimate_commands(self, measured_pose, dt):
        '''
        Helper function for retrieving the velocity and steering angle commands out of the measured pose.
        :param measured_pose: Currently measured 3-element vector of the pose measurement [x,y,theta]
        :return: The function doesn't return anything, just updates the two class members: self.est_velocity_cmd
        and self.est_steering_angle_cmd
        '''
        # Defining the car's wheelbase
        d = 0.3302

        # Computing the estimated velocity
        delta_x = measured_pose[0] - self.last_measrued_pose[0]
        delta_y = measured_pose[1] - self.last_measrued_pose[1]
        self.est_velocity_cmd = np.sqrt(delta_x ** 2 + delta_y ** 2) / dt

        # Computing the estimated steering angle using differential flatness delta=tan-1(d*theta_dot/v)
        theta_dot = (measured_pose[2] - self.last_measrued_pose[2]) / dt

        # Avoiding dividing by zero
        if abs(self.est_velocity_cmd) > 1e-3:
            self.est_steering_angle_cmd = np.arctan(d * theta_dot / self.est_velocity_cmd)
        else:
            self.est_steering_angle_cmd = 0

        # Storing the last measured pose for next cycle
        self.last_measrued_pose = np.copy(measured_pose)

    def timer_callback(self):
        self._scan_to_odom(self.curr_scan)
        measured_pose = self.laser_pose
        measured_covariance = self.laser_covariance

        ########## Begin of EKF ##########
        # Counting the actual dt since last EKF invocation
        current_time = self.get_clock().now().nanoseconds * 1e-9
        delta_time = current_time - self.last_time
        self.last_time = current_time

        # Setting x(t-1): On first iteration taking the direct measurement,
        # else using EKF's previous step estimation.
        if self.is_first:
            self.is_first = False
            Xprev = measured_pose
            self.last_measrued_pose = np.copy(measured_pose)
        else:
            Xprev = self.pose

        # Updating the commands estimation
        self.est_velocity_cmd = self.cmd[0]
        self.est_steering_angle_cmd = self.cmd[1]
        # self.estimate_commands(measured_pose, delta_time)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Matrices definitions
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        x = Xprev[0]
        y = Xprev[1]
        theta = Xprev[2]
        velocity = self.est_velocity_cmd
        delta = self.est_steering_angle_cmd

        G = self.compute_G_matrix(velocity, theta)
        H = np.eye(3)
        R = self.compute_R_matrix()
        z = measured_pose
        Q = measured_covariance

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Prediction Step
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Xpredict = forward_simulation_of_kineamtic_model(x, y, theta, velocity, delta, delta_time)
        Ppredict = G @ self.P @ G.T + R

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Computing the Kalman gain
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        K = Ppredict @ H.T @ np.linalg.pinv(H @ Ppredict @ H.T + Q)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update step
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Xnext = Xpredict + K @ (z - Xprev)  # h = Xprev since we have direct measurement of all states variables
        Pnext = (np.eye(3) - K @ H) @ Ppredict

        # TODO remove debugging
        Xnext = measured_pose
        Pnext = measured_covariance

        pose = Xnext
        covariance = Pnext

        # Debug printing
        print('------------------------------------------------------------------------------------------------------')
        print(f'EKF DeltaTime = {delta_time}')
        print(f'Laser: X=({round(measured_pose[0], 2)},{round(measured_pose[1], 2)},{round(measured_pose[2], 2)})\nP={measured_covariance}')
        print(f'CMDs: ({round(self.est_velocity_cmd, 2)},{round(self.est_steering_angle_cmd, 2)})')
        print(f'Xpredict: X=({round(Xpredict[0], 2)},{round(Xpredict[1], 2)},{round(Xpredict[2], 2)})')
        print(f'Ppredict: \n{Ppredict}')
        print(f'EKF: X=({round(Xnext[0],2)},{round(Xnext[1],2)},{round(Xnext[2],2)})\nP=({Pnext})')
        ########## End of EKF ##########

        self.pose = pose
        self.P = covariance
        
        # publish the estimated pose on the /ekf_pose topic
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = self.pose[0]
        msg.pose.pose.position.y = self.pose[1]
        q = euler.euler2quat(0, 0, self.pose[2])
        msg.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        msg.pose.covariance[0] = self.P[0,0]
        msg.pose.covariance[1] = self.P[0,1]
        msg.pose.covariance[5] = self.P[0,2]
        msg.pose.covariance[6] = self.P[1,0]
        msg.pose.covariance[7] = self.P[1,1]
        msg.pose.covariance[11] = self.P[1,2]
        msg.pose.covariance[30] = self.P[2,0]
        msg.pose.covariance[31] = self.P[2,1]
        msg.pose.covariance[35] = self.P[2,2]
        
        self.ekf_pose_pub.publish(msg)
        

def main(args=None):
    rclpy.init()

    if len(sys.argv) < 2:
        lab3 = Lab3()
    else:
        lab3 = Lab3(map_file=sys.argv[1])

    try:
        rclpy.spin(lab3)
    except NotImplementedError:
        rclpy.logging.get_logger('lab3').info("You havn't implemented the EKF yet!")
    except KeyboardInterrupt:
        pass
    
    lab3.destroy_node()
    rclpy.shutdown()
    
    
if __name__ == '__main__':
    main() 
    