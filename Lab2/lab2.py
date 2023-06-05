import sys

import numpy as np
from scipy.spatial import KDTree
from PIL import Image
import yaml
import os
import pathlib
from a_star import A_star


def load_map_and_metadata(map_file):
    # load the map from the map_file
    map_img = Image.open(map_file).transpose(Image.FLIP_TOP_BOTTOM)
    map_arr = np.array(map_img, dtype=np.uint8)
    map_arr[map_arr < 220] = 1
    map_arr[map_arr >= 220] = 0
    map_arr = map_arr.astype(bool)
    map_hight = map_arr.shape[0]
    map_width = map_arr.shape[1]
    # TODO: load the map dimentions and resolution from yaml file
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
    x_map = int((x - origin_x) / map_resolution)
    y_map = int((y - origin_y) / map_resolution)
    return y_map, x_map

def map2pose_coordinates(map_resolution, origin_x, origin_y, x_map, y_map):
    x = x_map * map_resolution + origin_x
    y = y_map * map_resolution + origin_y
    return x, y

def collision_check(map_arr, map_hight, map_width, map_resolution, origin_x, origin_y, x, y, theta):
    '''
    Collision check function for a given robot's configuration.
    The functions returns TRUE if collision occurred and FALSE otherwise.
    '''
    ####### your code goes here #######
    # TODO: transform configuration to workspace bounding box
    
    # TODO: overlay workspace bounding box on map (creating borders for collision search in the next step)
    
    # TODO: check for collisions by looking inside the bounding box on the map if there are values greater than 0
    
    ##################################

    # Compute the robot's bounding box corners in respect to map coordinates.
    # We assume that 3 times the car's wheelbase bounds the robot's geometry in 2D.
    wheelbase = 0.3302  # [meters
    robot_bb_height = 3 * wheelbase  # [meters]
    robot_bb_width = 3 * wheelbase  # [meters]

    # Computing robot's bounding box coordinates
    bb_ne_x = int(np.ceil(x + robot_bb_width/2))
    bb_ne_y = int(np.ceil(y + robot_bb_height / 2))
    bb_sw_x = int(np.floor(x - robot_bb_width / 2))
    bb_sw_y = int(np.floor(y - robot_bb_height / 2))

    # Transforming bounding box's coordinates to integer map indices
    bb_ne_row, bb_ne_col = pose2map_coordinates(map_resolution, origin_x, origin_y, bb_ne_x, bb_ne_y)
    bb_sw_row, bb_sw_col = pose2map_coordinates(map_resolution, origin_x, origin_y, bb_sw_x, bb_sw_y)

    # Overlay the robot's bounding box on map and check for obstacles existence
    return map_arr[bb_sw_row:bb_ne_row + 1, bb_sw_col:bb_ne_col + 1].any()

def sample_configuration(map_arr, map_hight, map_width, map_resolution, origin_x, origin_y, n_points_to_sample=2000, dim=2):
    # Extracting all FALSE indices locations (FALSE=free space)
    free_space_ind = np.argwhere(map_arr == False)

    # Computing the enclosing rectangle boundary indices
    up_ind = max(free_space_ind[:, 0])
    down_ind = min(free_space_ind[:, 0])
    right_ind = max(free_space_ind[:, 1])
    left_ind = min(free_space_ind[:, 1])

    # Converting the boundary indices to real coordinates
    upright_x, upright_y = map2pose_coordinates(map_resolution, origin_x, origin_y, right_ind, up_ind)
    downleft_x, downleft_y = map2pose_coordinates(map_resolution, origin_x, origin_y, left_ind, down_ind)

    # Drawing the random samples
    samples = np.random.rand(n_points_to_sample, dim)

    # Calibrating the samples to fit in the free space of the map
    samples[:, 0] = downleft_x + (upright_x - downleft_x) * samples[:, 0]
    samples[:, 1] = downleft_y + (upright_y - downleft_y) * samples[:, 1]

    return samples

def create_prm_traj(map_file):
    prm_traj = []
    mid_points = np.array([[0,0,0],
                           [9.5,4.5,np.pi/2],
                           [0,8.5,np.pi],
                           [-13.5,4.5,-np.pi/2]])
    map_arr, map_hight, map_width, map_resolution, origin_x, origin_y = load_map_and_metadata(map_file)

    ####### your code goes here #######
    # TODO: load the map and metadata
    samples = sample_configuration(map_arr, map_hight, map_width, map_resolution, origin_x, origin_y)
    # TODO: create PRM graph
    
    # TODO: create PRM trajectory (x,y) saving it to prm_traj list
    
    ##################################
    
    prm_traj = np.concatenate(prm_traj, axis=0)
    np.save(os.path.join(pathlib.Path(__file__).parent.resolve().parent.resolve(),'resource/prm_traj.npy'), prm_traj)


def sample_control_inputs(number_of_samples=10):
    '''
    Helper function for sampling controls according to predefined control cut-off limits
    :param number_of_samples: Number of control samples to be drawn
    :return: Matrix number_of_samplesX2, first column is velocity and second column is the steering angle command
    '''
    # Predefined control cut-off limits
    velocity_min = -0.5 # [m/s]
    velocity_max = 0.5 # [m/s]
    steering_angle_min = -0.3 # [rad]
    steering_angle_max = 0.3 # [rad]

    # Drawing random samples.
    control_samples = np.random.rand(number_of_samples, 2)

    # Setting the velocity values
    control_samples[:, 0] = velocity_min + (velocity_max - velocity_min) * control_samples[:, 0]
    control_samples[:, 1] = steering_angle_min + (steering_angle_max - steering_angle_min) * control_samples[:, 1]

    return control_samples

def forward_simulation_of_kineamtic_model(x, y, theta, v, delta, dt=0.5):
    '''
    Helper function for computing the next X state vector of the non-linear Ackerman car model
    using the current state and command vector.
    Inputs:
    @param x Current x state [m]
    @param y Current y state [m]
    @param theta Current theta state [m]
    @param v Current velocity command [m/s]
    @param delta Current Steering angle command [rad]
    @param dt Integration time step [sec]
    Outputs:
    @param x_new Next state x
    @param y_new Next state y
    @param theta_new Next state theta
    '''
    velocity = v
    steering_angle = delta
    wheelbase = 0.3302

    x_new = x + velocity * np.cos(theta) * dt
    y_new = y + velocity * np.sin(theta) * dt
    theta_new = theta + velocity / wheelbase * np.tan(steering_angle) * dt

    return x_new, y_new, theta_new

def create_kino_rrt_traj(map_file):
    kino_rrt_traj = []
    mid_points = np.array([[0,0,0],
                           [9.5,4.5,np.pi/2],
                           [0,8.5,np.pi],
                           [-13.5,4.5,-np.pi/2]])
    map_arr, map_hight, map_width, map_resolution, origin_x, origin_y = load_map_and_metadata(map_file)
    ####### your code goes here #######
    # TODO: load the map and metadata
    
    # TODO: create RRT graph and find the path saving it to kino_rrt_traj list
    
    ##################################
    
    kino_rrt_traj = np.array(kino_rrt_traj)
    np.save(os.path.join(pathlib.Path(__file__).parent.resolve().parent.resolve(),'resource/kino_rrt_traj.npy'), kino_rrt_traj)


if __name__ == "__main__":
    map_file = '../maps/levine.png'
    create_prm_traj(map_file)
    create_kino_rrt_traj(map_file)
