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
    ####### your code goes here #######
    # TODO: transform configuration to workspace bounding box
    
    # TODO: overlay workspace bounding box on map (creating borders for collision search in the next step)
    
    # TODO: check for collisions by looking inside the bounding box on the map if there are values greater than 0
    
    ##################################

    # Compute robot's bounding box corners in respect tot map coordinates
    robot_bb_height = 2
    robot_bb_width = 2

    bb_ne_x = np.ceil(x + robot_bb_width/2)
    bb_sw_x = np.floor(x - robot_bb_width / 2)
    bb_ne_y = np.ceil(y + robot_bb_height / 2)
    bb_sw_y = np.floor(y - robot_bb_height / 2)

    ## x = np.array(linspace([bb_ne_x]:[bb_sw_x]))

    # Computing bounding box in indices
    y_map, x_map = pose2map_coordinates(map_resolution, origin_x, origin_y, x, y)

    # Overlay the robot's bounding box on map and check for obstacles existence
    map_arr[y_map, x_map].any()


    raise NotImplementedError


def sample_configuration(map_arr, map_hight, map_width, map_resolution, origin_x, origin_y, n_points_to_sample=2000, dim=2):
    ####### your code goes here #######
    bbox_up = 0
    bbox_down = 0
    bbox_left = 0
    bbox_right = 0  # bounding box coordinates

    for i in range(map_hight):
        for j in range(map_width):
            if (map_arr[i][j] == False):
                bbox_up = i
                break
        if (map_arr[i][j] == False):
            break
    print(f'bbox_up = {bbox_up}')
    for i in range(map_hight-1, 0, -1):
        for j in range(map_width):
            if (map_arr[i][j] == False):
                bbox_down = i
                break
        if (map_arr[i][j] == False):
            break
    print(f'bbox_down = {bbox_down}')
    for j in range(map_width-1, 0, -1):
        for i in range(map_width):
            if (map_arr[i][j] == False):
                bbox_right = j
                break
        if (map_arr[i][j] == False):
            break
    print(f'bbox_right = {bbox_right}')
    for j in range(map_width):
        for i in range(map_width):
            if (map_arr[i][j] == False):
                bbox_left = j
                break
        if (map_arr[i][j] == False):
            break
    print(f'bbox_left = {bbox_left}')
    ##################################
    #raise NotImplementedError


def create_prm_traj(map_file):
    prm_traj = []
    mid_points = np.array([[0,0,0],
                           [9.5,4.5,np.pi/2],
                           [0,8.5,np.pi],
                           [-13.5,4.5,-np.pi/2]])
    map_arr, map_hight, map_width, map_resolution, origin_x, origin_y = load_map_and_metadata(map_file)

    ####### your code goes here #######
    # TODO: load the map and metadata
    sample_configuration(map_arr, map_hight, map_width, map_resolution, origin_x, origin_y)
    # TODO: create PRM graph
    
    # TODO: create PRM trajectory (x,y) saving it to prm_traj list
    
    ##################################
    
    prm_traj = np.concatenate(prm_traj, axis=0)
    np.save(os.path.join(pathlib.Path(__file__).parent.resolve().parent.resolve(),'resource/prm_traj.npy'), prm_traj)


def sample_conrol_inputs(number_of_samples=10):
    ####### your code goes here #######
    
    ##################################
    raise NotImplementedError


def forward_simulation_of_kineamtic_model(x, y, theta, v, delta, dt=0.5):
    ####### your code goes here #######
    
    
    ##################################
    raise NotImplementedError
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
