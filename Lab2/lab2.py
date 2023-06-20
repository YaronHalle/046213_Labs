import sys
import operator
import numpy as np
import scipy.spatial
from scipy.spatial import KDTree
from scipy import interpolate
from PIL import Image
import yaml
import os
import pathlib
from a_star import A_star
import time
from matplotlib import pyplot as plt

def plot_configs(tree, map_resolution, origin_x, origin_y, start_config, goal_config, plan=None):
    '''
    Helper function for plotting the Levine map with a planned trajectory on top of it.
    :param tree: The RRT tree. This argument can be left None.
    :param map_resolution: map parameter
    :param origin_x: map parameter
    :param origin_y: map parameter
    :param start_config: The starting configuration to be marked with a red point
    :param goal_config: The ending configuration to be marked with a red point
    :param plan: Nx2 matrix of (x,y) of the trajectory
    '''
    if tree is not None:
        col = []
        row = []
        for i in range(0, len(tree.vertices)):
            x = tree.vertices[i].state[0]
            y = tree.vertices[i].state[1]
            temp_row, temp_col = pose2map_coordinates(map_resolution, origin_x, origin_y, x, y)
            col.append(temp_col)
            row.append(temp_row)

    start_row, start_col = pose2map_coordinates(map_resolution, origin_x, origin_y, start_config[0], start_config[1])
    goal_row, goal_col = pose2map_coordinates(map_resolution, origin_x, origin_y, goal_config[0], goal_config[1])

    plt.figure()
    map_file = '../maps/levine.png'
    map_image = plt.imread(map_file)
    map_image = np.flip(map_image, 0)
    plt.imshow(map_image, cmap='gray', origin='lower')
    if tree is not None:
        plt.scatter(col, row, marker='.')
    plt.scatter(start_col, start_row, color='r', marker='o')
    plt.scatter(goal_col, goal_row, color='r', marker='o')
    plt.axis('off')

    if plan is not None:
        col = []
        row = []
        for i in range(0, plan.shape[0]):
            x = plan[i, 0]
            y = plan[i, 1]
            temp_row, temp_col = pose2map_coordinates(map_resolution, origin_x, origin_y, x, y)
            col.append(temp_col)
            row.append(temp_row)

        plt.plot(col, row, 'lime')

    plt.xlim(600, 1400)
    plt.ylim(800, 1400)
    plt.show()

def load_map_and_metadata(map_file):
    # load the map from the map_file
    map_img = Image.open(map_file).transpose(Image.FLIP_TOP_BOTTOM)
    map_arr = np.array(map_img, dtype=np.uint8)
    map_arr[map_arr < 220] = 1
    map_arr[map_arr >= 220] = 0
    map_arr = map_arr.astype(bool)
    map_hight = map_arr.shape[0]
    map_width = map_arr.shape[1]
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


def collision_check(map_arr, map_hight, map_width, map_resolution, origin_x, origin_y, x, y, theta=0):
    '''
    Collision check function for a specific (x,y) location of the robot
    :param map_arr: map parameter
    :param map_hight: map parameter
    :param map_width: map parameter
    :param map_resolution: map parameter
    :param origin_x: map parameter
    :param origin_y: map parameter
    :param x: The robot's x coordinate in [m]
    :param y: The robot's x coordinate in [m]
    :param theta: The robot's x coordinate in [theta]
    :return: True if collision was detected and False otherwise.
    '''

    # Compute the robot's bounding box corners in respect to map coordinates.
    # We assume that 3 times the car's wheelbase bounds the robot's geometry in 2D.
    wheelbase = 0.3302  # [meters]
    robot_bb_height = 2 * wheelbase  # [meters]
    robot_bb_width = 2 * wheelbase  # [meters]

    # Computing robot's bounding box coordinates
    bb_ne_x = x + robot_bb_width / 2
    bb_ne_y = y + robot_bb_height / 2
    bb_sw_x = x - robot_bb_width / 2
    bb_sw_y = y - robot_bb_height / 2

    # Transforming bounding box's coordinates to integer map indices
    bb_ne_row, bb_ne_col = pose2map_coordinates(map_resolution, origin_x, origin_y, bb_ne_x, bb_ne_y)
    bb_sw_row, bb_sw_col = pose2map_coordinates(map_resolution, origin_x, origin_y, bb_sw_x, bb_sw_y)

    # Overlay the robot's bounding box on map and check for obstacles existence
    return map_arr[bb_sw_row:bb_ne_row + 1, bb_sw_col:bb_ne_col + 1].any()


def sample_configuration(map_arr, map_hight, map_width, map_resolution, origin_x, origin_y, n_points_to_sample=2000,
                         dim=2):
    '''
    Returns 2-dimensional samples of (x,y) given a map.
    :param map_arr: map parameter
    :param map_hight: map parameter
    :param map_width: map parameter
    :param map_resolution: map parameter
    :param origin_x: map parameter
    :param origin_y: map parameter
    :param n_points_to_sample: Number of (x,y) pairs to be sampled
    :param dim: The samples dimensional, 2 is set as default, i.e. (x,y) without theta
    :return: Nx2 matrix comprised of (x,y) sampled configurations
    '''
    # Extracting all FALSE indices locations (FALSE=free space)
    free_space_ind = np.argwhere(map_arr == False)

    # Computing the enclosing rectangle boundary INDICES
    up_ind = max(free_space_ind[:, 0])
    down_ind = min(free_space_ind[:, 0])
    right_ind = max(free_space_ind[:, 1])
    left_ind = min(free_space_ind[:, 1])

    # Converting the boundary indices to real COORDINATES
    upright_x, upright_y = map2pose_coordinates(map_resolution, origin_x, origin_y, right_ind, up_ind)
    downleft_x, downleft_y = map2pose_coordinates(map_resolution, origin_x, origin_y, left_ind, down_ind)

    # Drawing random uniform samples
    samples = np.random.rand(n_points_to_sample, dim)

    # Scaling and shifting the samples to fit in the desired free space of the map
    samples[:, 0] = downleft_x + (upright_x - downleft_x) * samples[:, 0]
    samples[:, 1] = downleft_y + (upright_y - downleft_y) * samples[:, 1]

    return samples


def create_prm_traj(map_file):
    prm_traj = []
    mid_points = np.array([[0, 0],
                           [9.5, 4.5],
                           [0, 8.5],
                           [-13.5, 4.5]])
    map_arr, map_hight, map_width, map_resolution, origin_x, origin_y = load_map_and_metadata(map_file)

    ####### your code goes here #######

    # Create node list & collision check them
    samples = sample_configuration(map_arr, map_hight, map_width, map_resolution, origin_x, origin_y, 4000)
    test_x, test_y = map2pose_coordinates(map_resolution, origin_x, origin_y, 717, 1300)
    is_colliding = np.array([collision_check(map_arr, map_hight, map_width, map_resolution,
                                             origin_x, origin_y, sample[0], sample[1], 0) for sample in samples])
    free_space_samples = np.argwhere(is_colliding == False).flatten()

    samples = samples[free_space_samples]


    # Create edge list & collision check (knn & rnn, remove the least useful one)
    sample_tree = scipy.spatial.KDTree(samples)
    k = 8  # k-nn parameter
    r = 3  # r-nn parameter

    knn_edges = []
    rnn_edges = []
    for idx, node in enumerate(samples):
        d , knn_edge_sampling = sample_tree.query(x=node, k=k)  # returns list of k-nearest neighbours indexes
        rnn_edge_sampling = sample_tree.query_ball_point(x=node, r=r)
        for target_node_idx in knn_edge_sampling:  # for each node in the sampled list
            if target_node_idx == idx: continue # KDTree.query return will return same node as 1NN - disregard it.
            line = np.linspace(node, samples[target_node_idx], 10)  # connect starting node and original node in a line
            edge = (idx, target_node_idx)  # create the edge
            anti_edge = (target_node_idx, idx)  # create the reverse edge
            is_edge_colliding = np.array([collision_check(map_arr, map_hight, map_width, map_resolution,
                                             origin_x, origin_y, point[0], point[1], 0) for point in line]).any()  # check if any point in the line collides
            if(is_edge_colliding == False):
                knn_edges.append(edge)
                knn_edges.append(anti_edge)


        for target_node_idx in rnn_edge_sampling:  # same as knn - but for the rnn sampling
            line = np.linspace(node, samples[target_node_idx], 10)
            edge = (idx, target_node_idx)
            anti_edge = (target_node_idx, idx)
            is_edge_colliding = np.array([collision_check(map_arr, map_hight, map_width, map_resolution,
                                                          origin_x, origin_y, point[0], point[1], 0) for point in line]).any()  # check if any point in the line collides
            if (is_edge_colliding == False):
                rnn_edges.append(edge)
                rnn_edges.append(anti_edge)

    # Calculate costs for each edge
    knn_costs = {}
    for edge in knn_edges:
        edge_cost = np.linalg.norm(samples[edge[0]] - samples[edge[1]], ord = 2)
        knn_costs[edge[0], edge[1]] = edge_cost

    rnn_costs = {}
    for edge in rnn_edges:
        edge_cost = np.linalg.norm(samples[edge[0]] - samples[edge[1]], ord = 2)
        rnn_costs[edge[0], edge[1]] = edge_cost


    # Populate PRM graph with the previously calculated parameters
    knn_prm_graph = {'nodes': samples,
             'edges': knn_edges,
             'costs': knn_costs}

    rnn_prm_graph = {'nodes': samples,
                     'edges': rnn_edges,
                     'costs': rnn_costs}

    knn_astar = A_star(knn_prm_graph)
    rnn_astar = A_star(rnn_prm_graph)

    # TODO: create PRM trajectory (x,y) saving it to prm_traj list

    rnn_astar_traj = []
    knn_astar_traj = []

    for i in range(4):
        rnn_astar_traj = rnn_astar_traj + rnn_astar.a_star(mid_points[i], mid_points[(i+1)%4])
        knn_astar_traj = knn_astar_traj + knn_astar.a_star(mid_points[i], mid_points[(i+1)%4])

    if (len(rnn_astar_traj) < len(knn_astar_traj)):  # choose the shorter option of the two
        prm_traj = np.array(rnn_astar_traj)
    else:
        prm_traj = np.array(knn_astar_traj)

    plot_configs(None,map_resolution,origin_x,origin_y,prm_traj[0],prm_traj[-1],prm_traj)

    np.save(os.path.join(pathlib.Path(__file__).parent.resolve().parent.resolve(), 'resource/prm_traj.npy'), np.array(prm_traj))


def sample_control_inputs():
    '''
    Helper function for sampling a single controls input.
    :return: array of [velocity_command [m/s], steering angle command [rad], time duration [sec]]
    '''
    # Predefined control cut-off limits
    velocity_min = 0.5  # [m/s]
    velocity_max = 1  # [m/s]
    steering_angle_min = -0.3  # [rad]
    steering_angle_max = 0.3  # [rad]
    min_command_duration = 0.5  # [sec]
    max_command_duration = 5  # [sec]

    # Drawing random control commands and commands duration
    rand_v = np.random.uniform(velocity_min, velocity_max)
    rand_str_angle = np.random.uniform(steering_angle_min, steering_angle_max)
    rand_t = np.random.uniform(min_command_duration, max_command_duration)

    return rand_v, rand_str_angle, rand_t


def forward_simulation_of_kineamtic_model(state, start_time, v, delta, duration=0.5):
    '''
    Helper function for computing the next X state vector of the non-linear Ackerman car model
    using the current state and command vector.
    Inputs:
    @param state Current state comprised of 3 elements vector: [x,y,theta]
    @param start_time State start time [sec]
    @param v Current velocity command [m/s]
    @param delta Current Steering angle command [rad]
    @param dt Integration time step [sec]
    Outputs:
    @param new_states: a matrix of Nx3 propagated states including the final resultant state after dt duration of time
    '''
    time_res = 0.1
    time_samples = np.hstack((np.arange(start_time, start_time + duration, time_res), start_time + duration))
    velocity = v
    steering_angle = delta
    wheelbase = 0.3302

    # Creating empty matrix of propagated states, initializing with the input state
    new_states = np.zeros((time_samples.size, 3))
    new_states[0, :] = state

    # Propagating states forward using time integration
    for i in range(1, len(time_samples)):
        delta_t = time_samples[i] - time_samples[i - 1]
        new_states[i, 0] = new_states[i - 1, 0] + velocity * np.cos(new_states[i - 1, 2]) * delta_t
        new_states[i, 1] = new_states[i - 1, 1] + velocity * np.sin(new_states[i - 1, 2]) * delta_t
        new_states[i, 2] = new_states[i - 1, 2] + velocity / wheelbase * np.tan(steering_angle) * delta_t

    # Removing the first new_states element since it is redundant
    new_states = new_states[1:, :]
    time_stamps = time_samples[1:]

    return new_states, time_stamps


def compute_distance(first_config, second_config):
    '''
    Computes the distance between two configurations.
    :param first_config: (x,y,theta) configuration
    :param second_config: (x,y,theta) configuration
    :return: The Euclidean distance between the two (x,y) coordinates
    '''
    return np.linalg.norm(second_config[0:2] - first_config[0:2])


def are_configs_close(first_config, second_config):
    '''
    Helper function to determine if two configurations are "close" enough.
    :param first_config: The first 3-dimensional configuration (x,y,theta)
    :param second_config: The second 3-dimensional configuration (x,y,theta)
    :return: True if the two configurations are close enough and False otherwise
    '''
    geo_dist_thr = 0.5  # [m]
    angular_dist_thr = np.deg2rad(3)  # [rad]

    # Computing the geometric distance between two (x,y) coordinates of the configurations
    geo_dist = np.linalg.norm(second_config[0:2] - first_config[0:2])

    # Computing the angular distance between theta_1 and theta_2 of the two configurations
    ang_offset = (first_config[2] - second_config[2]) % (2 * np.pi)
    if ang_offset >= np.pi:
        ang_offset -= 2 * np.pi
    angular_dist = abs(ang_offset)

    return (geo_dist < geo_dist_thr) and (angular_dist < angular_dist_thr)


def is_edge_collision_free(states, map_arr, map_hight, map_width, map_resolution, origin_x,
                           origin_y):
    '''
    Helper function to decide if an edge is collision free
    :param states: A matrix Nx3 of states (each state is 3 elements: [x,y,theta]) to be checked for collision
    :param map_arr: map parameter
    :param map_hight: map parameter
    :param map_width: map parameter
    :param map_resolution: map parameter
    :param origin_x: map parameter
    :param origin_y: map parameter
    :return: True if edge is collision-free and False otherwise.
    '''
    for state in states:
        if collision_check(map_arr, map_hight, map_width, map_resolution, origin_x, origin_y, state[0], state[1]):
            # Collision was detected, returning False and breaking function
            return False

    # If got here, the edge is collision-free
    return True


def kino_rrt(start_config, start_time, goal_config, x_limit, y_limit,
             map_arr, map_hight, map_width, map_resolution, origin_x, origin_y):
    '''
    Performs a Kinodynamic-RRT search given start and goal configurations
    :param start_config: The start configuration given as (x,y,theta)
    :param start_time: Starting time in [sec] of the start_config state
    :param goal_config: The goal configuration given as (x,y,theta)
    :param x_limit: x-coords. boundaries given as [xmin,xmax] array
    :param y_limit: y-coords. boundaries given as [ymin,ymax] array
    :param map_arr: map parameter
    :param map_hight: map parameter
    :param map_width: map parameter
    :param map_resolution: map parameter
    :param origin_x: map parameter
    :param origin_y: map parameter
    :return: positions: An Nx2 matrix comprised of (x,y) coordinates to traverse.
    :return: times: the corresponding time stamps of each element in positions
    '''

    rrt_start_time = time.time()

    # Initializations before search start
    path_to_goal_found = False

    # Creating the RRT tree data structure
    tree = RRTTree()

    # Seeding the tree from the start state
    tree.add_vertex(start_config, start_time, np.hstack((start_time, start_config)))

    # Main search loop
    while not path_to_goal_found:
        # Sample a new random state
        rand_x = np.random.uniform(x_limit[0], x_limit[1])
        rand_y = np.random.uniform(y_limit[0], y_limit[1])
        rand_state = [rand_x, rand_y, 0]

        # Find the nearest neighbor from the previously explored states
        sid, nn_state = tree.get_nearest_state(rand_state)

        # Sampling a control sequence and time duration
        rand_v, rand_str_angle, rand_dt = sample_control_inputs()

        # Propagating next states starting from the nearest state
        mid_states, time_stamps = forward_simulation_of_kineamtic_model(nn_state.state, nn_state.time,
                                                                        rand_v, rand_str_angle, rand_dt)

        # Check if candidate new edge is collision-free
        is_valid_edge = is_edge_collision_free(mid_states, map_arr, map_hight, map_width,
                                               map_resolution, origin_x, origin_y)
        if is_valid_edge:
            # Add new vertex to the RRT tree
            new_state = mid_states[-1, :]
            new_time = time_stamps[-1]
            hr_states = np.zeros((mid_states.shape[0], 4))
            hr_states[:, 0] = time_stamps
            hr_states[:, 1:4] = mid_states

            # Adding the new vertex to the RRT tree
            eid = tree.add_vertex(new_state, new_time, hr_states)

            # Add new edge to the RRT tree
            tree.add_edge(sid, eid)

            # Check if the goal state's vicinity was approached. If so, the search can be terminated.
            path_to_goal_found = are_configs_close(new_state, goal_config)

            # Printing the status of the goal's closest state
            _, temp_state = tree.get_nearest_state(goal_config)
            print(f'Goal is at ({goal_config[0]},{goal_config[1]}), closest state is at ({temp_state.state[0]},{temp_state.state[1]}), Vertices # = {len(tree.vertices)}')

    # Once got here the goal was reached. Populating the plan list by back-tracking the path
    current_vert_id = len(tree.vertices) - 1
    plan = np.array(tree.vertices[current_vert_id].high_res_states)
    while current_vert_id > 0:
        current_vert_id = tree.edges[current_vert_id]
        plan = np.vstack((tree.vertices[current_vert_id].high_res_states, plan))

    # Report search time
    print('Total time: {:.2f} [sec]'.format(time.time() - rrt_start_time))

    return plan


def create_kino_rrt_traj(map_file):
    kino_rrt_traj = np.zeros((0, 2))
    mid_points = np.array([[0, 0, 0],
                           [9.5, 4.5, np.pi / 2],
                           [0, 8.5, np.pi],
                           [-13.5, 4.5, -np.pi / 2],
                           [0, 0, 0]])

    # Loading the map and metadata
    map_arr, map_hight, map_width, map_resolution, origin_x, origin_y = load_map_and_metadata(map_file)

    # Planning iteratively through all mid-points segments
    for i in range(0, mid_points.shape[0] - 1):
        # Determining the start and goal configurations
        goal_config = mid_points[i + 1, :]
        if i == 0:
            # Setting time to zero on first segment
            start_time = 0
            start_config = mid_points[i, :]
        else:
            # Next, taking the previous segment last time and configuration as the new segment start time and configuration
            start_config = path[-1, 1:]
            start_time = path[-1, 0]

        # Focusing the x and y samples to reside only in the relevant rectangle defined by the mid_points locations
        x_limit = [min(mid_points[i, 0], mid_points[i + 1, 0]), max(mid_points[i, 0], mid_points[i + 1, 0])]
        y_limit = [min(mid_points[i, 1], mid_points[i + 1, 1]), max(mid_points[i, 1], mid_points[i + 1, 1])]

        # Computing a path for the current segment using KinoRRT
        path = kino_rrt(start_config, start_time, goal_config, x_limit, y_limit,
                 map_arr, map_hight, map_width, map_resolution, origin_x, origin_y)

        # Interpolating the path to match standard 0.5 [sec] time steps between consecutive locations
        f_x = interpolate.interp1d(path[:, 0], path[:, 1])
        f_y = interpolate.interp1d(path[:, 0], path[:, 2])
        time_samples = np.hstack((np.arange(path[0, 0], path[-1, 0], 0.5), path[-1, 0]))
        path_interp = np.array([f_x(time_samples), f_y(time_samples)]).transpose()

        # Concatenating the path with the general solution path computed so far. Taking only the x,y coordinates
        kino_rrt_traj = np.vstack((kino_rrt_traj, path_interp))

    # Plotting the final planned trajectory
    plot_configs(None, map_resolution, origin_x, origin_y, [0, 0], [0, 0], kino_rrt_traj)

    # Saving to file
    np.save(os.path.join(pathlib.Path(__file__).parent.resolve().parent.resolve(), 'resource/kino_rrt_traj.npy'),
            kino_rrt_traj)


class RRTTree(object):
    '''
    RRT Tree class
    '''
    def __init__(self):
        self.vertices = {}
        self.edges = {}

    def add_vertex(self, final_state, time, mid_states=None):
        '''
        Add a state to the tree.
        @param final_state: state to add to the tree
        @param time: the time of the new added state
        @param mid_states: Sequence of configurations that end up in the final_state configuration
        '''
        vid = len(self.vertices)
        self.vertices[vid] = RRTVertex(final_state, time, mid_states)
        return vid

    def add_edge(self, sid, eid):
        '''
        Adds an edge in the tree.
        @param sid start state ID
        @param eid end state ID
        '''
        self.edges[eid] = sid

    def get_nearest_state(self, state):
        '''
        Find the nearest vertex for the given state and returns its state index and state
        @param state Sampled state.
        '''
        # compute distances from all vertices
        dists = []
        for _, vertex in self.vertices.items():
            dists.append(compute_distance(state, vertex.state))

        # retrieve the id of the nearest vertex
        vid, _ = min(enumerate(dists), key=operator.itemgetter(1))

        return vid, self.vertices[vid]


class RRTVertex(object):
    '''
    RRT node class
    '''
    def __init__(self, state, time, hr_states):
        self.state = state
        self.time = time
        if hr_states is None:
            self.high_res_states = np.zeros((0, 4))
        else:
            self.high_res_states = np.copy(hr_states)


if __name__ == "__main__":
    map_file = '../maps/levine.png'
    create_prm_traj(map_file)
    create_kino_rrt_traj(map_file)