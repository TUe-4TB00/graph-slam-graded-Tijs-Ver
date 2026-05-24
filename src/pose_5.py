import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.``
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    # TODO: Initialize the optimizer 

    params = gtsam.LevenbergMarquardtParams()

    # TODO: Perform the optimization and print the result
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest sum of marginals.
    lowest_sum = float('inf') 
   
    for label, coords in pose_options.items():
        for lm_idx in [1, 2]:
            graph_new = gtsam.NonlinearFactorGraph(graph)
            estimate_new = gtsam.Values(initial_estimate)
            
            graph_new, estimate_new = add_pose(graph_new, estimate_new, coords)
            estimate_new = optimize(graph_new, estimate_new)
            graph_new = add_landmark_measurement(graph_new, estimate_new, coords, lm_idx)
            res_trial = optimize(graph_new, estimate_new)
            
            marginal = gtsam.Marginals(graph_new, res_trial)
            current_sum = marginal.marginalCovariance(L(lm_idx)).sum()
            
            if current_sum < lowest_sum:
                lowest_sum = current_sum
                best_pose = label
                best_landmark = lm_idx

    pose_5 = pose_options[best_pose]
    graph, initial_estimate = add_pose(graph, initial_estimate, pose_5)
    result = optimize(graph, initial_estimate)
    graph = add_landmark_measurement(graph, result, pose_5, best_landmark)
    result = optimize(graph, initial_estimate)

    # TODO: Calculate marginal covariances for the relevant variables and visualize the updated factor graph with covariances
    marginals_obj = gtsam.Marginals(graph, result)
    
    # The sum of the marginals for each landmark can be computed using marginals.marginalCovariance(L(x)).sum()
    sum_of_marginals = marginals_obj.marginalCovariance(L(1)).sum() + \
                       marginals_obj.marginalCovariance(L(2)).sum()
    return best_pose, best_landmark, sum_of_marginals


def minimize_errors(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest resulting error.
    best_pose = "a"      # chosen pose option
    best_landmark = 1    # chosen landmark (1 or 2)
    pose_5 = pose_options[best_pose]
    graph, initial_estimate = add_pose(graph, initial_estimate, pose_5)
    result = optimize(graph, initial_estimate)
    graph = add_landmark_measurement(graph, result, pose_5, best_landmark)
    result = optimize(graph, initial_estimate)

    # TODO: create a list of errors (each index corresponds to a pose) and add the error of each pose to the list
    list_of_errors = []
    # TODO: compute the sum of the errors and return it along with the best pose and landmark
    sum_of_errors = 0
    return best_pose, best_landmark, sum_of_errors 