
import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate):
    t1 = gtsam.Pose2(0 ,0 ,np.deg2rad(45))
    m1 = gtsam.Pose2(2.0, 0, 0)
    t2 = gtsam.Pose2(0 ,0 ,np.deg2rad(45))
    
    
    
    odometry_delta = t1.compose(m1).compose(t2)
    # TODO: Add the odometry factor between X(4) and X(5) to the graph (BetweenFactorPose2)
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), odometry_delta, ODOMETRY_NOISE))

    # TODO: Based on the odometry, find the initial estimate for the pose of X(5) and add it to the graph
    
    new_pose_guess = gtsam.Pose2(4 + 2 * math.cos(np.deg2rad(45)), 2 * math.sin(np.deg2rad(45)), np.deg2rad(90))
    initial_estimate.insert(X(4), new_pose_guess)
    return graph, initial_estimate