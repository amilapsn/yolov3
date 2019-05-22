import numpy as np
from sklearn.externals import joblib


def find_lane(sorted_lane_list, centroid):
    """
    Given the centroid of the vehicle and list of lines seperating the lanes, outputs to which lane the vehicle belong
    :param sorted_lane_list: numpy array of lines separating the lanes shape : n x 2 x 2
    (number of points, two points, x and y coords)
    :param centroid: centroid of the vehicle bounding box [x,y]
    :return: lane ID
    """
    centroid = centroid.reshape(2,-1)
    centroid = np.repeat(centroid, sorted_lane_list.shape[0], axis=1)

    sorted_lane_list = sorted_lane_list.transpose([1,2,0])
    lane_count = find_sign(centroid, sorted_lane_list)

    return (np.sum(lane_count))


def find_sign(point, line):
    """
    Given a point and a line gives out True or false based on the side it is to the side
    :param point: point to lookup [x,y] x n
    :param line: line to be compared [[x1, y1],[x2,y2]] x n
    :return: False if same side as origin, True if opposite side
    """
    d = (point[0] - line[0,0]) * (line[0,1] - line[1,1]) + (point[1] - line[0,1]) * (line[1,0] - line[0,0])
    d0 = (0 - line[0,0]) * (line[0,1] - line[1,1]) + (0 - line[0,1]) * (line[1,0] - line[0,0])
    return d * d0 < 0


if __name__ == "__main__":
    calibration_params = joblib.load("../weights/calibration_output.joblib")
    sorted_lanes = calibration_params['lane_list']
    point = np.array([750,900])
    lane_ID = find_lane(sorted_lanes,point)
    print(lane_ID)