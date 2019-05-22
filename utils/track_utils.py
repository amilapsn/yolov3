import cv2
import numpy as np
import time
import torch
from utils.utils import bbox_iou
from utils.path_visualization_utils import draw_path
from utils.lane_speed_utils import find_lane
from sklearn.externals import joblib

NO_OF_CLASSES = 7
DISTANCE_THRESHOLD = 50
OLD_LIMIT = 20
MIN_PRESENT = 10

calibration_params = joblib.load("/home/madshan/video-analytics/traffic-poc/prior-implements/yolov3/weights/calibration_output.joblib")
lane_list = calibration_params['lane_list']
class NewBBox:

    def __init__(self, keypoints, descriptors, cls, probability, xyxy):

        global lane_list
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.bbox = xyxy
        self.cls = cls
        # self.centroid = (int(xyxy[0] + xyxy[2]) / 2, int(xyxy[1] + xyxy[3]) / 2)
        self.centroid = (int(xyxy[0] + xyxy[2]) / 2, int(xyxy[3]))
        self.lane = find_lane(lane_list, np.array(self.centroid))
        self.probability = probability

    @classmethod
    def no_keypoints(my_cls,cls, probability, xyxy):
        return my_cls([], [], cls, probability, xyxy)


class TrackedBBox:

    vehicleID = 0

    def __init__(self, new_bbox):
        self.name = 'v'+str(TrackedBBox.vehicleID)
        self.centroid_list = [new_bbox.centroid]
        self.lane_list = [new_bbox.lane]
        self.probability_list = np.zeros(NO_OF_CLASSES)
        self.probability_list[new_bbox.cls] += new_bbox.probability
        self.bbox = new_bbox.bbox
        self.keypoints = new_bbox.keypoints
        self.descriptors = new_bbox.descriptors

        self.frames_present = 1
        self.frames_absent = 0                      # keeps track of number of frames elapsed since last seen

        TrackedBBox.vehicleID += 1

    def print_details(self):
        print("BBox Name : ", self.name)
        print("Frames present : ", self.frames_present)
        print(self.probability_list)

    def update_location(self, new_bbox):
        self.centroid_list.append(new_bbox.centroid)
        self.probability_list[new_bbox.cls] += new_bbox.probability
        self.lane_list.append(new_bbox.lane)
        self.bbox = new_bbox.bbox
        self.keypoints = new_bbox.keypoints
        self.descriptors = new_bbox.descriptors

        self.frames_absent = 0
        self.frames_present += 1

    def update_missing(self):
        self.frames_absent += 1


def initiate_tracker(nfeatures=500):
    return cv2.ORB_create(nfeatures)


def initiate_matcher():
    return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def detect_key_points(image,bbox,tracker):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)
    mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 1
    kp, des = tracker.detectAndCompute(img_gray, mask)
    return kp, des


def match_two_boxes(bf, desc1, desc2):
    if (type(desc1) != type(None)) and (type(desc2) != type(None)):
        matches = bf.match(desc1, desc2)
        return get_matching_distance(sorted(matches, key=lambda x: x.distance))
    else:
        return(float('inf'))


def get_matching_distance(match_list):
    distance = 0
    i=0
    for i, match in enumerate(match_list):
        distance += match.distance
        if i == 9:
            break

    return distance/(i+1)


def initiate_tracked_bboxes(new_bbox_list):
    tracked_bbox_list = []

    for bbox in new_bbox_list:
        tracked_bbox_list.append(TrackedBBox(bbox))

    return tracked_bbox_list


def get_nearest_tracked_bbox(bf, new_bbox, tracked_bbox_list):
    min_distance = float('inf')
    nearest_tracked_bbox_index = None

    for idx, tracked_bbox in enumerate(tracked_bbox_list):
        distance = match_two_boxes(bf, tracked_bbox.descriptors, new_bbox.descriptors)
        if distance < min_distance:
            min_distance = distance
            nearest_tracked_bbox_index  = idx
    # print("Min Distance : ", min_distance)
    return nearest_tracked_bbox_index, min_distance


def generate_distance_matrix(bf, new_bbox_list, tracked_bbox_list):

    distance_matrix = np.empty([len(new_bbox_list), len(tracked_bbox_list)])

    for i, new_bbox in enumerate(new_bbox_list):
        for j, tracked_bbox in enumerate(tracked_bbox_list):
            distance_matrix[i, j] = match_two_boxes(bf, tracked_bbox.descriptors, new_bbox.descriptors)

    return distance_matrix


def track_bboxes(bf, tracked_bbox_list, new_bbox_list, count_list, label_list, blank_frame):

    if (len(tracked_bbox_list)==0) & (len(new_bbox_list)>0):
        return initiate_tracked_bboxes(new_bbox_list), count_list
    else:
        new_tracked_bbox_list = []

        for new_bbox in new_bbox_list:
            nearest_idx, min_distance = get_nearest_tracked_bbox(bf, new_bbox, tracked_bbox_list)
            if (min_distance < DISTANCE_THRESHOLD):
                updated_tracked_bbox = tracked_bbox_list.pop(nearest_idx)
                updated_tracked_bbox.update_location(new_bbox)
                new_tracked_bbox_list.append(updated_tracked_bbox)
            else:
                new_tracked_bbox_list.append(TrackedBBox(new_bbox))

        for bbox in tracked_bbox_list:
            bbox.update_missing()
            if bbox.frames_absent >= OLD_LIMIT:
                count_list = validate_tracked_bbox(bbox,count_list, label_list, blank_frame)
            else:
                new_tracked_bbox_list.append(bbox)

        return new_tracked_bbox_list, count_list


def generate_mask(mask,bbox):
    mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 1
    return mask


def detect_kp2(img_gray,mask,tracker):
    kp, des = tracker.detectAndCompute(img_gray, mask)
    return kp, des


def assign_keypoints(bbox_list, keypoint_list, des):
    for idx, kp in enumerate(keypoint_list):
        for bb in bbox_list:
            if(kp.pt[0] > bb.bbox[0]) and (kp.pt[0] < bb.bbox[2]) and (kp.pt[1] > bb.bbox[1]) and \
                    (kp.pt[1] < bb.bbox[3]):
                bb.keypoints.append(kp)
                bb.descriptors.append(des[idx, :])

    for bb in bbox_list:
        bb.descriptors = np.array(bb.descriptors)

def detect_key_points_crop(image,bbox,tracker):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray = img_gray[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    mask = np.ones_like(img_gray)
    kp, des = tracker.detectAndCompute(img_gray, mask)
    return kp, des


def print_id(img, tracked_bbox_list):
    for tbb in tracked_bbox_list:
        label = tbb.name
        lane = str(tbb.lane_list[-1])
        if(tbb.frames_absent==0):
            c = tbb.centroid_list[-1]
            cv2.putText(img, label, (int(c[0]), int(c[1]) - 2), 0, 1, [0, 0, 255], thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(img, lane, (int(c[0]+100), int(c[1]) - 2), 0, 1, [0, 0, 255], thickness=2, lineType=cv2.LINE_AA)


def inter_cls_nms(detections, threshold=0.8):
    # detections are expected to be sorted (max to min in this function since the previous function does this
    # (x1, y1, x2, y2, object_conf, class_conf, class)
    det_max=[]
    while(detections.shape[0]):
        det_max.append(detections[0,:])
        iou = bbox_iou(detections[0,:],detections[:,:])
        # print(iou)
        detections = detections[iou<threshold,:]
    det_max = torch.cat(det_max).reshape((-1,7))
    return det_max


def validate_tracked_bbox(tracked_bbox, count_vector, label_vector, img):
    print("Validating...")

    tracked_bbox.print_details()            # Need to extend the function to validate and assign the best class
    if tracked_bbox.frames_present >= MIN_PRESENT:
        max_index = np.argmax(tracked_bbox.probability_list)
        print(tracked_bbox.name, label_vector[max_index])
        count_vector[max_index] += 1
        draw_path(img,tracked_bbox.centroid_list, tracked_bbox.name)
    else:
        print(tracked_bbox.name, "Tracker discarded")

    return count_vector


def matrix_based_track_bboxes(bf, tracked_bbox_list, new_bbox_list, count_list, label_list, blank_frame, test_flag=False):

    if (len(tracked_bbox_list)==0) & (len(new_bbox_list)>0):
        return initiate_tracked_bboxes(new_bbox_list), count_list
    else:
        distance_matrix = generate_distance_matrix(bf, new_bbox_list,tracked_bbox_list)
        unused_new_bboxes = list(range(len(new_bbox_list)))
        unused_tracked_bboxes = list(range(len(tracked_bbox_list)))
        new_tracked_bbox_list = []
        while(1):
            if(len(new_bbox_list)==0):
                break
            min_index = np.unravel_index(np.argmin(distance_matrix, axis=None), distance_matrix.shape)
            min_distance = distance_matrix[min_index]
            if (min_distance > DISTANCE_THRESHOLD):
                break
            else:
                unused_new_bboxes.remove(min_index[0])
                unused_tracked_bboxes.remove(min_index[1])

                updated_tracked_bbox = tracked_bbox_list[min_index[1]]
                updated_tracked_bbox.update_location(new_bbox_list[min_index[0]])
                new_tracked_bbox_list.append(updated_tracked_bbox)

                distance_matrix[min_index[0],:] = float('inf')
                distance_matrix[:,min_index[1]] = float('inf')

        for k in unused_new_bboxes:
            new_tracked_bbox_list.append(TrackedBBox(new_bbox_list[k]))

        for k in unused_tracked_bboxes:
            bbox = tracked_bbox_list[k]
            bbox.update_missing()
            if bbox.frames_absent >= OLD_LIMIT:
                count_list = validate_tracked_bbox(bbox, count_list, label_list, blank_frame)
            else:
                new_tracked_bbox_list.append(bbox)

        return new_tracked_bbox_list, count_list
