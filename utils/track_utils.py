import cv2
import numpy as np
import time
import torch
from utils.utils import bbox_iou
from utils.path_visualization_utils import draw_path
from utils.lane_speed_utils import find_lane, do_intersect
from sklearn.externals import joblib

NO_OF_CLASSES = 7
DISTANCE_THRESHOLD = 50
OLD_LIMIT = 20
MIN_PRESENT = 10
FPS = 30
START_TO_END_DISTANCE = 5

calibration_params = joblib.load("weights/calibration_output.joblib")
lane_list = calibration_params['lane_list']
start_triggers = calibration_params['start_trigger_list']
end_triggers = calibration_params['end_trigger_list']


class NewBBox:
    """
    Class used to encapsulate the attributes corresponding to a newly detected bounding box

    Attributes:
        keypoints: ORB (or any other) keypoints found within the detected bounding box (list of n [x,y] coordinates)
        descriptors: descriptor (32 or m dimensional) corresponding to each keypoint (nxm dimensional np array)
        bbox: bounding box coordinates (list of four coordinates [x1,y1,x2,y2] corresponding to top left and bottom
        right corners)
        cls: vehicle class of the detected bounding box (int)
        centroid: bottom center point of the bounding box (tuple (x,y)) or middle point
        lane: estimated laneID of the bounding box (int)
        probability: class probability of the bounding box

    """

    def __init__(self, keypoints, descriptors, cls, probability, xyxy):
        """
        Constructor of a NewBBox
        :param keypoints: ORB (or any other) keypoints found within the detected bounding box (list of n [x,y]
        coordinates)
        :param descriptors: descriptor (32 or m dimensional) corresponding to each keypoint (nxm dimensional np array)
        :param cls: vehicle class of the detected bounding box (int)
        :param probability: class probability of the bounding box
        :param xyxy: bounding box coordinates (list of four coordinates [x1,y1,x2,y2] corresponding to top left and
        bottom
        """

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
    def no_keypoints(my_cls, cls, probability, xyxy):
        """
        Class method to construct a newBBox without keypoints and descriptors
        :param cls: vehicle class of the detected bounding box (int)
        :param probability: class probability of the bounding box
        :param xyxy: bounding box coordinates (list of four coordinates [x1,y1,x2,y2] corresponding to top left and
        bottom
        :return:
        """

        return my_cls([], [], cls, probability, xyxy)


class TrackedBBox:
    """
    Class used to accumulate attributes corresponding to tracked bounding boxes.

    Attributes:
        name: a unique ID corresponding to each instance of newly tracked bounding box
        centroid_list: list of all the centroids corresponding to the tracked bounding box
        lane_list: list of lane IDs the bounding box has appeared over time
        probability_list: numpy array consisting of accumulated probability of each class (shape: NO_OF_CLASSES)
        keypoints: latest set of ORB (or any other) keypoints found within the detected bounding box (list of n [x,y]
        coordinates)
        descriptors: latest descriptors (32 or m dimensional) corresponding to each keypoint (nxm dimensional np array)
        frames_present: count variable that accumulate over how many frames the particular tracked bbox has appeared
        frames_absent: count variable that keeps track of number of frames elapsed since last seen
        start_frame: frame at which the bbox crosses a start_trigger
        end_frame: frame at which the bbox crosses a end_trigger
        speed: speed of the vehicle

    Class Attributes:
        vehicleID: class variable to store the ID of the next vehicle

    Methods:
        print_details: method to print out basic details about the trackedBBox
        update_location: method to update/accumilate features of the trackedBBox when it is matched with a newBBox
        update_missing: method to update the attributes when the tracked bbox is missing/doesn't match in a frame
    """

    vehicleID = 0

    def __init__(self, new_bbox):
        """
        Constructor to initiate a new tracked
        :param new_bbox: newBBox instance which can be used to initiate a new trackedBBox
        """

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

        self.start_frame = None
        self.end_frame = None
        self.speed = None

    def print_details(self):
        """
        prints out basic details about the trackedBBox
        :return: None
        """

        print("BBox Name : ", self.name)
        print("Frames present : ", self.frames_present)
        print(self.probability_list)
        if self.speed is not None:
            print("speed : ", self.speed)

    def update_location(self, new_bbox, frame_id):
        """
        updates/accumilates features of the trackedBBox when it is matched with a newBBox
        :param new_bbox: Matching newBBox instance
        :param frame_id: id of the current frame
        :return: None
        """

        self.centroid_list.append(new_bbox.centroid)
        self.probability_list[new_bbox.cls] += new_bbox.probability
        self.lane_list.append(new_bbox.lane)
        self.bbox = new_bbox.bbox
        self.keypoints = new_bbox.keypoints
        self.descriptors = new_bbox.descriptors

        self.frames_absent = 0
        self.frames_present += 1

        if self.start_frame is None:
            for l in range(start_triggers.shape[0]):
                if do_intersect(start_triggers[l,:,:],np.array((self.centroid_list[-1],self.centroid_list[-2]))):
                    print('\n\n',self.name, " triggered start.................\n")
                    self.start_frame = frame_id

        elif self.end_frame is None:
            for l in range(end_triggers.shape[0]):
                if do_intersect(end_triggers[l,:,:],np.array((self.centroid_list[-1],self.centroid_list[-2]))):
                    print('\n\n',self.name, " triggered end.................\n")
                    self.end_frame = frame_id
                    time = (self.end_frame - self.start_frame)/FPS
                    self.speed = START_TO_END_DISTANCE/time * 3.6


    def update_missing(self):
        """
        Updates the instance when it is missing in the current frame
        :return: None
        """
        self.frames_absent += 1


def initiate_tracker(nfeatures=500):
    """
    function to initiate an ORB feature detector
    :param nfeatures: No. of features
    :return: ORB feature detector instance
    """

    return cv2.ORB_create(nfeatures)


def initiate_matcher():
    """
    function to initiate a brute force matcher for keypoint matching.
    :return: brute force matcher instance
    """

    return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def detect_key_points(image, bbox, tracker):
    """
    Detects key points in an image within a bounding box region. (feeds in the whole image but masks out the bbox
    region) (obsolete due to slow)
    :param image: image/frame
    :param bbox: bounding box coordinates (list of four coordinates [x1,y1,x2,y2] corresponding to top left and bottom
        right corners)
    :param tracker: key point detector instance
    :return: list of keypoints and an np array of descriptors
    """

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)
    mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 1
    kp, des = tracker.detectAndCompute(img_gray, mask)
    return kp, des


def match_two_boxes(bf, desc1, desc2):
    """
    When given descriptors corresponding to two bounding boxes and a brute force matching instance. Returns the
    distance between the two descriptors
    :param bf: Brute force matcher instance
    :param desc1: descriptor corresponding to bbox 1
    :param desc2: descriptor corresponding to bbox 2
    :return: distance between the two bounding boxes
    """

    if (type(desc1) != type(None)) and (type(desc2) != type(None)):
        matches = bf.match(desc1, desc2)
        return get_matching_distance(sorted(matches, key=lambda x: x.distance))
    else:
        return float('inf')


def get_matching_distance(match_list):
    """
    Given a matching list generated by a bf matcher. Returns the distance between the top 10 matches.
    :param match_list: list of DMatch objects generated by BF matcher instance
    :return: Distance between the top 10 matches
    """

    distance = 0
    i = 0
    for i, match in enumerate(match_list):
        distance += match.distance
        if i == 9:
            break

    return distance/(i+1)


def initiate_tracked_bboxes(new_bbox_list):
    """
    Given a list of newBBox instances, generates a list of trackedBBox instances corresponding to each newBBox.
    :param new_bbox_list: list of newBBox instances
    :return: list of trackedBBox instances
    """

    tracked_bbox_list = []

    for bbox in new_bbox_list:
        tracked_bbox_list.append(TrackedBBox(bbox))

    return tracked_bbox_list


def get_nearest_tracked_bbox(bf, new_bbox, tracked_bbox_list):
    """
    Given a new bounding box and a list of previously tracked bounding boxes. returns the nearest tracked bounding box
    to the given new bounding box (obsolete due to looping)
    :param bf: brute force matcher instance
    :param new_bbox: NewBBox instance
    :param tracked_bbox_list: lst of previously trackedBBoxes
    :return: index of nearest tracckedBBox, and distance between the newBBox andd nearest trackedBBox
    """

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
    """
    Given a list of new bounding boxes and list of tracked bounding boxes matches the key points and generates a
    distance matrix with distance_matrix[i,j] giving the distance between ith newBBox and jth trackedBBox
    :param bf: Brute force matcher
    :param new_bbox_list: list of new bounding box instances
    :param tracked_bbox_list: list of tracked bounding box instances
    :return: distance matrix (np.array)
    """

    distance_matrix = np.empty([len(new_bbox_list), len(tracked_bbox_list)])

    for i, new_bbox in enumerate(new_bbox_list):
        for j, tracked_bbox in enumerate(tracked_bbox_list):
            distance_matrix[i, j] = match_two_boxes(bf, tracked_bbox.descriptors, new_bbox.descriptors)

    return distance_matrix


def track_bboxes(bf, tracked_bbox_list, new_bbox_list, count_list, label_list, blank_frame):
    """
    Main top level function used for tracking bounding boxes. When called upon matches the given new bounding boxes
    with already existing tracked bounding boxes. when a tracked bounding box expires discards it and calls
    validate_tracked_bbox function
    (obsolete due to inherent ability to overlook more robust matches due to for loop and pop operation)
    :param bf: Brute force matcher
    :param tracked_bbox_list: list of tracked bounding box instances
    :param new_bbox_list: list of new bounding box instances
    :param count_list: list to keep count of vehicles corresponding to each category
    :param label_list: list of labels (names) of each vehicle class.
    :param blank_frame: blank frame (used for drawing the path for visualization)
    :return: updated tracked_bounding_box_list and updated count_list
    """

    if (len(tracked_bbox_list)==0) & (len(new_bbox_list)>0):
        return initiate_tracked_bboxes(new_bbox_list), count_list
    else:
        new_tracked_bbox_list = []

        for new_bbox in new_bbox_list:
            nearest_idx, min_distance = get_nearest_tracked_bbox(bf, new_bbox, tracked_bbox_list)
            if min_distance < DISTANCE_THRESHOLD:
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
    """
    updates the mask with the new bbox. This function together with detect_kp2 can be used to detect keypoints in a f
    rame in one shot, by generating a mask that has ones only in the detected bounding box region
    :param mask: mask to be updated with the new bounding box
    :param bbox: bounding box coordinates (list of four coordinates [x1,y1,x2,y2] corresponding to top left and bottom
        right corners)
    :return: updated mask
    """
    mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 1
    return mask


def detect_kp2(img_gray,mask,tracker):
    """
    function used detect keypoints in an image in the region specified by the mask. Can be used to detect keypoints in
    a frame in one shot if mask generated exposes only the vehicle pesent regions.
    (obsolute due to low number of keypoints were detected for each bbox. Although it is faster than detect_keypoints
    function after detection still associating keypoints to their corresponding bboxes is still slow and not so
    intutive)
    :param img_gray: grayscale image
    :param mask: mask with ones in region of interst and zeroes elsewhere
    :param tracker: keypoint detector instance
    :return: list of keypoints and an np array of descriptors
    """

    kp, des = tracker.detectAndCompute(img_gray, mask)
    return kp, des


def assign_keypoints(bbox_list, keypoint_list, des):
    """
    Function to assign key points detected from detect_kp2 to corresponding bounding boxes.
    (obsolete since detect_kp2 is not used anymore)
    :param bbox_list: list of new BBoxes
    :param keypoint_list: list of keypoints detected by detect_kp2 function
    :param des: descriptor np array corresponding to keypoints
    :return: None. (updates the newBBox instances with the kp and des)
    """

    for idx, kp in enumerate(keypoint_list):
        for bb in bbox_list:
            if(kp.pt[0] > bb.bbox[0]) and (kp.pt[0] < bb.bbox[2]) and (kp.pt[1] > bb.bbox[1]) and \
                    (kp.pt[1] < bb.bbox[3]):
                bb.keypoints.append(kp)
                bb.descriptors.append(des[idx, :])

    for bb in bbox_list:
        bb.descriptors = np.array(bb.descriptors)

def detect_key_points_crop(image,bbox,tracker):
    """
    Detects key points in an image within a bounding box region. (inputs the cropped out region only. Hence loses
    information about absolute locations of key points)
    :param image: image/frame for kp detection
    :param bbox: bounding box coordinates (list of four coordinates [x1,y1,x2,y2] corresponding to top left and bottom
        right corners)
    :param tracker: key point detector instance
    :return: list of keypoints and an np array of descriptors
    """

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray = img_gray[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    mask = np.ones_like(img_gray)
    kp, des = tracker.detectAndCompute(img_gray, mask)
    return kp, des


def print_id(img, tracked_bbox_list):
    """
    Helper function to print the vehicle IDs to the current frame
    :param img: image/ frame
    :param tracked_bbox_list: list of tracked bounding box instances
    :return: None
    """

    for tbb in tracked_bbox_list:
        label = tbb.name
        lane = str(tbb.lane_list[-1])
        if(tbb.frames_absent==0):
            c = tbb.centroid_list[-1]
            cv2.putText(img, label, (int(c[0]), int(c[1]) - 2), 0, 1, [0, 0, 255], thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(img, lane, (int(c[0]+100), int(c[1]) - 2), 0, 1, [0, 0, 255], thickness=2, lineType=cv2.LINE_AA)


def inter_cls_nms(detections, threshold=0.8):
    """
    A NMS function similar to the one found in utils.utils but unlike that does suppression regardless of the class
    :param detections: output from network/ previous class level NMS stage
    :param threshold: NMS threshold level
    :return: Non overlapping detection regions
    """

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
    """
    Function which is triggered when a tracked bounding box expires (when a tracked bounding boxes is not detected
    for more than N frames). Checks the parameters and assigns a class to the trackedBBox. If not valid rejects it.
    :param tracked_bbox: TrackedBBox instance to be validated
    :param count_vector: list containing the vehicle counts.
    :param label_vector: name_list corresponding to each vehicle class.
    :param img: blank frame for drawing path
    :return: updated count_vector
    """

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


def matrix_based_track_bboxes(bf, tracked_bbox_list, new_bbox_list, count_list, label_list, blank_frame, frame_id,
                              test_flag=False):
    """
    Alternative main top level function used for tracking bounding boxes. When called upon matches the given new
    bounding boxes with already existing tracked bounding boxes. when a tracked bounding box expires discards it and
    calls validate_tracked_bbox function. More robust than track_bboxes function, since it uses a matrix based matching
    :param bf: Brute force matcher
    :param tracked_bbox_list: list of tracked bounding box instances
    :param new_bbox_list: list of new bounding box instances
    :param count_list: list to keep count of vehicles corresponding to each category
    :param label_list: list of labels (names) of each vehicle class.
    :param blank_frame: blank frame (used for drawing the path for visualization)
    :param frame_id: id of the current frame
    :return: updated tracked_bounding_box_list and updated count_list
    """

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
                updated_tracked_bbox.update_location(new_bbox_list[min_index[0]], frame_id)
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
