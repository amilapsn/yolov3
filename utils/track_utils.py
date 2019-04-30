import cv2
import numpy as np
import time

NO_OF_CLASSES = 7
DISTANCE_THRESHOLD = 50
OLD_LIMIT = 5


class NewBBox:

    def __init__(self, keypoints, descriptors, cls, probability, xyxy):
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.bbox = xyxy
        self.cls = cls
        self.centroid = (int(xyxy[0] + xyxy[2]) / 2, int(xyxy[1] + xyxy[3]) / 2)
        self.probability = probability

    @classmethod
    def no_keypoints(my_cls,cls, probability, xyxy):
        return my_cls([], [], cls, probability, xyxy)


class TrackedBBox:

    vehicleID = 0

    def __init__(self, new_bbox):
        self.name = 'v'+str(TrackedBBox.vehicleID)
        self.centroid_list = [new_bbox.centroid]

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
    matches = bf.match(desc1, desc2)
    return get_matching_distance(sorted(matches, key=lambda x: x.distance))


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
    print("Min Distance : ", min_distance)
    return nearest_tracked_bbox_index, min_distance


def validate_tracked_bbox(tracked_bbox):
    print("Validating...")
    tracked_bbox.print_details()            # Need to extend the function to validate and assign the best class


def track_bboxes(bf, tracked_bbox_list, new_bbox_list):

    if (len(tracked_bbox_list)==0) & (len(new_bbox_list)>0):
        return initiate_tracked_bboxes(new_bbox_list)
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
                validate_tracked_bbox(bbox)
            else:
                new_tracked_bbox_list.append(bbox)

        return new_tracked_bbox_list


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
        c = tbb.centroid_list[-1]
        cv2.putText(img, label, (int(c[0]), int(c[1]) - 2), 0, 1, [0, 255, 0], thickness=2, lineType=cv2.LINE_AA)
