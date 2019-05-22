import cv2
import numpy as np
import pickle as pkl
import random

def get_str_from_tensor(xyxy):
    string = "["
    for p in xyxy:
        string += str(p.item()) + ","

    return string[:-2]+']'

colors = pkl.load(open("pallete", "rb"))

def draw_path(img,centroid_list, name):
    color = random.choice(colors)
    cv2.putText(img, name, (int(centroid_list[0][0]),int(centroid_list[0][1])), 0, 1, color, thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(img, name, (int(centroid_list[-1][0]),int(centroid_list[-1][1])), 0, 1, color, thickness=2, lineType=cv2.LINE_AA)
    centroid_list = np.array(centroid_list,dtype=int).reshape((-1,1,2))
    cv2.polylines(img, [centroid_list], False, color, 3)


def print_counts(im0, i, count_list, classes, draw_lanes=None):

    cv2.putText(im0, str(i), (100, 100), 0, 1, [255, 0, 0], thickness=2, lineType=cv2.LINE_AA)
    # cv2.line(im0,(0,360),(1920,360),[255, 0, 0], thickness=2)

    cv2.putText(im0, classes[0] + " : " + str(count_list[0]), (100, 130), 0, 1, [255, 0, 255], thickness=2,
                lineType=cv2.LINE_AA)
    cv2.putText(im0, classes[1] + " : " + str(count_list[1]), (100, 160), 0, 1, [255, 0, 255], thickness=2,
                lineType=cv2.LINE_AA)
    cv2.putText(im0, classes[2] + " : " + str(count_list[2]), (100, 190), 0, 1, [255, 0, 255], thickness=2,
                lineType=cv2.LINE_AA)
    cv2.putText(im0, classes[3] + " : " + str(count_list[3]), (100, 220), 0, 1, [255, 0, 255], thickness=2,
                lineType=cv2.LINE_AA)
    cv2.putText(im0, classes[4] + " : " + str(count_list[4]), (100, 250), 0, 1, [255, 0, 255], thickness=2,
                lineType=cv2.LINE_AA)
    cv2.putText(im0, classes[5] + " : " + str(count_list[5]), (100, 280), 0, 1, [255, 0, 255], thickness=2,
                lineType=cv2.LINE_AA)
    cv2.putText(im0, classes[6] + " : " + str(count_list[6]), (100, 310), 0, 1, [255, 0, 255], thickness=2,
                lineType=cv2.LINE_AA)

    if draw_lanes != None:
        lane_lines = draw_lanes['lane_list']
        for i in range(lane_lines.shape[0]):
            cv2.line(im0,tuple(lane_lines[i,0,:]),tuple(lane_lines[i,1,:]),(255,0,0),2)

        start_lines = draw_lanes['start_trigger_list']
        for i in range(start_lines.shape[0]):
            cv2.line(im0,tuple(start_lines[i,0,:]),tuple(start_lines[i,1,:]),(0,255,0),2)

        end_lines = draw_lanes['end_trigger_list']
        for i in range(end_lines.shape[0]):
            cv2.line(im0, tuple(end_lines[i, 0, :]), tuple(end_lines[i, 1, :]), (0, 0, 255), 2)