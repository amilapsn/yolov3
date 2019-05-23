import cv2
import numpy as np
from sklearn.externals import joblib

from utils.calibration_utils import calculate_area_from_origin

windowName = "frame"
calibration_image = cv2.imread("data/blank_frames/blank_frame0.png")
img_backup = calibration_image.copy()
cv2.namedWindow(windowName)

drawing = False
mode = 0
color = (255, 0, 0)

lane_list = []
start_trigger_list = []
end_trigger_list = []


def line_draw(event, x, y, flags, param):
    """
    A callback function for interactive line drawing for marking lanes and speed triggers
    saves the (sorted) lane_list, start_trigger_list, end_trigger_list as a dictionary named calibration_output
    :param event: Mouse click event
    :param x: x coordinate of mouse during event trigger
    :param y: y coordinate of mouse during event trigger
    :param flags: dummy
    :param param: dummy
    :return: None
    """
    global ix, iy, drawing, calibration_image, img_backup, color, mode, \
        lane_list, start_trigger_list, end_trigger_list, area_list

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True  # start to draw when L button down
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            calibration_image = img_backup.copy()
            cv2.line(calibration_image,(ix,iy), (x,y), color, 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False     # end drawing when L button up
        cv2.line(img_backup,(ix,iy), (x,y), color, 2)
        calibration_image = img_backup.copy()

        if mode == 0:
            lane_list.append([[ix, iy], [x, y]])

            if len(lane_list) == lanes+1:
                mode = 1
                color = (0, 255, 0)

        elif mode == 1:
            start_trigger_list.append([[ix, iy], [x, y]])

            if len(start_trigger_list) == entrance_triggers:
                mode = 2
                color = (0, 0, 255)

        elif mode == 2:
            end_trigger_list.append([[ix, iy], [x, y]])

            if len(end_trigger_list) == exit_triggers:
                cv2.imwrite('output/calibrated_frame.jpg', calibration_image)
                lane_list = np.array(lane_list)
                start_trigger_list = np.array(start_trigger_list)
                end_trigger_list = np.array(end_trigger_list)
                mode = 3
                area_list = calculate_area_from_origin(lane_list[:,0,:].T,lane_list[:,1,:].T)
                lane_list = lane_list[np.argsort(area_list)]
                output_dict = {"lane_list" : lane_list, "start_trigger_list" : start_trigger_list,
                               "end_trigger_list" : end_trigger_list}
                joblib.dump(output_dict, "weights/calibration_output.joblib")

                exit()


cv2.setMouseCallback(windowName, line_draw)


def main():

    global lanes, entrance_triggers, exit_triggers

    lanes = int(input("Enter the number of lanes : "))
    entrance_triggers = int(input("Enter the number of entrances : "))
    exit_triggers = int(input("Enter the number of exits : "))

    while (True):
        cv2.imshow(windowName, calibration_image)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
