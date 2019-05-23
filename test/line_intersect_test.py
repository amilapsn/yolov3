from utils.lane_speed_utils import do_intersect

import cv2
import numpy as np
from sklearn.externals import joblib

windowName = "frame"
calibration_image = cv2.imread("../blank_frame0.png")
img_backup = calibration_image.copy()
cv2.namedWindow(windowName)

drawing = False
color = (255, 0, 0)

line_list =[]


def line_draw(event, x, y, flags, param):
    """
    A callback function for interactive line drawing
    :param event: Mouse click event
    :param x: x coordinate of mouse during event trigger
    :param y: y coordinate of mouse during event trigger
    :param flags: dummy
    :param param: dummy
    :return: None
    """
    global ix, iy, drawing, calibration_image, img_backup, color, line_list
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
        line_list.append(np.array(((ix,iy), (x,y))))
        calibration_image = img_backup.copy()



cv2.setMouseCallback(windowName, line_draw)

def main():
    global line_list
    while (True):
        cv2.imshow(windowName, calibration_image)

        if len(line_list) == 2:
            val = do_intersect(line_list[0],line_list[1])
            print(val)
            line_list=[]

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()