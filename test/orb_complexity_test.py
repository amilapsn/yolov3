import cv2
import numpy as np
import time
from utils.utils import *

images = "../data/track_images"
img_size = 416
img = cv2.imread('../data/track_images/1.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
output = '../output/1.jpg'
output2 = '../output/2.jpg'

orb = cv2.ORB_create()
mask = np.zeros_like(img_gray)

mask[400:700, 400:900] = 1

# compute the descriptors with ORB
t = time.time()
kp, des = orb.detectAndCompute(img_gray, mask)
print("time taken (masking) : ", time.time()-t)
# draw only keypoints location,not size and orientation

img_gray2 = img_gray[400:700, 400:900]
img2 = img[400:700, 400:900]
mask2 = np.ones_like(img_gray2)
t = time.time()
kp2, des2 = orb.detectAndCompute(img_gray2,mask2)
print("time taken (cropping) : ", time.time()-t)

cv2.drawKeypoints(img, kp, img, color=(0,255,0), flags=0)
cv2.imwrite(output, img)

cv2.drawKeypoints(img2, kp2, img2, color=(0,255,0), flags=0)
cv2.imwrite(output2, img2)




