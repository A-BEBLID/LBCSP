import numpy as np
import cv2
from matplotlib import pyplot as plt
from cv2.xfeatures2d import matchGMS
import time
from numpy import *
from numpy.linalg import *


img1 = cv2.imread('hpatches/v_feast/1.ppm')  # queryImage
img2 = cv2.imread('hpatches/v_feast/4.ppm')  # trainImage


# Initiate AKAZE detector
detector = cv2.xfeatures2d.SIFT_create()



# Find the keypoints and descriptors with BEBLID
kp1,des1 = detector.detectAndCompute(img1, None)
kp2,des2 = detector.detectAndCompute(img2, None)





# create BFmatcher object



bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
# Match descriptors.
matches = bf.match(des1, des2)
matches_gms = matchGMS(img1.shape[:2], img2.shape[:2], kp1, kp2, matches, withScale=True, withRotation=True,
                       thresholdFactor=10)

#registrion
if len(matches_gms) > 4:
    ptsA= np.float32([kp1[m.queryIdx].pt for m in matches_gms]).reshape(-1, 1, 2)
    ptsB = np.float32([kp2[m.trainIdx].pt for m in matches_gms]).reshape(-1, 1, 2)
    ransacReprojThreshold = 4
    H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold);

    imgOut = cv2.warpPerspective(img2, H, (img1.shape[1],img1.shape[0]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)




#draw match
height = max(img1.shape[0], img2.shape[0])
width = img1.shape[1] + img2.shape[1]
output = np.zeros((height, width,3), dtype=np.uint8)
output[0:img1.shape[0], 0:img1.shape[1]] = img1
output[0:img2.shape[0], img1.shape[1]:] = img2[:]
#NO.1 draw circle
for i in range(len(matches_gms[:700])):
    left = kp1[matches_gms[i].queryIdx].pt
    right = tuple(sum(x) for x in zip(kp2[matches_gms[i].trainIdx].pt, (img1.shape[1], 0)))
    cv2.circle(output, tuple(map(int, left)), 1, (0, 255, 255), 6)
    cv2.circle(output, tuple(map(int, right)), 1, (0, 255, 0), 6)


#NO.2 draw line
for i in range(len(matches_gms[:800])):
    left = kp1[matches_gms[i].queryIdx].pt
    right = tuple(sum(x) for x in zip(kp2[matches_gms[i].trainIdx].pt, (img1.shape[1], 0)))
    cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (0, 0, 255))



# save result
cv2.imwrite('Accent1_sift.jpg', output)
cv2.imwrite('Accent2.jpg', imgOut)

