# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 13:10:34 2019

@author: shiha
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.animation as animation
from collections import deque
from ReadCameraModel import*
from UndistortImage import*
from EstimateFundamentalMatrix import*
from GetInlierRANSANC import*
from EastimateEssentialMatrix import*
from ExtractCameraPose import*
from disambiguatepose import*

def featureDetection():
    thresh = dict(threshold=25, nonmaxSuppression=True);
    fast = cv2.FastFeatureDetector_create(**thresh)
    return fast

def featureMatching(img_1, img_2, p1):
    #parameter img_1: first image
    #parameter img_2: the successive image
    #parameter p1: feature points got from image1.
    
    #############################################

    lk_params = dict( winSize  = (21,21),
                      maxLevel = 3,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    p2, st, err = cv2.calcOpticalFlowPyrLK(img_1, img_2, p1, None, **lk_params)
    st = st.reshape(st.shape[0])
    ##find good one
    p1 = p1[st==1]
    p2 = p2[st==1]

    return p1,p2

poses = deque()
fig, ax = plt.subplots()
cap = cv2.VideoCapture('sample.avi')

fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('Oxford_dataset/model') 
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

ret, pre_frame = cap.read()
pre_frame = cv2.cvtColor(pre_frame, cv2.COLOR_BGR2GRAY)
pos = np.asarray([0, 0, 0])
Rpos = np.eye(3)
count = 0

while ret:
    ret, cur_frame = cap.read()
    if ret is not True:
        break
    cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

    detector = featureDetection()
    kp1 = detector.detect(pre_frame)
    ori = np.array([ele.pt for ele in kp1],dtype='float32')
    ori, sub = featureMatching(pre_frame, cur_frame, ori)
    
    #get fundamental matrix
    F, inliers1, inliers2= F_RANSAC(ori, sub)
    
    #estimate essential matrix
    E = Ematrix(F, K)
    
    #get pose
    T = get_pose(E)
    R, C = linear_tri_disambiguatePoint(T)  

    
    #estimate camera pose based on rotational as well as traslational matrix.
    pos = pos + np.dot(Rpos, C)
    Rpos = np.dot(Rpos, R)
    poses.append(pos)
    
    #scatter the points that represent vehicle position.
    ax.scatter(-pos[0], pos[2], color = 'r')
    fig.canvas.draw()
    filename='trajectory/car_steps'+str(count).zfill(4)+'.png'
    plt.savefig(filename, dpi=96)
    plt.gca()
    
    count += 1
    pre_frame = cur_frame


    cv2.imshow('frame', cur_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

