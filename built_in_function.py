# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:38:17 2019

@author: shiha
"""


import numpy as np
import matplotlib.pyplot as plt
import cv2
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

    lk_params = dict( winSize  = (21,21),
                      maxLevel = 3,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    p2, st, err = cv2.calcOpticalFlowPyrLK(img_1, img_2, p1, None, **lk_params)
    st = st.reshape(st.shape[0])
    ##find good one
    p1 = p1[st==1]
    p2 = p2[st==1]

    return p1,p2

fig, ax = plt.subplots()
cap = cv2.VideoCapture('sample.avi')

fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('Oxford_dataset/model') 
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

ret, pre_frame = cap.read()
pre_frame = cv2.cvtColor(pre_frame, cv2.COLOR_BGR2GRAY)
pos = np.asarray([0, 0, 0])
Rpos = np.eye(3)
while ret:
    ret, cur_frame = cap.read()
    cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

    detector = featureDetection()
    kp1 = detector.detect(pre_frame)
    ori = np.array([ele.pt for ele in kp1],dtype='float32')
    ori, sub = featureMatching(pre_frame, cur_frame, ori)

    #get fundamental matrix
    E, mask = cv2.findEssentialMat(sub, ori, fx, (cx, cy), cv2.RANSAC,0.999,1.0)
    _, R, t, mask = cv2.recoverPose(E, sub, ori, focal=fx, pp = (cx, cy))    
    

    Rpos = np.dot(R, Rpos)
    pos = pos + np.dot(Rpos, t)

    #plt.figure(1)
    ax.scatter(pos[2], pos[0], color = 'r')
    fig.canvas.draw()
    pre_frame = cur_frame
    #img3 = cv2.drawMatches(pre_frame,kp1,cur_frame,kp2,matches[:300], cur_frame, flags=0)
    cv2.imshow('frame', cur_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()