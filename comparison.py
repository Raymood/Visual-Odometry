# -*- coding: utf-8 -*-
"""
Created on Thu May  2 14:05:30 2019

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

poses = deque()
poses1 = deque()
fig, ax1 = plt.subplots()
#fig, ax2 = plt.subplots()

cap = cv2.VideoCapture('sample.avi')

fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('Oxford_dataset/model') 
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

ret, pre_frame = cap.read()
pre_frame = cv2.cvtColor(pre_frame, cv2.COLOR_BGR2GRAY)
pos = np.asarray([0, 0, 0])
Rpos = np.eye(3)
pos1 = np.asarray([0, 0, 0])
Rpos1 = np.eye(3)
while ret:
    ret, cur_frame = cap.read()
    cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

    detector = featureDetection()
    kp1 = detector.detect(pre_frame)
    ori = np.array([ele.pt for ele in kp1],dtype='float32')
    ori, sub = featureMatching(pre_frame, cur_frame, ori)

    #get fundamental matrix
    F, inliers1, inliers2= F_RANSAC(ori, sub)
    #F = Fmatrix(ori, sub)
    #F, mask = cv2.findFundamentalMat(ori,sub,cv2.FM_RANSAC)

    E = Ematrix(F, K)
    T = get_pose(E)
    R, C = linear_tri_disambiguatePoint(T)
    #inliers1 = np.hstack((inliers1, np.ones((inliers1.shape[0], 1))))
    #inliers2 = np.hstack((inliers2, np.ones((inliers2.shape[0], 1))))

    #built_in functioin
    E1, mask = cv2.findEssentialMat(sub, ori, fx, (cx, cy), cv2.RANSAC,0.999,1.0)
    #print(E, E1)

    _, R1, C1, mask = cv2.recoverPose(E1, sub, ori, focal=fx, pp = (cx, cy))
  
    #print(C)
    
    pos = pos + np.dot(Rpos, C)
    Rpos = np.dot(R, Rpos)
    poses.append(pos)
    
    pos1 = pos1 + np.dot(Rpos1, C1)
    Rpos1 = np.dot(R1, Rpos1)
    poses1.append(pos1)
    
    #print(np.dot(Rpos, C))
    #plt.figure(1)
    ax1.scatter(-pos[0], pos[2], color = 'r')
    ax1.scatter(pos1[0], pos1[2], color = 'b')
    fig.canvas.draw()
    pre_frame = cur_frame
    #img3 = cv2.drawMatches(pre_frame,kp1,cur_frame,kp2,matches[:300], cur_frame, flags=0)
    cv2.imshow('frame', cur_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''
images = [cv2.imread(file) for file in glob.glob('Oxford_dataset\stereo\centre\*.png')]
images.reverse()
fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('Oxford_dataset/model') 

out = cv2.VideoWriter('sample.avi', 0,cv2.VideoWriter_fourcc('M','J','P','G'),40.0,(images[0].shape[1],images[0].shape[0]))
while images:
    frame = images.pop()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BayerGR2BGR)
    frame = UndistortImage(frame, LUT)
    out.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1)&0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()
'''