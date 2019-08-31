# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:09:24 2019

@author: shiha
"""

import numpy as np
import random
import cv2 as f

def linear_triangulation(K, T, ori, sub):
    #parameter K: camera matirx.
    #parameter T: array of sets of camera poses.
    #parameter ori: current frame.
    #parameter sub: successive frame.
    
    ############################################
    
    #linear_triangulation
    X = np.zeros((4, ori.shape[0], 4))
    for i in range(4):
        for j in range(i + 1, 4):
            p1 = np.matmul(np.matmul(K, T[i][1]), np.hstack((np.eye(3), -T[i][0].reshape(3,1))))
            p2 = np.matmul(np.matmul(K, T[j][1]), np.hstack((np.eye(3), -T[j][0].reshape(3,1))))
            
            for t in range(ori.shape[0]):
                A = np.asarray([[np.dot(ori[t][0], p1[2]) - p1[0]], [np.dot(ori[t][1], p1[2]) - p1[1]], [np.dot(sub[t][0], p2[2]) - p1[0]], [np.dot(sub[t][1], p1[2]) - p1[1]]]).reshape(4, 4)
                u, s, v = np.linalg.svd(A)
                x_tri = v[:, -1]
                x_tri = x_tri/x_tri[-1]
                X[:,t,i] = x_tri
    return X
                
def disambiguatePoint(T, X, X_new):

    cnt = []
    for i in range(len(T)):
        cnt.append(np.sum(X[3,:,i] > 0) + np.sum(X_new[3,:,i] > 0))
        
    if cnt[0] == 0 & cnt[1] == 0 & cnt[2] == 0 & cnt[3] == 0:
        R = np.eye(3)
        t = np.zeros(3)
    else:
        index = cnt.index(max(cnt))
        R = T[index][1]
        t = T[index][0]
        if t[2] < 0:
            t = -t
            
    if abs(R[0,2]) < 0.001:
        R[0,2] = 0
    
    if abs(R[2,0]) < 0.001:
        R[2, 0] = 0
        
    if abs(t[0]) < 0.01 or R[0,0] > 0.99:
        t[0] = 0
        t[1] = 0
         
    return R, t

def Fmatrix_(ori, sub):
    return f.findFundamentalMat(ori, sub, f.FM_RANSAC)

def linear_triangulation1(K, T, ori, sub):
    #parameter K: camera matirx.
    #parameter T: array of sets of camera poses.
    #parameter ori: current frame.
    #parameter sub: successive frame.
    
    ############################################
    
    #linear_triangulation
    X = np.zeros((4, ori.shape[0], 4))
    X_new = np.zeros((4, ori.shape[0], 4))
    for i in range(4):
        p1 = np.eye(3, 4)
        p2 = np.dot(np.dot(K, T[i][1]), np.hstack((np.eye(3), -T[i][0].reshape(3,1))))
        print(p2)
        
        H = np.vstack((np.hstack((T[i][1], T[i][0].reshape(3,1))), [0,0,0,1]))
            
        for t in range(ori.shape[0]):
            A = np.asarray([[np.dot(ori[t][0], p1[2]) - p1[0]], [np.dot(ori[t][1], p1[2]) - p1[1]], [np.dot(sub[t][0], p2[2]) - p2[0]], [np.dot(sub[t][1], p2[2]) - p2[1]]]).reshape(4, 4)
            u, s, v = np.linalg.svd(A)
            x_tri = v[-1,:]
            x_tri = x_tri/x_tri[-1]
            X[:,t,i] = x_tri
            X_new[:,t,i] = np.dot(H, X[:,t,i])
    return X, X_new


def linear_tri_disambiguatePoint(T):
    #parameter T: array of sets of camera poses.
    
    ############################################    
    
    ind = []
    
    for i in range(len(T)):
        if T[i][0][2] > 0:
            ind.append(i)
            
    R_n = []
    ind2 = []
    
    if len(ind) > 0:
        #print(len(ind))
        for i in range(len(ind)):
            R_temp = T[ind[i]][1]
            if R_temp[1][1] > 0.9 and abs(R_temp[0][1]) < 0.1 and abs(R_temp[1][0]) < 0.1 and abs(R_temp[1][2]) < 0.1 and abs(R_temp[2][1]) < 0.1:
                R_n.append(R_temp)
                ind2.append(ind[i])
        #print(len(ind2))        
        if len(ind2) > 0:
            R = R_n[0]
            t = np.array([T[ind2[0]][0][0], 0, T[ind2[0]][0][2]])
            y_min = abs(T[ind2[0]][0][1])
            
            for j in range(1, len(ind2)):
                cur_y = abs(T[ind2[j]][0][1])
                
                if cur_y < y_min:
                    y_min = cur_y
                    R = np.array(R_n[j])
                    t = np.array([T[ind2[j]][0][0], 0, T[ind2[j]][0][2]])
                    
            R[0][1] = 0
            R[1][0] = 0
            R[1][2] = 0
            R[2][1] = 0
            
            if abs(R[0][2]) < 0.001:
                R[0][2] = 0
    
            if abs(R[2][0]) < 0.001:
                R[2][0] = 0
        
            if abs(t[0]) < 0.01 or R[0][0] > 0.99:
                t[0] = 0
                t[1] = 0      
        else:
            R = np.eye(3)
            t = np.zeros(3)
        
    else:
        R = np.eye(3)
        t = np.zeros(3)
    
    return R, t
                    
            
            
        
        