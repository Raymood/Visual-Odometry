# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 15:18:04 2019

@author: shiha
"""

import numpy as np
import matplotlib.pyplot as plt



def Fmatrix(ori, sub):
    #ori, sub stand for the coordinates of correspondences between two frames respectively
    #T1 = np.hstack((ori, 1)).reshape(1, 3)
    #T2 = np.hstack((sub, 1)).reshape(3, 1)
    
    #perform the translation
    center_ori = np.mean(ori, axis = 0)
    center_sub = np.mean(sub, axis = 0)
    
    ori_T = ori - center_ori 
    sub_T = sub - center_sub
    
    #dis_mean_ori = np.sum(np.sqrt(np.add(np.square(ori_T[:, 0]), np.square(ori_T[:, 1]))))/ori.shape[0]
    #dis_mean_sub = np.sum(np.sqrt(np.add(np.square(sub_T[:, 0]), np.square(sub_T[:, 1]))))/sub.shape[0]
    
    #dis_mean_sub = np.mean(np.sqrt(np.sum(np.square(sub_T), axis = 1)))
    #dis_mean_ori = np.mean(np.sqrt(np.sum(np.square(ori_T), axis = 1)))
    dis_mean_sub = np.sqrt(np.sum(np.sum(np.square(sub_T), axis = 1)))/sub.shape[0]
    dis_mean_ori = np.sqrt(np.sum(np.sum(np.square(ori_T), axis = 1)))/sub.shape[0]
    scale_ori = np.sqrt(2)/dis_mean_ori
    scale_sub = np.sqrt(2)/dis_mean_sub
    #print(dis_mean_sub)

    ori_T = scale_ori * ori_T
    sub_T = scale_sub * sub_T
    
    T1 = np.asarray([[scale_ori, 0, -scale_ori * center_ori[0]], [0, scale_ori, -scale_ori * center_ori[1]], [0, 0, 1]])
    T2 = np.asarray([[scale_sub, 0, -scale_sub * center_sub[0]], [0, scale_sub, -scale_sub * center_sub[1]], [0, 0, 1]])
    
    A = np.asarray([np.array(ori_T[:,0]*sub_T[:,0]).reshape(-1, 1), np.array(ori_T[:,0]*sub_T[:,1]).reshape(-1, 1),
                    ori_T[:, 0].reshape(-1, 1), np.array(ori_T[:,1]*sub_T[:,0]).reshape(-1, 1), np.array(ori_T[:,1]*sub_T[:,1]).reshape(-1, 1),
                    ori_T[:, 1].reshape(-1, 1), sub_T[:, 0].reshape(-1, 1), sub_T[:, 1].reshape(-1, 1), np.ones((ori_T.shape[0], 1))])
    A = A[:,:, -1].T

    u, s, v = np.linalg.svd(A)
    F_3 = v[-1,:].reshape(3,3)
    #print(F_3)
    F_3 = F_3/np.linalg.norm(F_3)
    u1, s1, v1 = np.linalg.svd(F_3)

    s1 = np.diag(s1)
    s1[2,2] = 0
    F_2 = np.dot(np.dot(u1, s1), v1)
    
    F = np.dot(np.dot(T2.T, F_2), T1).T
    #print(F)

    return F, T1, T2, ori_T, sub_T
    
    
    
    
    
    
    