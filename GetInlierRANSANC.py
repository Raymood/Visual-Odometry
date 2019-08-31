# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 22:26:47 2019

@author: shiha
"""

import numpy as np
import random
from disambiguatepose import*
from EstimateFundamentalMatrix import*
from getfitnessscore import*


def F_RANSAC(ori, sub):
    n = 0
    threh = 0.7
    inliers = []
    F_B = 0
    for i in range(1200):
        index = random.sample(range(0, ori.shape[0]), 8)
        ori_8 = ori[index]
        sub_8 = sub[index]
        
        F_8, T1, T2, ori_T, sub_T = Fmatrix(ori_8, sub_8)
        inliers_temp = []
        for j in range(8):
            x1 = np.hstack((ori_8[j], 1)).reshape(-1, 1)
            x2 = np.hstack((sub_8[j], 1))
            
            temp = np.matmul(np.matmul(x2, F_8), x1)
            
            if abs(temp) < threh:
                inliers_temp.append(index[j])
                
        if n < len(inliers_temp):
            n = len(inliers_temp)
            inliers = inliers_temp
            bestinliers1 = ori_T
            bestinliers2 = sub_T
            F_B = np.matmul(np.matmul(T2.T, F_8), T1)
            if F_B.flat[-1] < 0:
                F_B = -F_B
            if n == 8:
                break
    F, _ = Fmatrix_(ori, sub)
    F_final = F
    return F_final, bestinliers1, bestinliers2

def F_RANSAC1(ori, sub):
    n = 0
    chose_final = []
    N = 2000
    F = np.zeros((3,3))
    old_score, _= getfitscore(F, ori, sub)
    p = 0.95
    sz = int(ori.shape[0])
    
    while n < N:
        index = random.sample(range(0, ori.shape[0]), 8)
        ori_8 = ori[index]
        sub_8 = sub[index]

        F_8, T1, T2, ori_T, sub_T = Fmatrix(ori_8, sub_8)
        new_score, chose_id = getfitscore(F_8, ori, sub)
        #print(F_8, new_score, ori_8)
        if new_score > old_score:
            F_final = F_8
            chose_final = chose_id
            #bestinliers1 = ori_8
            #bestinliers2 = sub_8
            N = min(N,np.log(1-p) / np.log(1 - new_score**8))
            old_score = new_score
        n = n + 1
    #print(len(chose_final))
    bestinliers1 = ori[chose_final]
    bestinliers2 = sub[chose_final]
    print(bestinliers1)
    if len(bestinliers1) >= 8:
        F_final = Fmatrix(bestinliers1, bestinliers2)
    else:
        F_final = F_8

    return F_final, bestinliers1, bestinliers2
                
                
            
    