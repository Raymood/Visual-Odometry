# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:25:18 2019

@author: shiha
"""

import numpy as np
import random

def get_pose(E):
    W = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    u, _, v = np.linalg.svd(E)
    
    R1 = np.dot(np.dot(u, W), v)
    R2 = np.dot(np.dot(u, W.T), v)
    C = u[:, -1]
    T = []
    
    if np.linalg.det(R1) < 0:
        R1 = -R1
        C = -C
        
    t1 = np.dot(-R1.T, C)    
    T.append((C, R1))
    t2 = np.dot(-R1.T, -C)
    T.append((-C, R1))
    
    if np.linalg.det(R2) < 0:
        R2 = -R2
        C = -C
        
    t3 = np.dot(-R2.T, C)
    T.append((C, R2))
    t4 = np.dot(-R2.T, -C)
    T.append((-C, R2))

    return T