# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:10:31 2019

@author: shiha
"""

import numpy as np
import random


def Ematrix(F, K):
    #parameter F: fundamental matrix.
    #parameter K: camera matrix.
    
    ################################
    
    E = np.dot(np.dot(K.T, F), K)
    
    u, d, v = np.linalg.svd(E)
    d = np.diag([1, 1, 0])
    E = np.dot(np.dot(u, d), v)
    E = E/np.linalg.norm(E)
    
    return E