# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 17:13:51 2019

@author: shiha
"""

import numpy as np
import random

def getfitscore(F, ori, sub):
    #parameter F: fundamental matrix.
    #parameter ori: current frame.
    #parameter sub: successive frame.
    
    ################################
    
    score = 0
    threh = 3

    chose_id = []
    sz = ori.shape[0]
    for j in range(sz):
        x1 = np.hstack((ori[j], 1)).reshape(-1, 1)
        x2 = np.hstack((sub[j], 1))
            
        temp = abs(np.dot(np.dot(x2, F), x1))
        epline = np.dot(F, x1)
        temp_err= temp/(np.sqrt(epline[0]**2 + epline[1]**2))
        
        if temp_err < threh:
            score = score + 1
            chose_id.append(j)
    return score/sz, chose_id