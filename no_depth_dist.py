#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np

def r_u_sd_in_2d(landmarks, threshold=72):
    """
    determines if people in photo are at least a certain distance apart 
    *assumes they are at the same depth *
    
    Parameters:
    -----------
    landmarks: 3d list where dims are [face, landmark, x_y]
    threshold: int, required distance in inches (6ft = 72 inches)
    
    Returns:
    --------
    are_you: True/False --> indicates whether threshhold is passed for all faces 
    """
    
    actual_dist = 2.48 #average distance between eyes in inches
    
    #for one face find ratio between pixels btw eyes and actual_dist
    #landmarks[face, 0] = eye1, landmarks[face, 1] = eye2

    dist_x = landmarks[0,0,0] - landmarks[0,1,0]
    dist_y = landmarks[0,0,1] - landmarks[0,0,1]
    print(dist_x,dist_y)
    dist_in_pixels = np.sqrt(dist_x ** 2 + dist_y ** 2)
    ratio = actual_dist / dist_in_pixels
    
    #compare to other faces
    are_you = True
    for face_idx in range(len(landmarks)-1):
        #nose --> x, y
        nose = landmarks[face_idx, 2]

        other_nose = landmarks[face_idx+1, 2]
            
        #compute & compare
        distance_btw_faces_pxls = np.sqrt((nose[0] - other_nose[0]) ** 2 + (nose[1] - other_nose[1]) ** 2)
        distance_btw_faces_actual = distance_btw_faces_pxls * ratio
        print(distance_btw_faces_actual)
        if(distance_btw_faces_actual < threshold):
            are_you = False
            break
            
    return are_you

