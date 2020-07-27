import data
import os
from os import path
import numpy as np

folder = input("Folder where pictures are stored: ") # Identify what to do

try: 
    value = int(input("Input 1 for masks and 0 for no masks: "))
    new_faces = np.array([])

    for filename in os.listdir(folder):
        cropped_face, resized_crop = data.find_faces(folder + "/" + filename)
        new_faces = np.append(new_faces, resized_crop)

    if value==1:
        if path.exists("with_masks.npy"):
            old_faces = np.load("with_masks.npy")
            new_faces = np.append(new_faces, old_faces)
        np.save("with_masks.npy", new_faces, allow_pickle=False)
    else:
        if path.exists("without_masks.npy"):
            old_faces = np.load("without_masks.npy")
            new_faces = np.append(new_faces, old_faces)
        np.save("without_masks.npy", new_faces, allow_pickle=False)
except ValueError: 
    print('Please enter only "1" for with masks, or "0" for no masks.')
except:
    print('Something went wrong. Please try again.') # Probably shouldn't run, but just in case I miss something


