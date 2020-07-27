# Imports
from facenet_models import FacenetModel
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import camera

# Create Rectangles
def find_faces(image):
    """ Using facenet_models, locate faces in a given picture and create descriptor vecors
    Parameters:
    -----------
    image: Path to image file OR (X, Y, 3) numpy array of pixels
    
    Returns:
    --------
    Tuple: (cropped_faces: List of cropped faces numpy arrays (N, X, Y, 3), where N is number of identified faces
            resized_crop: cropped_faces resized to (N, 160, 160, 3), where N is number of identified faces)

    """
    # Format image
    if type(image).__module__ is not np.__name__:
        img = cv2.imread(image)
        img = img[:,:,::-1]
    else:
        img = image

    # Create model
    model = FacenetModel()

    # Detect Faces
    bounding_boxes, probabilities, landmarks = model.detect(img)
    for bound in bounding_boxes:
        bound[bound<0]=0

    # Create descriptors
    # descriptors = model.compute_descriptors(img, bounding_boxes)

    # Cropped Face
    cropped_face = [img[int(bounding_boxes[n][1]):int(bounding_boxes[n][3]), int(bounding_boxes[n][0]):int(bounding_boxes[n][2])] for n in range(bounding_boxes.shape[0])]
    resized_crop = np.array([cv2.resize(img, (160, 160)) for img in cropped_face])
    return cropped_face, resized_crop
