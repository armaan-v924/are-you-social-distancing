"""Functionality to display an image with a box around detected faces with labels to indicate matches or an “Unknown” label otherwise"""
from facenet_models import FacenetModel
import cv2
import data
import numpy as np
def display_image(image, model):
    """Using camera.take_picture() and mf.match_face, plot image and display boxes & names around faces

    Parameters:
    -----------
    None

    Returns:
    --------
    None 
    """
    # for i in image:
    height = image.shape[0]
    bounding_boxes, resized_crop = data.find_faces(image)
    bounding_boxes = np.rint(bounding_boxes).astype(np.int)
    pass_data = resized_crop[:,np.newaxis,:,:]

    pass_data = pass_data.astype(np.float32)
    pass_data /= 255.
    preds = model(pass_data)
    preds = np.argmax(preds, axis=1)
    for box, img, pred in zip(bounding_boxes, resized_crop, preds):
        print(pred)
        if pred==0:
            color = (0, 255, 0)
            text = "Mask"
        else:
            color = (255, 0, 0)
            text = "No Mask"
        
        image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)

        if box[1] > height - 10:
            image = cv2.putText(image, text, (box[0], box[1]-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
        else:
            image = cv2.putText(image, text, (box[0], box[3]+15), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

