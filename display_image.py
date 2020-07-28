"""Functionality to display an image with a box around detected faces with labels to indicate matches or an “Unknown” label otherwise"""
from facenet_models import FacenetModel
import cv2
import data
def display_image(image, model):
    """Using camera.take_picture() and mf.match_face, plot image and display boxes & names around faces

    Parameters:
    -----------
    None

    Returns:
    --------
    None 
    """
    for i in image:
        bounding_boxes, resized_crop = data.find_faces(image)
        preds = model(resized_crop)
        preds = np.argmax(preds, axis=1)
        for box, img, pred in zip(bounding_boxes, resized_crop, preds):
            if pred==0:
                color = (0, 255, 0)
                text = "Mask"
            else:
                color = (0, 0, 255)
                text = "No Mask"
            
            image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)

            if box[1] < 10:
                img = cv2.putText(img, text, (box[0], box[1]-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
            else:
                img = cv2.putText(img, text, (box[0], box[3]-15), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)

    cv2.imshow("Image", image)

