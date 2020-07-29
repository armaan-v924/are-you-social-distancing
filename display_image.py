"""Functionality to display an image with a box around detected faces with labels to indicate matches or an “Unknown” label otherwise"""
from facenet_models import FacenetModel
import cv2
import data
import numpy as np
def display_image(images, model):
    """Uses data.find_faces and a trained model to identify faces in an image, draw
    green boxes around the faces with masks on, and draw red boxes around the faces without
    masks on.

    Resizes original image to be of height 1000 px or width 1000 px.
    Displays a legend next to the image with boxes drawn on the faces.

    CURRENTLY DOESN'T SUPPORT VIDEOS

    Parameters:
    -----------
    images: list; list of np.ndarrays from cv2.imread() for images to be displayed

    Returns:
    --------
    None 
    """
    legend = cv2.imread("legend.png")
    cv2.imshow("Legend", legend)

    for image in images:
        height = image.shape[0]
        width = image.shape[1]

        if height > width:
            sf = 1000 / height
        else:
            sf = 1000 / width

        bounding_boxes, resized_crop = data.find_faces(image)
        bounding_boxes = np.rint(bounding_boxes*sf).astype(np.int)

        pass_data = resized_crop[:,np.newaxis,:,:]
        pass_data = pass_data.astype(np.float32)
        pass_data /= 255.

        preds = model(pass_data)
        preds = np.argmax(preds, axis=1)

        image = cv2.resize(image, (np.rint(width*sf).astype(np.int), np.rint(height*sf).astype(np.int)))
        
        for box, img, pred in zip(bounding_boxes, resized_crop, preds):
            if pred==0:
                color = (0, 255, 0)
                # text = "Mask"
            else:
                color = (255, 0, 0)
                # text = "Fail"
            
            image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)

            # if box[1] > (height*sf) - 10:
            #     image = cv2.putText(image, text, (box[0], box[1]-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)
            # else:
            #     image = cv2.putText(image, text, (box[0], box[3]+15), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyWindow("Image")

