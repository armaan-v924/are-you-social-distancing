"""Functionality to display an image with a box around detected faces with labels to indicate matches or an “Unknown” label otherwise"""
import cv2
import data
import numpy as np
import model_setup as ms

def convert_image(image, model, bounding_boxes, resized_crop, bgr=True, resize=True):
    """Uses faces found from data.find_faces and a trained model to, draw green boxes 
    around the faces with masks on, and draw red boxes around the faces without masks on.

    If desired, resizes original image to be of height 1000 px or width 1000 px.

    Parameters:
    -----------
    image: np.ndarray, describes image to be displayed
    model: Model, trained model that will predict mask/no mask category for each face
    bounding_boxes: list, list of coordinates for each face (returned by data.find_faces)
    resized_crop: list, list of faces in image, (returned by data.find_faces)
    bgr: boolean (optional), True = bgr images, need to be converted (cv2.readimg was used)
                             False = rgb image, does not need to be converted
    resize: boolean (optional), True = resize image to be of height/width 1000px if original
                                image had height/width smaller than 500px
                                False = don't resize image
    Returns:
    --------
    np.ndarray
    Describes image that now has boxes around faces and text describing mask/no mask
    """
    height = image.shape[0]
    width = image.shape[1]
    
    if resize and (height < 500 or width < 500):
        if height > width:
            sf = 1000 / height
        else:
            sf = 1000 / width
    else:
        sf = 1

    image = cv2.resize(image, (np.rint(width*sf).astype(np.int), np.rint(height*sf).astype(np.int)))
    bounding_boxes = np.rint(bounding_boxes*sf).astype(np.int)

    convertedOne, convertedTwo = ms.convert_data(resized_crop)
    converted = np.append(convertedOne, convertedTwo, axis=0)

    preds = model(converted)
    preds = np.argmax(preds, axis=1)
    num_wearing_masks = np.count_nonzero(preds)

    for box, pred in zip(bounding_boxes, preds):
        if pred==1:
            color = (0, 255, 0)
            text = "Mask"
        else:
            if bgr:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            text = "No Mask"
        
        image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)

        if box[1] > (height*sf) - 10:
            image = cv2.putText(image, text, (box[0], box[1]-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)
        else:
            image = cv2.putText(image, text, (box[0], box[3]+15), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
    if bgr:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, num_wearing_masks