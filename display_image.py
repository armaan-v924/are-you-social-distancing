"""Functionality to display an image with a box around detected faces with labels to indicate matches or an “Unknown” label otherwise"""
import cv2
import data
import numpy as np
import model_setup as ms
import torch
import torch_model as tm
import no_depth_dist as ndd

def convert_image(image, model, landmarks, bounding_boxes, resized_crop, bgr=True, resize=True):
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    height = image.shape[0]
    width = image.shape[1]
    
    if resize and (height < 500 or width < 500):
        if height > width:
            sf = 1000 / height
        else:
            sf = 1000 / width
    else:
        sf = 1

    if bgr:
        red = (255, 0, 0)
    else:
        red = (0, 0, 255)

    image = cv2.resize(image, (np.rint(width*sf).astype(np.int), np.rint(height*sf).astype(np.int)))
    bounding_boxes = np.rint(bounding_boxes*sf).astype(np.int)
    landmarks = np.rint(landmarks*sf).astype(np.int)

    convertedOne, convertedTwo = tm.convert_data(resized_crop)
    converted = np.append(convertedOne, convertedTwo, axis=0)

    preds = model(torch.Tensor(converted).to(device))
    preds = np.argmax(preds.data, axis=1)
    num_wearing_masks = np.count_nonzero(preds)

    for box, pred in zip(bounding_boxes, preds):
        if pred==1:
            color = (0, 255, 0)
            text = "Mask"
        else:
            color = red
            text = "No Mask"
        
        image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)

        if box[1] > (height*sf) - 10:
            image = cv2.putText(image, text, (box[0], box[1]-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)
        else:
            image = cv2.putText(image, text, (box[0], box[3]+15), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
    
    # if len(resized_crop) > 1:
    #     for i in range(len(resized_crop)-1):
    #         if ndd.r_u_sd_in_2d(landmarks[i:i+2], threshold=72) == False:
    #             image = cv2.line(image,(landmarks[i, 2][0],landmarks[i, 2][1]),
    #                             (landmarks[i+1, 2][0],landmarks[i+1, 2][1]),red,2)

    if bgr:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, num_wearing_masks