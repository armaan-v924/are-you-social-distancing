"""Functionality to display an image with a box around detected faces with labels to indicate matches or an “Unknown” label otherwise"""
from facenet_models import FacenetModel
import cv2
import data
import numpy as np
import os

def convert_image(images, model, bgr=True):
    """Uses data.find_faces and a trained model to identify faces in an image, draw
    green boxes around the faces with masks on, and draw red boxes around the faces without
    masks on.

    Resizes original image to be of height 1000 px or width 1000 px.

    Parameters:
    -----------
    images: list; list of np.ndarrays from cv2.imread() for images to be displayed
    model: Model, trained model that will predict mask/no mask category for each face
    bgr: boolean, True=bgr images that need to be converted (if cv2.readimg was used)
                  False=rgb image that does not need to be converted
    Returns:
    --------
    list: list of converted (boxes drawn around faces) images 
    """
    for i, image in enumerate(images):
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
                text = "Mask"
            else:
                if bgr:
                    color = (255, 0, 0)
                else:
                    color = (0, 0, 255)
                text = "Fail"
            
            image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)

            if box[1] > (height*sf) - 10:
                image = cv2.putText(image, text, (box[0], box[1]-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)
            else:
                image = cv2.putText(image, text, (box[0], box[3]+15), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
        if bgr:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images[i] = image
    return images

def show_image(images, model, bgr):
    """Displays images with boxes drawn around faces (classified as wearing a mask
    or not wearing a mask); also displays legend for classification

    Resizes original image to be of height 1000 px or width 1000 px.

    Parameters:
    -----------
    images: list; list of np.ndarrays from cv2.imread() for images to be displayed
    model: Model, trained model that will predict mask/no mask category for each face
    bgr: boolean, True=bgr images that need to be converted (if cv2.readimg was used)
                  False=rgb image that does not need to be converted
    Returns:
    --------
    None 
    """
    images = convert_image(images, model, bgr)
    for image in images:
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyWindow("Image")
    cv2.destroyAllWindows()

def show_video(file_path, model):
    """Displays video with boxes drawn around faces (classified as wearing a mask
    or not wearing a mask); also displays legend for classification

    Resizes original video window to be of height 1000 px or width 1000 px.

    Parameters:
    -----------
    file_path: string, path to video that will be displayed
    model: Model, trained model that will predict mask/no mask category for each face
    Returns:
    --------
    None 
    """
    cap = cv2.VideoCapture(file_path)
    img_array = []

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            img_array.append(frame)
        else:
            break
    cap.release()
    # print("video loaded") # DEBUGGING MESSAGES
    # print(len(img_array))

    img_array = convert_image(img_array, model, False)
    # print("video edited") # DEBUGGING

    height, width, layers = img_array[0].shape
    size = (width, height)

    out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    # print("video saved") # DEBUGGING

    cap = cv2.VideoCapture('project.avi')

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            cv2.imshow('Video',frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        else:
            break
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
    os.remove('project.avi')
