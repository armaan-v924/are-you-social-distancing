import time #necessary to allow the person to take a picture with the camera module
import camera
import cv2
import os
from model_setup import *
from data import find_faces
vid = cv2.VideoCapture(0)

#loading screen
time.sleep(1)
print("Welcome to \"Are You Social Distancing?\"") #introduction/name of the program
time.sleep(2)

#main
c = 0
print("Commands:\n----------\n1 - Take a Picture via Camera\n2 - Upload an Image\n3 - Record a Video (WIP)\n4 - Quit")
func = 0
time.sleep(2)
#Supposed to be in a for loop. That way, you wouldn't have to keep typing python main.py after you finish uploading the picture.
while func != 4:
    try:
        func = int(input("Please enter a number to which method you prefer to upload the image: "))
    except ValueError: 
        print('Invalid Input. Please enter only \"1\" or \"2\" or \"3\" or \"4\"\n')
        func = 0
    except:
        time.sleep(2)
        print('Something went wrong. Please try again.\n')
        time.sleep(2)
        func = 0
    if func == 1:
        time.sleep(2)
        print("Taking a picture in 5...\r")
        time.sleep(1)
        print("4\r")
        time.sleep(1)
        print("3\r")
        time.sleep(1)
        print("2\r")
        time.sleep(1)
        print("1\r")
        time.sleep(1)
        print("0")
        
        cropped_faces, resized_crop = find_faces(camera.take_picture())
        num_wearing_masks = 0
        for face in resized_crop:
            predictions = model(resized_crop)
            if(predictions[0] > predictions[1]): #wearing mask
                num_wearing_masks += 1
        print(num_wearing_masks + " people wearing masks / ", len(resized_crop), " total people --> ", num_wearing_masks/len(resized_crop)) #print stats

        time.sleep(2)
        print()
        func = 0
    elif func == 2:
        time.sleep(2)
        cropped_faces, resized_crop = find_faces(input("Please enter the complete image file path:").strip('"'))
        num_wearing_masks = 0
        for face in resized_crop:
            predictions = model(resized_crop)
            if(predictions[0] > predictions[1]): #wearing mask
                num_wearing_masks += 1
        print(num_wearing_masks + " people wearing masks / ", len(resized_crop), " total people --> ", num_wearing_masks/len(resized_crop)) #print stats

        time.sleep(2)
        print()
        func = 0

    elif func == 3:
        time.sleep(2)
        print("Recording a video in 5...\r")
        time.sleep(1)
        print("4\r")
        time.sleep(1)
        print("3\r")
        time.sleep(1)
        print("2\r")
        time.sleep(1)
        print("1\r")
        time.sleep(1)
        print("0")
        vidframe = []
        if not vid.isOpened():
            raise IOError("Cannot open webcam")
        print("Press the esc button to stop recording the video\n")
        while True:
            ret, frame = vid.read()
            frame = cv2.resize(frame,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
            cv2.imshow('Input',frame)
            vidframe.append(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                vid.release()
                cv2.destroyAllWindows()
                break
        print("bees")
        #set two lists so that it includes cropped faces and resized images.
        cropped_faces = []
        resized_crops = []
        #adds a cropped face and the resized frame for each frame of the video.
        for v in vidframe:
            cropped_face, resized_crop = find_faces(v)
            cropped_faces.append(cropped_face)
            resized_crops.append(resized_crop)
        
        #now we itlerate over the frames to see which faces are wearing a mask. We are counting the maximum number of people shown in the recording.
        max_masks = 0
        max_people = 0
        model = Model(f1=20, f2=10, d1=20, input_dim=1, num_classes=2)
        for crop in resized_crops:
            num_wearing_masks = 0
            predictions = (crop[:,np.newaxis,:,:].astype(np.float32)) / 255.
            for face in crop:
                print(predictions)
                if(predictions[0] > predictions[1]): #wearing mask
                    num_wearing_masks += 1

            #the stats will be modified to indicate the maximum number of masks over the total number of people in the video.
            if max_masks < num_wearing_masks:
                max_masks = num_wearing_masks
            if max_people < len(crop):
                max_people = len(crop)

        print(max_masks + " people wearing masks / ", max_people, " total people --> ", max_masks/max_people) #print stats

        time.sleep(2)
        print()
        func = 0

    # Since the function is in a for loop, a shutdown module must be needed to break the loop and exit the program.
    elif func == 4:
        time.sleep(2)
        print("@2020 @therealshazam \n -----------------------")
        break


        













