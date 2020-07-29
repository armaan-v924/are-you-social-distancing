import time #necessary to allow the person to take a picture with the camera module
import camera
import cv2
import os
import model_setup as ms
from model_setup import *
from data import find_faces
import display_image as di
from facenet_models import FacenetModel
vid = cv2.VideoCapture(0)

#loading screen

print("\nWelcome to \"Are You Social Distancing?\"\n") #introduction/name of the program

#main
model = Model(f1=20, f2=10, d1=20, input_dim=1, num_classes=2)
model.load_model("trained_parameters.npz")
model2 = FacenetModel()
c = 0
print("Commands:\n----------\n1 - Take a Picture via Camera\n2 - Upload an Image\n3 - Record a Video (WIP)\n4 - Upload a Video\n5 - Quit\n")
func = 0
#Supposed to be in a for loop. That way, you wouldn't have to keep typing python main.py after you finish uploading the picture.
while func != 5:
    try:
        func = int(input("Please enter a number to which method you prefer to upload the image: "))
    except ValueError: 
        print('Invalid Input. Please enter only \"1\" or \"2\" or \"3\" or \"4\" or \"5\"\n')
        func = 0
    except:
        # time.sleep(2)
        print('Something went wrong. Please try again.\n')
        # time.sleep(2)
        func = 0
    if func == 1:
        #Take a picture using the camera
        # time.sleep(2)
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
        
        image = camera.take_picture()

        bb, cropped_faces, resized_crop = find_faces(image,model2)
        if(type(resized_crop) == int and resized_crop == 0):
                print("No faces detected")
        else:
            di.show_image(image, model, bb, resized_crop, True)

        # time.sleep(2)
        print()
        func = 0
    elif func == 2:
        # time.sleep(2)
        image = cv2.imread(input("Please enter the complete image file path:").strip('"'))
        image = image[:,:,::-1]
        bb, cropped_faces, resized_crop = find_faces(image,model2)
        if(type(resized_crop) == int and resized_crop == 0):
                print("No faces detected")
        else:
            di.show_image(image, model, bb, resized_crop, True)

        # time.sleep(2)
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
            
            if ret == True:
                frame = cv2.resize(frame,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
                cv2.imshow('Input',frame)
                vidframe.append(frame)
                width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
                height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
                #each frame calculate # with/without masks (live)
                bb, cropped_faces, resized_crop = find_faces(frame,model2)
                num_wearing_masks = 0
                if(type(resized_crop) == int and resized_crop == 0):
                    print("No faces detected")
                else:
                    frame = di.convert_image(frame, model, bb, resized_crop, bgr=False, resize=False)
            else:
                break
            cv2.imshow('Input', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                vid.release()
                cv2.destroyAllWindows()
                break
        print("bees")
        
        #now we itlerate over the frames to see which faces are wearing a mask. We are counting the maximum number of people shown in the recording.
        max_masks = 0
        max_people = 0
        for crop in resized_crop:
            num_wearing_masks = 0
            predictions = (crop[:,np.newaxis,:,:].astype(np.float32)) / 255.
            for face in crop:
                print(predictions)
                if(predictions[1] > predictions[0]): #wearing mask
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
        vid = cv2.VideoCapture(input("Please enter the complete video file path: ").strip('"'))

        while (vid.isOpened()):
            ret, frame = vid.read()

            if ret == True:
                frame = cv2.resize(frame,None,fx=0.25,fy=0.25,interpolation=cv2.INTER_AREA)
                cv2.imshow('Input',frame)
                width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
                height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
                #each frame calculate # with/without masks (live)
                bb, cropped_faces, resized_crop = find_faces(frame,model2)
                num_wearing_masks = 0
                if(type(resized_crop) == int and resized_crop == 0):
                    print("No faces detected")
                else:
                    frame = di.convert_image(frame, model, bb, resized_crop, bgr=False, resize=False)

                cv2.imshow('Input', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): 
                    vid.release()
                    cv2.destroyAllWindows()
                    break
            else:
                break
        vid.release()
        cv2.destroyAllWindows()

        # time.sleep(2)
        print()
        func = 0
    elif func == 5:
        time.sleep(2)
        print("@2020 @therealshazam \n -----------------------")
        break


        













