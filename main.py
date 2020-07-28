import time #necessary to allow the person to take a picture with the camera module
import camera
import cv2
from data import find_faces
vid = cv2.VideoCapture(0)

#loading screen
time.sleep(1)
print("Welcome to \"Are You Social Distancing?\"") #introduction/name of the program
time.sleep(2)

#main
c = 0
print("Commands:\n----------\n1 - Take a Picture via Camera\n2 - Upload an Image\n3 - Record a Video\n4 - Quit")
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
        if not vid.isOpened():
            raise IOError("Cannot open webcam")
        while True:
            ret, frame = vid.read()
            frame = cv2.resize(frame,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
            cv2.imshow('Input',frame)

            c = cv2.waitKey(1)
            if c == 27:
                break
        
        vid.release()
        cv2.destroyAllWindows()

        cropped_faces, resized_crop = find_faces(vid)
        num_wearing_masks = 0
        for face in resized_crop:
            predictions = model(resized_crop)
            if(predictions[0] > predictions[1]): #wearing mask
                num_wearing_masks += 1
        print(num_wearing_masks + " people wearing masks / ", len(resized_crop), " total people --> ", num_wearing_masks/len(resized_crop)) #print stats

        time.sleep(2)
        print()
        func = 0

    # Since the function is in a for loop, a shutdown module must be needed to break the loop and exit the program.
    elif func == 4:
        time.sleep(2)
        print("@2020 @therealshazam \n -----------------------")
        break


        













