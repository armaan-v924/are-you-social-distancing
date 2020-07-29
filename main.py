import time #necessary to allow the person to take a picture with the camera module
import camera
import cv2
import os
import pyaudio
import wave
import model_setup as ms
from model_setup import *
from data import find_faces
from facenet_models import FacenetModel
vid = cv2.VideoCapture(0)

#loading screen

print("\nWelcome to \"Are You Social Distancing?\"\n") #introduction/name of the program

#main
model = Model(f1=20, f2=10, d1=20, input_dim=1, num_classes=2)
model.load_model("trained_parameters.npz")
model2 = FacenetModel()
c = 0
print("Commands:\n----------\n1 - Take a Picture via Camera\n2 - Upload an Image\n3 - Capture a Video\n4 - Upload a Video\n5 - Quit\n")
func = 0
time.sleep(1)
#Supposed to be in a for loop. That way, you wouldn't have to keep typing python main.py after you finish uploading the picture.
while func != 5:
    try:
        func = int(input("Please enter a number to which method you prefer to upload the image: "))
    except ValueError: 
        print('Invalid Input. Please enter only \"1\" or \"2\" or \"3\" or \"4\" or \"5\"\n')
        func = 0
    except:
        print('Something went wrong. Please try again.\n')
        print()
        func = 0
    if func == 1:
        #Take a picture using the camera
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
        
        bb, cropped_faces, resized_crop = find_faces(camera.take_picture(),model2)
        if(type(resized_crop) == int and resized_crop == 0):
                print("No faces detected")
        else:
            num_wearing_masks = 0
            for face in resized_crop:
                convertedOne, convertedTwo = ms.convert_data(face.reshape(1, 160, 160)) #have to reshape for one image and send to convert data to normalize
                converted = np.append(convertedOne, convertedTwo, axis=0)
                predictions = model(converted)
                
                print("PREDICTIONS = ", predictions)
                if(predictions[0,1] > predictions[0,0]): #wearing mask
                    num_wearing_masks += 1
            print(num_wearing_masks, " people wearing masks / ", len(resized_crop), " total people --> ", float(num_wearing_masks/len(resized_crop))*100, "%") #print stats

        print()
        func = 0
    elif func == 2:
        bb, cropped_faces, resized_crop = find_faces(input("Please enter the complete image file path:").strip('"'),model2)
        if(type(resized_crop) == int and resized_crop == 0):
                print("No faces detected")
        else:
            num_wearing_masks = 0
            for face in resized_crop:
                convertedOne, convertedTwo = ms.convert_data(face.reshape(1, 160, 160)) #have to reshape for one image and send to convert data to normalize
                converted = np.append(convertedOne, convertedTwo, axis=0)
                predictions = model(converted)
                
                print("PREDICTIONS = ", predictions)
                if(predictions[0,1] > predictions[0,0]): #wearing mask
                    num_wearing_masks += 1
            print(num_wearing_masks, " people wearing masks / ", len(resized_crop), " total people --> ", num_wearing_masks/len(resized_crop)) #print stats

        print()
        func = 0

    elif func == 3:
        #siren set up
        filename = 'siren.wav'
        chunk = 1024  

        wf = wave.open(filename, 'rb')
        p = pyaudio.PyAudio()

        # Open a .Stream object to write the WAV file to
        # 'output = True' indicates that the sound will be played rather than recorded
        stream = p.open(format = p.get_format_from_width(wf.getsampwidth()),
                        channels = wf.getnchannels(),
                        rate = wf.getframerate(),
                        output = True)

        # Read data in chunks
        data = wf.readframes(chunk)

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
            print()
            func = 0
        print("Press the 'q' button to stop capturing the video\n")
        while True:
            ret, frame = vid.read()
            frame = cv2.resize(frame,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
            cv2.imshow('Input',frame)
            vidframe.append(frame)
            width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
            height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
            #each frame calculate # with/without masks (live)
            bb, cropped_faces, resized_crop = find_faces(frame,model2)
            num_wearing_masks = 0
            if(type(resized_crop) == int and resized_crop == 0):
                frame = cv2.putText(frame, "No faces detected", (30,30), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (255, 0, 0), 1, cv2.LINE_AA)
            else:
                counter = 0
                for face in resized_crop:
                    convertedOne, convertedTwo = ms.convert_data(face.reshape(1, 160, 160)) #have to reshape for one image and send to convert data to normalize
                    converted = np.append(convertedOne, convertedTwo, axis=0)
                    predictions = model(converted)
                    if(predictions[0,1] > predictions[0,0]): #wearing mask
                        num_wearing_masks += 1
                        frame = cv2.rectangle(frame, (bb[counter][0],bb[counter][1]), (bb[counter][2], bb[counter][3]),
                                              (0, 255, 0), 2)
                        frame = cv2.putText(frame, 'MASK', (bb[counter][0],bb[counter][1]), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    else:
                        frame = cv2.rectangle(frame, (bb[counter][0], bb[counter][1]), (bb[counter][2], bb[counter][3]),
                                              (0, 0, 255), 2)
                        frame = cv2.putText(frame, 'NO MASK', (bb[counter][0], bb[counter][1]), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    counter+=1

                percent_wearing_masks = num_wearing_masks/len(resized_crop)*100
                frame = cv2.putText(frame, (str(percent_wearing_masks) + "% wearing masks"), (30,30), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (255, 0, 0), 1, cv2.LINE_AA)
                if(percent_wearing_masks != 100) and data != '':
                    #AAAAAAAAAAAAA
                    stream.write(data)
                    data = wf.readframes(chunk)
                    percent_wearing_masks = 100

            cv2.imshow('Input', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                vid.release()
                cv2.destroyAllWindows()
                break


        print()
        func = 0
        stream.close()
        p.terminate()

    elif func == 4:
        pass
    # Since the function is in a for loop, a shutdown module must be needed to break the loop and exit the program.
    elif func == 5:
        time.sleep(2)
        print("@2020 @therealshazam\n-----------------------")
        break



        













