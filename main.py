import time #necessary to allow the person to take a picture with the camera module
import camera
from data import find_faces

#loading screen
time.sleep(1)
print("Welcome to \"Are You Social Distancing?\"") #introduction/name of the program
time.sleep(2)

#main
c = 0
print("Commands:\n----------\n1 - Take a Picture via Camera\n2 - Upload an Image\n3 - Quit")
func = 0
time.sleep(2)
#Supposed to be in a for loop. That way, you wouldn't have to keep typing python main.py after you finish uploading the picture.
while func != 3:
    try:
        func = int(input("Please enter a number to which method you prefer to upload the image: "))
    except ValueError: 
        print('Invalid Input. Please enter only \"1\" or \"2\" or \"3\"\n')
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
        find_faces(camera.take_picture())
        time.sleep(2)
        print()
        func = 0
    elif func == 2:
        time.sleep(2)
        find_faces(input("Please enter the complete image file path:").strip('"'))
        time.sleep(2)
        print()
        func = 0

    # Since the function is in a for loop, a shutdown module must be needed to break the loop and exit the program.
    elif func == 3:
        time.sleep(2)
        print("@2020 @therealshazam \n -----------------------")
        break


        













