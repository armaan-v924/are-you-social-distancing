import time
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
while func != 3:
    try:
        func = int(input("Please enter a number to which method you prefer to upload the image: "))
    except ValueError: 
        print('Invalid Input. Please enter only \"1\" or \"2\" or \"3\"\n')
        func = 0
    except:
        print('Something went wrong. Please try again.')
        func = 0
    if func == 1:
        time.sleep(2)
        print("Taking a picture in 5...")
        time.sleep(1)
        print("4")
        time.sleep(1)
        print("3")
        time.sleep(1)
        print("2")
        time.sleep(1)
        print("1")
        time.sleep(1)
        find_faces(camera.take_picture())
    elif func == 2:
        find_faces(input("Please enter the complete image file path:").strip('"'))

    # This will never run?
    elif func == 3:
        time.sleep(2)
        print("@2020 @therealshazam \n -----------------------")
        break
    print()
    func = 0


        













