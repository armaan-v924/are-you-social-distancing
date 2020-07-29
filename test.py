import display_image as di
import model_setup as ms
import cv2
import os

fi = input("Input path to folder with images: ")

model = ms.Model(f1=20, f2=10, d1=20, input_dim=1, num_classes=2) # TODO insert parameters

images = []
for filename in os.listdir(fi):
    img = cv2.imread(fi + "/" + filename)
    img = img[:,:,::-1]
    images.append(img)

di.display_image(images, model)
