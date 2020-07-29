import display_image as di
import model_setup as ms
import cv2
import os

fi = input("Input path to folder with images: ")

model = ms.Model(f1=20, f2=10, d1=20, input_dim=1, num_classes=2) # TODO insert parameters

di.show_video(fi, model)
# img = cv2.imread(fi)
# img = img[:,:,::-1]
# di.show_image([img], model, True)
