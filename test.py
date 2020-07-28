import display_image as di
import model_setup as ms

fi = input("Input path to image file: ")

try:
    model = ms.Model(f1=20, f2=10, d1=20, input_dim=1, num_classes=10) # TODO insert parameters

    di.display_image(fi, model)
except:
    print("Oop")