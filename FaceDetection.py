import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("Image.png") #You can also use JPEG file
plt.imshow(img)
img.shape

import cv2 # Installation command for OpenCV package : pip install -c opencv-python
model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #Xml file download link is given in Readme
faces = model.detectMultiScale(img)
for face in faces:
    x,y,w,h = face
    im = cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,0), 2)
    
plt.imshow(img)
