from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
import pandas as pd
import numpy as np
import pickle
import os, cv2, math

try:
    DATA_PATH  = open("data_location.txt", "r").read().strip()
except:
    DATA_PATH = "."

def get_image_with_non_people_blacked_out(fnum):
    try:
        res = pickle.load(open("%s/data/images/mask_rcnn_results/res_%d.p"%(DATA_PATH,fnum),"rb"))
    except:
        return None
    masks, ids = res[1], res[2]

    people_indices = [i for i in range(0,masks.shape[0]) if ids[i] == 0]
    if len(people_indices) == 0:
        return None

    im = cv2.imread("%s/data/images/smaller_images/%d.jpg"%(DATA_PATH,fnum))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if (im.shape[0] != masks.shape[1] or im.shape[1] != masks.shape[2]):
        print ("Dimensional problem on %d, image:%d, %d vs masks: %d, %d"%(fnum, im.shape[0],im.shape[1],masks.shape[1],masks.shape[2]))
        return None

    people_img = masks[people_indices[0]]
    for i in range(1,len(people_indices)):
        people_img += masks[people_indices[i]]
    im[people_img == 0] = [0, 0, 0]
    return im

def get_face_mask(fnum):#size of image, 0 where no face, 1 where is face
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Read the input image
    img = cv2.imread("%s/data/images/smaller_images/%d.jpg"%(DATA_PATH,fnum))
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    print (type(faces))
    print (faces)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display the output
    cv2.imshow('img', img)
    cv2.waitKey()

#https://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/
def get_skin_mask(fnum):
    return ""

#im = get_image_with_non_people_blacked_out(5)
get_face_mask(5)
