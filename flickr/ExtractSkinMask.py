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

#https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
def get_face_histograms(fnum):#size of image, 0 where no face, 1 where is face
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread("%s/data/images/smaller_images/%d.jpg"%(DATA_PATH,fnum))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # face cascade needs grayscale
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    histograms = []
    for (x, y, w, h) in faces:
        crop_face = img[y+int(h/5):y+4*int(h/5), x+int(w/5):x+4*int(w/5)]  # crop abit more tightly
        crop_face = cv2.blur(crop_face,(int(w/4),int(h/4)))
        histr = cv2.calcHist([crop_face],[0,1], mask=None, histSize=[80, 256], ranges=[0, 180, 0, 256] )
        histograms.append(histr)
    return histograms
#https://www.learnopencv.com/blob-detection-using-opencv-python-c/
def remove_blobs(mask):
    print ("Removing small blobs")
    im = cv2.imread("%s/data/images/smaller_images/%d.jpg"%(DATA_PATH,fnum), cv2.IMREAD_GRAYSCALE)

    params = cv2.SimpleBlobDetector_Params()
    detector = cv2.SimpleBlobDetector_create(params)


    # Detect blobs.
    keypoints = detector.detect(mask)
    im=cv2.bitwise_not(im)
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show keypoints
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)
# 1) https://nalinc.github.io/blog/2018/skin-detection-python-opencv/
# 1) tends to get much more pixels than is actually skin
# 2) #https://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/
# 2) Using the zebra stuff from this for the histogram
def convolve(B, r):
    D = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r,r))
    cv2.filter2D(B, -1, D, B)
    return B
def get_skin_mask(fnum):
    im = cv2.imread("%s/data/images/smaller_images/%d.jpg"%(DATA_PATH,fnum))
    im = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    print (im.shape)
    skin_mask = np.zeros((im.shape[0],im.shape[1]),dtype=np.uint8)
    histograms = get_face_histograms(fnum)
    for histr in histograms:
        B = cv2.calcBackProject([im], channels=[0,1], hist=histr, ranges=[0,180,0,256], scale=1)
        B = convolve(B, r=5)
        skin_mask += B

    skin_mask[skin_mask > 0] = 1
    return skin_mask



#im = get_image_with_non_people_blacked_out(5)
#get_face_histograms(154)
fnum = 26 #5, 154
skin_mask = get_skin_mask(fnum)
remove_blobs(skin_mask)
print (skin_mask.shape)
print (skin_mask)
im = cv2.imread("%s/data/images/smaller_images/%d.jpg"%(DATA_PATH,fnum))
new_im =cv2.bitwise_and(im,im, mask = skin_mask)
cv2.imshow("new image",new_im)
cv2.waitKey(0)