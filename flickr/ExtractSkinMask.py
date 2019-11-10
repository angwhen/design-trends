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
        crop_face_before_blur = img[y+int(h/6):y+5*int(h/6), x+int(w/6):x+5*int(w/6)]  # crop abit more tightly
        crop_face_blurred = cv2.blur(crop_face_before_blur,(int(w/4),int(h/4)))

        face_features = crop_face_before_blur-crop_face_blurred
        face_features = cv2.cvtColor(face_features,cv2.COLOR_HSV2BGR)

        face_features_gray = cv2.cvtColor(face_features, cv2.COLOR_BGR2GRAY)
        print (face_features_gray.shape)
        new = np.ones([face_features.shape[0],face_features.shape[1]],dtype=np.uint8)
        new[np.where(face_features_gray >50)] = 0

        crop_face_temp = cv2.cvtColor(crop_face_before_blur,cv2.COLOR_HSV2BGR)
        cv2.imshow("okay",cv2.bitwise_and(crop_face_temp,crop_face_temp, mask =new))
        cv2.waitKey(0)

        histr = cv2.calcHist([crop_face_blurred],[0,1], mask=new, histSize=[80, 256], ranges=[0, 180, 0, 256] )
        histograms.append(histr)
    return histograms

#https://www.learnopencv.com/blob-detection-using-opencv-python-c/
#https://stackoverflow.com/questions/30369031/remove-spurious-small-islands-of-noise-in-an-image-python-opencv
def remove_small_blobs(mask):
    from skimage import morphology
    import skimage
    small_piece = mask.shape[0]*mask.shape[1]/500
    processed = morphology.remove_small_objects(mask.astype(bool), min_size=small_piece, connectivity=5).astype(np.uint8)

    return processed

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
fnum = 136#26 27,,5, 154
skin_mask = get_skin_mask(fnum)
processed_skin_mask = remove_small_blobs(skin_mask)
print (skin_mask.shape)
print (skin_mask)

im = cv2.imread("%s/data/images/smaller_images/%d.jpg"%(DATA_PATH,fnum))
new_im =cv2.bitwise_and(im,im, mask = skin_mask)
cv2.imshow("new image",new_im)
cv2.waitKey(0)

new_im =cv2.bitwise_and(im,im, mask = processed_skin_mask)
cv2.imshow("new image 2",new_im)
cv2.waitKey(0)
