from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
import pandas as pd
import numpy as np
import pickle
import os
import cv2

try:
    DATA_PATH  = open("data_location.txt", "r").read().strip()
except:
    DATA_PATH = "."

fnums_list = pickle.load(open("%s/data/basics/fnums_list.p"%DATA_PATH,"rb"))
finished_peopled_ims =set(os.listdir("%s/data/images/mask_rcnn_results/people_seg_images/"%DATA_PATH))
finished_masked_ims =set(os.listdir("%s/data/images/mask_rcnn_results/"%DATA_PATH))
for fnum in fnums_list:
    print (fnum)
    if "%d.png"%fnum in finished_peopled_ims:
        continue
    if not "res_%d.png"%fnum in finished_masked_ims:
        continue
    try:
        res = pickle.load(open("%s/data/images/mask_rcnn_results/res_%d.p"%(DATA_PATH,fnum),"rb"))
    except:
        continue

    masks = res[1]
    ids = res[2]

    people_indices = []
    for i in range(0,masks.shape[0]): #the masks we have for people
        if ids[i] == 0:
            people_indices.append(i)

    if len(people_indices) == 0:
        continue

    im = cv2.imread("%s/data/images/smaller_images/%d.jpg"%(DATA_PATH,fnum))
    if (im.shape[0] != masks.shape[1] or im.shape[1] != masks.shape[2]):
        print ("some dimensional problem with the mask and image for this one")
        continue
    height,width,depth = im.shape
    circle_img = masks[people_indices[0]]
    for i in range(1,len(people_indices)):
        circle_img += masks[people_indices[i]]
    sub = np.true_divide(im,5)
    im = im - sub
    im[circle_img == 0] = [0, 0, 0]
    im  = im + sub
    cv2.imwrite("%s/data/images/mask_rcnn_results/people_seg_images/%d.png"%(DATA_PATH,fnum), im)
