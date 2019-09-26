from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
import pandas as pd
import numpy as np
import pickle
import os
import cv2

df =  pd.read_csv("data/url_title_and_file_data.csv")
fnames_list = df[["file_name"]].values.tolist()
finished_peopled_ims =set(os.listdir("data/images/mask_rcnn_results/people_seg_images/"))
finished_masked_ims =set(os.listdir("data/images/mask_rcnn_results/"))
for fname in fnames_list:
    fname_num = fname[0].split("/")[-1]
    fname_num = (int) (fname_num.split(".jpg")[0])
    print (fname_num)
    if "%d.png"%fname_num in finished_peopled_ims:
        continue
    if not "%d.png"%fname_num in finished_masked_ims:
        continue
    try:
        res = pickle.load(open("data/images/mask_rcnn_results/res_%d.p"%fname_num,"rb"))
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

    im = cv2.imread("data/images/smaller_images/%d.jpg"%fname_num)
    if (im.shape[0] != masks.shape[1] or im.shape[1] != masks.shape[2]):
        print ("some dimensional problem with the mask and image for this one")
    height,width,depth = im.shape
    circle_img = masks[people_indices[0]]
    for i in range(1,len(people_indices)):
        circle_img += masks[people_indices[i]]
    sub = np.true_divide(im,5)
    im = im - sub
    im[circle_img == 0] = [0, 0, 0]
    im  = im + sub
    cv2.imwrite("data/images/mask_rcnn_results/people_seg_images/%d.png"%fname_num, im)
