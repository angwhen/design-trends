import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import pandas as pd
import pickle, math, random
from time import time
import cv2
from collections import Counter
from functools import reduce
import colorsys
import MonochromeDetector

try:
    DATA_PATH  = open("data_location.txt", "r").read().strip()
except:
    DATA_PATH = "."

try:
    monochrome_list = set(pickle.load(open("%s/data/monochrome_list_%s.p"%(DATA_PATH,"hsv"),"rb")))
except:
    monochrome_list = None

try:
    list_of_predom_faces_fnums = set(pickle.load(open("%s/data/predom_faces_list.p"%(DATA_PATH),"rb")))
except:
    list_of_predom_faces_fnums = None

def get_people_cutout(masks,people_indices):
    sum_mask = masks[people_indices[0]]
    for ind in people_indices[1:]:
        sum_mask += masks[ind]
    return sum_mask

def get_skin_mask(fnum):
    try:
        return pickle.load(open("%s/data/images/mask_rcnn_results/skin_masks/%d.png"%(DATA_PATH,fnum),"rb"))
    except:
        return None

def get_pixels_in_file(fnum, color_rep = "rgb",remove_monochrome=False, remove_predom_faces = False, remove_skin=False):
    global monochrome_list, list_of_predom_faces_fnums

    if remove_predom_faces:
        if list_of_predom_faces_fnums is None:
            import PredomFaceDetector
            list_of_predom_faces_fnums = PredomFaceDetector.save_predom_faces_fnums_list()
        if fnum in list_of_predom_faces_fnums:
            return []

    if remove_monochrome:
        if monochrome_list is None:
            import MonochromeDetector
            monochrome_list = MonochromeDetector.save_monochrome_fnums_list()
            MonochromeDetector.make_react_code(method = "hsv")
        if fnum in monochrome_list:
            return []
    try:
        res = pickle.load(open("%s/data/images/mask_rcnn_results/res_%d.p"%(DATA_PATH,fnum),"rb"))
    except:
        return []

    masks, ids = res[1], res[2]
    people_indices = [i for i in range(0,masks.shape[0]) if ids[i] == 0]
    if len(people_indices) == 0:
        return []

    im = cv2.imread("%s/data/images/smaller_images/%d.jpg"%(DATA_PATH,fnum))
    if (im.shape[0] != masks.shape[1] or im.shape[1] != masks.shape[2]):
        print ("Dimensional problem on %d, image:%d, %d vs masks: %d, %d"%(fnum, im.shape[0],im.shape[1],masks.shape[1],masks.shape[2]))
        return []

    if color_rep == "hsv":
        im = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    elif color_rep == "rgb":
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        raise  ValueError('invalid color representation')

    people_cutout = get_people_cutout(masks,people_indices)
    if remove_skin:
        skin_mask = get_skin_mask(fnum)
        if skin_mask is not None:
            people_cutout = cv2.bitwise_and(people_cutout,skin_mask)

    my_pixels = im[ people_cutout != 0]
    return shuffle(my_pixels, random_state=0)

def get_pixels_in_fnums(fnums, color_rep="rgb", remove_monochrome=False, remove_predom_faces = False, remove_skin=False):
    all_pixels = []
    for fnum in fnums:
        my_pixels = get_pixels_in_file(fnum, color_rep=color_rep, remove_monochrome=remove_monochrome, remove_predom_faces=remove_predom_faces,remove_skin=remove_skin)
        all_pixels.extend(my_pixels)
    return shuffle(all_pixels,random_state=0)[:360000]

def get_pixels_dict_info_string(color_rep="rgb", remove_monochrome=False, remove_predom_faces = False, remove_skin=False):
    info_string = "_color%s"%(color_rep)
    if remove_monochrome:
        info_string += "_removemonochrome"
    if remove_predom_faces:
        info_string += "_removepredomfaces"
    if remove_skin:
        info_string += "_removeskin"
    return info_string

def get_year_to_pixels_dict(color_rep="rgb", remove_monochrome=False, remove_predom_faces = False, remove_skin=False):
    info_string =  get_pixels_dict_info_string( color_rep=color_rep, remove_monochrome=remove_monochrome, remove_predom_faces=remove_predom_faces,remove_skin=remove_skin)
    try:
        return pickle.load(open("%s/data/saved_pixels/year_to_pixels_dict%s.p"%(DATA_PATH,info_string),"wb"))
    except:
        year_to_fnums_dict=pickle.load(open("%s/data/basics/year_to_fnums_dict.p"%DATA_PATH,"rb"))
        year_to_pixels_dict= {}
        for year in year_to_fnums_dict.keys():
            year_pixels = get_pixels_in_fnums(year_to_fnums_dict[year], color_rep=color_rep, remove_monochrome=remove_monochrome, remove_predom_faces=remove_predom_faces,remove_skin=remove_skin)
            if len(year_pixels) == 0:
                print ("year %d has no valid pixels to use"%year)
                continue
            year_to_pixels_dict[year] = year_pixels
        pickle.dump(year_to_pixels_dict,open("%s/data/saved_pixels/year_to_pixels_dict%s.p"%(DATA_PATH,info_string),"wb"))
        return year_to_pixels_dict

def get_fnum_to_pixels_dict_and_all_colors(color_rep="rgb", remove_monochrome=False, remove_predom_faces = False, remove_skin=False):
    info_string =  get_pixels_dict_info_string( color_rep=color_rep, remove_monochrome=remove_monochrome, remove_predom_faces=remove_predom_faces,remove_skin=remove_skin)
    try:
        fnum_to_pixels_dict = pickle.load(open("%s/data/saved_pixels/fnum_to_pixels_dict%s.p"%(DATA_PATH,info_string),"wb"))
        all_colors= pickle.load(open("%s/data/saved_pixels/all_colors%s.p"%(DATA_PATH,info_string),"wb"))
    except:
        fnums_list = pickle.load(open("%s/data/basics/fnums_list.p"%DATA_PATH,"rb"))
        fnum_to_pixels_dict = {}
        all_colors = []
        for fnum in fnums_list:
            curr_pixels = get_pixels_in_file(fnum , color_rep=color_rep, remove_monochrome=remove_monochrome, remove_predom_faces=remove_predom_faces,remove_skin=remove_skin)
            if len(curr_pixels) != 0:
                num_to_pixels_dict[fnum] = curr_pixels
                all_colors.extend(curr_pixels)
        all_colors = shuffle(np.array(all_colors), random_state=0)[:3600000]
        pickle.dump(year_to_pixels_dict,open("%s/data/saved_pixels/fnum_to_pixels_dict%s.p"%(DATA_PATH,info_string),"wb"))
        pickle.dump(all_colors,open("%s/data/saved_pixels/all_colors%s.p"%(DATA_PATH,info_string),"wb"))

    return fnum_to_pixels_dict, all_colors
