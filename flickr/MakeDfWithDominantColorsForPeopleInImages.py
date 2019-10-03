from matplotlib import pyplot as plt
import pandas as pd
import pickle
import os
from ColorThiefModified import ColorThief
import re
import colorsys
from collections import deque
import math
import numpy as np
import cv2

def make_df():
    df =  pd.read_csv("data/url_title_and_file_data.csv")
    fnames_list = df[["file_name"]].values.tolist()

    palettes = pickle.load(open("data/color_palettes.p","rb"))

    results = []
    count = 0
    for fname in fnames_list:
        fname_num = fname[0].split("/")[-1]
        fname_num = (int) (fname_num.split(".jpg")[0])
        if fname_num in palettes:
            print ("already done with %d"%fname_num)
            continue
        try:
            res = pickle.load(open("data/images/mask_rcnn_results/res_%d.p"%fname_num,"rb"))
        except:
            continue
        masks = res[1]
        ids = res[2]
        scores = res[3]

        people_indices = []
        for i in range(0,masks.shape[0]): #the masks we have for people
            if ids[i] == 0:
                people_indices.append(i)

        if len(people_indices) == 0:
            continue

        print (fname_num)
        my_pixels = []
        inner_count = 0
        im = cv2.imread("data/images/smaller_images/%d.jpg"%fname_num)
        if (im.shape[0] != masks.shape[1] or im.shape[1] != masks.shape[2]):
            print ("some dimensional problem with the mask and image for this one")
            continue
        for ind in people_indices:
            curr_mask =  masks[ind]
            for row in range(0,curr_mask.shape[0]):
                for col in range(0,curr_mask.shape[1]):
                    if inner_count % 10 == 0:
                        my_pixels.append(im[row][col])
                    inner_count +=1
        ct = ColorThief(my_pixels)
        color_list = ct.get_palette()
        palettes[fname_num] = color_list

        if count % 5 == 0: #save frequently to avoid having to rerun too often
            pickle.dump(palettes,open("data/color_palettes.p","wb"))
            print ("current part saved")
        count +=1

    if len(results) != 0:
        pickle.dump(palettes,open("data/color_palettes.p","wb"))

"""
# not really done yet maybe dont need
def save_yearly_palettes():
    df =  pd.read_csv("data/url_title_and_file_data.csv")
    fnames_list = df[["file_name"]].values.tolist()

    palettes = pickle.load(open("data/yearly_color_palettes.p","rb"))

    results = []
    count = 0
    for fname in fnames_list:
        fname_num = fname[0].split("/")[-1]
        fname_num = (int) (fname_num.split(".jpg")[0])
        if fname_num in palettes:
            print ("already done with %d"%fname_num)
            continue
        try:
            res = pickle.load(open("data/images/mask_rcnn_results/res_%d.p"%fname_num,"rb"))
        except:
            continue
        orig_img = res[0]
        masks = res[1]
        ids = res[2]
        scores = res[3]

        people_indices = []
        for i in range(0,masks.shape[0]): #the masks we have for people
            if ids[i] == 0:
                people_indices.append(i)

        if len(people_indices) == 0:
            continue

        print (fname_num)
        my_pixels = []
        inner_count = 0
        for ind in people_indices:
            curr_mask =  masks[ind]
            for row in range(0,curr_mask.shape[0]):
                for col in range(0,curr_mask.shape[1]):
                    if inner_count % 10 == 0:
                        my_pixels.append(orig_img[row][col])
                    inner_count +=1
        ct = ColorThief(my_pixels)
        color_list = ct.get_palette()
        palettes[fname_num] = color_list

        pickle.dump(palettes,open("data/yearly_color_palettes.p","wb"))


    if len(results) != 0:
        pickle.dump(palettes,open("data/yearly_color_palettes.p","wb"))"""

def diff_score(prev_row,curr_row):
    sum_dist = 0
    curr_row =  [colorsys.rgb_to_hsv(c[0],c[1],c[2]) for c in curr_row]
    prev_row =  [colorsys.rgb_to_hsv(c[0],c[1],c[2]) for c in prev_row]
    for i in range(0,len(curr_row)):
        sum_dist += math.sqrt(math.pow((prev_row[i][0]/255.0-curr_row[i][0]/255.0),2)+math.pow((prev_row[i][1]/255.0-curr_row[i][1]/255.0),2)+math.pow((prev_row[i][2]/255.0-curr_row[i][2]/255.0),2))
        # diagonals
        sum_dist += math.sqrt(math.pow((prev_row[i][0]/255.0-curr_row[i-1][0]/255.0),2)+math.pow((prev_row[i][1]/255.0-curr_row[i-1][1]/255.0),2)+math.pow((prev_row[i-1][2]/255.0-curr_row[i][2]/255.0),2))/2.0
        sum_dist += math.sqrt(math.pow((prev_row[i-1][0]/255.0-curr_row[i][0]/255.0),2)+math.pow((prev_row[i-1][1]/255.0-curr_row[i][1]/255.0),2)+math.pow((prev_row[i-1][2]/255.0-curr_row[i][2]/255.0),2))/2.0

    return sum_dist

def rotate_until_most_contig(prev_row,curr_row):
    d=deque(curr_row)
    min_diff_score = 999999999999
    best_rot = 0
    for i in range(0,len(curr_row)):
        d.rotate(1)
        my_diff_score = diff_score(prev_row,curr_row)
        if my_diff_score < min_diff_score:
            min_diff_score= my_diff_score
            best_rot = i+1
    d = deque(curr_row)
    d.rotate(best_rot)
    return list(d)
    #return curr_row

def sort_colors_lists(all_colors_list):
    new_all_colors_list = []
    for rgb_colors_list in all_colors_list:
        hue_colors_list = [colorsys.rgb_to_hsv(c[0],c[1],c[2])[0] for c in rgb_colors_list]
        rgb_colors_list = [x for _,x in sorted(zip(hue_colors_list,rgb_colors_list))]
        if len(new_all_colors_list) == 0:
            new_all_colors_list.append(rgb_colors_list)
        else:
            #rotate until rgb colors list matches the prev as best as possible
            new_all_colors_list.append( rotate_until_most_contig(new_all_colors_list[len(new_all_colors_list)-1],rgb_colors_list))
    return new_all_colors_list

def convert_df_into_list_for_react():
    df =  pd.read_csv("data/url_title_and_file_data.csv")
    years_list = df[["file_name","year"]].values.tolist()
    years_list.sort(key=lambda x: x[1])
    palettes = pickle.load(open("data/color_palettes.p","rb"))

    used_years_list = []
    all_colors_list = []
    years_start_and_end = {}
    i = 0
    for el in years_list:
        fname_num = int(el[0].split(".")[0].split("/")[-1])
        year = el[1]
        if fname_num in palettes:
            used_years_list.append(year)
            all_colors_list.append(palettes[fname_num])
            if year not in years_start_and_end:
                years_start_and_end[year] = (i,i+1)
            else:
                years_start_and_end[year] = (years_start_and_end[year][0],i+1)
            i+=1

    # sort rows within the same year
    # sort by average hue of the row
    for year in years_start_and_end.keys():
        avg_hues_list = []
        for  row in  all_colors_list[years_start_and_end[year][0]:years_start_and_end[year][1]]:
            #ct = ColorThief(row)
            #dom_col = ct.get_color(quality=100)
            #avg_hues_list.append(colorsys.rgb_to_hsv(dom_col[0],dom_col[1],dom_col[2])[0])
            avg_hues_list.append(np.mean([colorsys.rgb_to_hsv(c[0],c[1],c[2])[0] for c in row]))
        all_colors_list[years_start_and_end[year][0]:years_start_and_end[year][1]] =  [x for _,x in sorted(zip(avg_hues_list,all_colors_list[years_start_and_end[year][0]:years_start_and_end[year][1]]))]

    # sort within row
    all_colors_list = sort_colors_lists(all_colors_list)

    # MAKE STRING
    my_str = "colors:["
    for i in range(0,len(all_colors_list)):
        curr_colors = all_colors_list[i]
        curr_year = used_years_list[i]
        my_str += "["
        for c in curr_colors:
            my_str+="'#%02x%02x%02x'," %(c[0],c[1],c[2])
        my_str +="%d"%curr_year
        my_str += "],\n"
    my_str = my_str[:-2] + "],\n"


    # make yearly colors
    yearly_colors = []
    yearly_years = []
    for year in range(1800,2020):
        if year not in years_start_and_end:
            continue
        colors_range = all_colors_list[years_start_and_end[year][0]:years_start_and_end[year][1]]
        flat_colors_range = [item for sublist in colors_range for item in sublist]
        ct = ColorThief(flat_colors_range)
        yearly_colors.append(ct.get_palette(color_count=10))
        yearly_years.append(year)

    yearly_colors_hold = yearly_colors
    yearly_colors = []
    for row in yearly_colors_hold:
        hue_colors_list = [colorsys.rgb_to_hsv(c[0],c[1],c[2])[0] for c in row]
        yearly_colors.append([x for _,x in sorted(zip(hue_colors_list,row))])

    my_str += "yearly_colors:["
    for i in range(0,len(yearly_colors)):
        curr_colors = yearly_colors[i]
        curr_year = yearly_years[i]
        my_str += "["
        for i in range(0,9):
            if i < len(curr_colors):
                c = curr_colors[i]
                my_str+="'#%02x%02x%02x'," %(c[0],c[1],c[2])
            else:
                my_str+="'#ffffff',"
        my_str +="%d"%curr_year
        my_str += "],\n"
    my_str = my_str[:-2] + "],"


    text_file = open("data/react_colors_list_for_colors_slides.txt", "w")
    text_file.write(my_str)
    text_file.close()


convert_df_into_list_for_react()
#make_df()
