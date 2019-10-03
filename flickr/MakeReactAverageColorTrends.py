from matplotlib import pyplot as plt
import pandas as pd
import pickle
import os
import re
import colorsys
from collections import deque
import math
import numpy as np
import cv2

# list of average H, S, V for each year to make react plot

def get_pixels_in_file(fname_num):
    try:
        res = pickle.load(open("data/images/mask_rcnn_results/res_%d.p"%fname_num,"rb"))
    except:
        return []
    masks = res[1]
    ids = res[2]

    people_indices = []
    for i in range(0,masks.shape[0]): #the masks we have for people
        if ids[i] == 0:
            people_indices.append(i)

    if len(people_indices) == 0:
        return []
    im = cv2.imread("data/images/smaller_images/%d.jpg"%fname_num)
    if (im.shape[0] != masks.shape[1] or im.shape[1] != masks.shape[2]):
        print ("some dimensional problem")
        return []

    print (fname_num)
    my_pixels = []
    inner_count = 0
    for ind in people_indices:
        curr_mask =  masks[ind]
        for row in range(0,curr_mask.shape[0]):
            for col in range(0,curr_mask.shape[1]):
                if inner_count % 10 == 0: # dont save all pixels to save space
                    my_pixels.append(im[row][col])
                inner_count +=1
    return my_pixels

def make_pickle_sums_and_counts():
    df =  pd.read_csv("data/url_title_and_file_data.csv")
    years_list = df[["file_name","year"]].values.tolist()
    years_list.sort(key=lambda x: x[1])
    rgb_colors_sums = {}
    #pickle.load(open("data/years_to_rgb_colors_sums.p","rb")) # bigger people weighted more heavily
    rgb_colors_counts = {}
    #pickle.load(open("data/years_to_rgb_colors_counts.p","rb"))
    done_fname_nums = []
    #pickle.load(open("data/years_done_fname_nums_for_colors_avgs.p","rb"))
    #restart since not keeping track of the fnames done anyways
    count = 0
    for el in years_list:
        fname_num = int(el[0].split(".")[0].split("/")[-1])
        if fname_num in done_fname_nums:
            continue
        year = el[1]
        pixels_list = get_pixels_in_file(fname_num)
        if len(pixels_list) == 0:
            continue

        pixels_list_colors_sums = list(sum(np.array(np.array(pixels_list).tolist())))
        print (pixels_list_colors_sums)
        if year in rgb_colors_sums:
            rgb_colors_sums[year] = [sum(x) for x in zip(rgb_colors_sums[year],pixels_list_colors_sums)]
            rgb_colors_counts[year] +=len(pixels_list)
        else:
            rgb_colors_sums[year] = pixels_list_colors_sums
            rgb_colors_counts[year] =len(pixels_list)

        if count % 5 == 0: #save frequently to avoid having to rerun too often
            pickle.dump(rgb_colors_sums,open("data/years_to_rgb_colors_sums.p","wb"))
            pickle.dump(rgb_colors_counts,open("data/years_to_rgb_colors_counts.p","wb"))
            pickle.dump(done_fname_nums,open("data/years_done_fname_nums_for_colors_avgs.p","wb"))
            print ("current part saved")
        count +=1
        done_fname_nums.append(fname_num)
    pickle.dump(rgb_colors_sums,open("data/years_to_rgb_colors_sums.p","wb"))
    pickle.dump(rgb_colors_counts,open("data/years_to_rgb_colors_counts.p","wb"))
    pickle.dump(done_fname_nums,open("data/years_done_fname_nums_for_colors_avgs.p","wb"))

def make_current(rgb_colors_avg, my_type = "red"):
    my_str = ""
    if my_type == "red":
        my_type_num = 0
        my_title = "Red Over Time"
    elif my_type == "green":
        my_type_num = 1
        my_title = "Green Over Time"
    elif my_type == "blue":
        my_type_num = 2
        my_title = "Blue Over Time"
    elif my_type == "hsv_hue":
        my_type_num = 3
        my_title = "HSV Hue Over Time"
    elif my_type == "hsv_sat":
        my_type_num = 4
        my_title = "HSV Saturation Over Time"
    elif my_type == "hsv_val":
        my_type_num = 5
        my_title = "HSV Value Over Time"
    my_str += "if (this.state.currentChartNum == %d){\n"%my_type_num
    my_str +=  """
         code.push(
         <div>
         <center><h1>%s</h1></center>\n"""%(my_title)
    my_str += """
         <BarChart
          axisLabels={{x: 'My x Axis', y: 'My y Axis'}}
          axes
          width = {window.innerWidth*0.95}
            height = {700}
            xType={'time'}
            tickTimeDisplayFormat={'%Y'}
            datePattern="%Y"
            colorBars
          data={[\n"""
    for year in range(1852,2019):
        if year not in rgb_colors_avg:
            continue
        if my_type == "red":
            val = int(rgb_colors_avg[year][0])
            color = "#%02x%02x%02x" %(val,100,100)
        elif my_type == "green":
            val = int(rgb_colors_avg[year][1])
            color = "#%02x%02x%02x" %(100,val,100)
        elif my_type == "blue":
            val = int(rgb_colors_avg[year][2])
            color = "#%02x%02x%02x" %(100,100,val)
        elif "hsv" in my_type:
            hsv_cols = colorsys.rgb_to_hsv(rgb_colors_avg[year][0]/255,rgb_colors_avg[year][1]/255,rgb_colors_avg[year][2]/255)
            if my_type == "hsv_hue":
                val = int(hsv_cols[0]*255)
                col_rgb = [int(el*255) for el in colorsys.hsv_to_rgb(val/255,1,1)]
                color = "#%02x%02x%02x" %(col_rgb[0],col_rgb[1],col_rgb[2])
            elif my_type == "hsv_sat":
                val = int(hsv_cols[1]*255)
                col_rgb = [int(el*255) for el in colorsys.hsv_to_rgb(0,val/255,1)]
                color = "#%02x%02x%02x" %(col_rgb[0],col_rgb[1],col_rgb[2])
            elif my_type == "hsv_val":
                val = int(hsv_cols[2]*255)
                col_rgb = [int(el*255) for el in colorsys.hsv_to_rgb(0,1,val/255)]
                color = "#%02x%02x%02x" %(col_rgb[0],col_rgb[1],col_rgb[2])
        my_str += "\t\t{ x: '%s', y: %s, color: '%s' },\n"%(year,val,color)
    my_str += """]}/> </div>)
                }\n"""
    return my_str


#make_pickle_sums_and_counts()

#open it and make charts
rgb_colors_sums = pickle.load(open("data/years_to_rgb_colors_sums.p","rb")) # bigger people weighted more heavily
rgb_colors_counts = pickle.load(open("data/years_to_rgb_colors_counts.p","rb"))
rgb_colors_avg = {}
for k in rgb_colors_sums:
    rgb_colors_avg[k] = [c/rgb_colors_counts[k] for c in rgb_colors_sums[k]]

my_str = ""
my_str +=  make_current(rgb_colors_avg,my_type="red")
my_str +=  make_current(rgb_colors_avg,my_type="green")
my_str +=  make_current(rgb_colors_avg,my_type="blue")
my_str +=  make_current(rgb_colors_avg,my_type="hsv_hue")
my_str +=  make_current(rgb_colors_avg,my_type="hsv_sat")
my_str +=  make_current(rgb_colors_avg,my_type="hsv_val")
text_file = open("data/react_colors_charts_for_trends.txt", "w")
text_file.write(my_str)
text_file.close()
