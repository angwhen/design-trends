from matplotlib import pyplot as plt
import pandas as pd
import pickle
import os
import re
import colorsys
from collections import deque
import math
import numpy as np

# list of average H, S, V for each year to make react plot

def get_pixels_in_file(fname_num):
    try:
        res = pickle.load(open("data/images/mask_rcnn_results/res_%d.p"%fname_num,"rb"))
    except:
        return []
    orig_img = res[0]
    masks = res[1]
    ids = res[2]

    people_indices = []
    for i in range(0,masks.shape[0]): #the masks we have for people
        if ids[i] == 0:
            people_indices.append(i)

    if len(people_indices) == 0:
        return []

    print (fname_num)
    my_pixels = []
    inner_count = 0
    for ind in people_indices:
        curr_mask =  masks[ind]
        for row in range(0,curr_mask.shape[0]):
            for col in range(0,curr_mask.shape[1]):
                if inner_count % 10 == 0: # dont save all pixels to save space
                    my_pixels.append(orig_img[row][col])
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
    #restart since not keeping track of the fnames done anyways
    count = 0
    for el in years_list:
        fname_num = int(el[0].split(".")[0].split("/")[-1])
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
            print ("current part saved")
        count +=1
    pickle.dump(rgb_colors_sums,open("data/years_to_rgb_colors_sums.p","wb"))
    pickle.dump(rgb_colors_counts,open("data/years_to_rgb_colors_counts.p","wb"))

def make_current(my_str,rgb_colors_avg, type = "red"):
    my_str +=  """
         if (this.state.currentChartNum == 0){
         code.push(<BarChart
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
        val = int(rgb_colors_avg[year][0])
        color = "#%02x%02x%02x" %(val,0,0)
        my_str += "\t\t{ x: '%s', y: %s, color: '%s' },\n"%(year,val,color)
    my_str += """]}/>)
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
my_str +=  make_current(my_str,rgb_colors_avg)

text_file = open("data/react_colors_charts_for_trends", "w")
text_file.write(my_str)
text_file.close()
