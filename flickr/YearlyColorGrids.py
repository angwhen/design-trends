from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os, re, pickle, math, cv2
from ColorThiefModified import ColorThief
import colorsys
from collections import deque
from sklearn.utils import shuffle
import HSVHelpers

DATA_PATH = ""
try:
    f=open("data_location.txt", "r")
    DATA_PATH  = f.read().strip()
except:
    print ("data is right here")


def rgb_list_to_hex_list(rgb_list):
    return ["#%02x%02x%02x"%(int(c[0]),int(c[1]),int(c[2])) for c in rgb_list]

class QuantizedImageBreakdown():
    def __init__(self,colors_definitions,year_to_counts_of_each_color_in_image_dict):
        self.colors_definitions = rgb_list_to_hex_list(colors_definitions)
        self.year_to_counts_of_each_color_in_image_dict = year_to_counts_of_each_color_in_image_dict
    def get_year_to_hex_colors_proportions_dict(self):
        year_to_hex_colors_proportions_dict = {}
        for year in self.year_to_counts_of_each_color_in_image_dict.keys():
            total_pixels = sum(self.year_to_counts_of_each_color_in_image_dict[year])
            hex_colors_proportions_dict = {}
            for i,color in enumerate(self.colors_definitions):
                hex_colors_proportions_dict[color] = self.year_to_counts_of_each_color_in_image_dict[year][i]/total_pixels
            year_to_hex_colors_proportions_dict[year] = hex_colors_proportions_dict
        return year_to_hex_colors_proportions_dict

def get_pixels_in_fnums(fnums,sample_amount):
    all_pixels = []
    for fnum in fnums:
        try:
            res = pickle.load(open("%s/data/images/mask_rcnn_results/res_%d.p"%(DATA_PATH,fnum),"rb"))
        except:
            continue
        masks, ids = res[1], res[2]
        people_indices = [i for i in range(0,masks.shape[0]) if ids[i] == 0]
        if len(people_indices) == 0:
            continue

        im = cv2.imread("%s/data/images/smaller_images/%d.jpg"%(DATA_PATH,fnum))
        if (im.shape[0] != masks.shape[1] or im.shape[1] != masks.shape[2]):
            print ("Dimensional problem on %d, image:%d, %d vs masks: %d, %d"%(fnum, im.shape[0],im.shape[1],masks.shape[1],masks.shape[2]))
            continue

        sum_mask = masks[people_indices[0]]
        for ind in people_indices[1:]:
            sum_mask += masks[ind]
        my_pixels = im[sum_mask!=0]
        all_pixels.extend(shuffle(my_pixels, random_state=0)[:int(len(my_pixels)/sample_amount)])

    return shuffle(all_pixels,random_state=0)[:360000]

def make_yearly_color_palettes(num_colors=10,sample_amount=5):
    year_to_fnums_dict=pickle.load(open("%s/data/basics/year_to_fnums_dict.p"%DATA_PATH,"rb"))
    year_to_color_palettes_dict = {}
    for year in year_to_fnums_dict.keys():
        if year in year_to_color_palettes_dict:
            continue
        year_pixels = get_pixels_in_fnums(year_to_fnums_dict[year],sample_amount)
        if len(year_pixels) == 0:
            print ("year %d has no valid pixels to use"%year)
            continue
        ct = ColorThief(year_pixels)
        color_list = ct.get_palette(color_count=num_colors)
        year_to_color_palettes_dict[year] = color_list
    pickle.dump(year_to_color_palettes_dict,open("%s/data/year_to_%d_color_palettes_dict.p"%(DATA_PATH,num_colors),"wb"))

def get_all_colors_and_year_to_pixels_dict():
    fnums_list = pickle.load(open("%s/data/basics/fnums_list.p"%DATA_PATH,"rb"))
    fnum_to_year_dict=pickle.load(open("%s/data/basics/fnum_to_year_dict.p"%DATA_PATH,"rb"))
    all_colors = []
    year_to_pixels_dict = {}
    for fnum in fnums_list:
        all_pixels_curr = get_pixels_in_fnums([fnum])
        all_colors.extend(all_pixels_curr)
        if len(all_pixels_curr) != 0:
            year = fnum_to_year_dict[fnum]
            if year not in year_to_pixels_dict:
                year_to_pixels_dict[year] = []
            year_to_pixels_dict[year].extend(all_pixels_curr)
    return all_colors, year_to_pixels_dict

def make_yearly_quantization_based_color_palettes(num_quantized_colors=20):
    fnums_list = pickle.load(open("%s/data/basics/fnums_list.p"%DATA_PATH,"rb"))
    all_colors, year_to_pixels_dict = get_all_colors_and_year_to_pixels_dict()

    all_colors_array_sample = shuffle(np.array(all_colors), random_state=0)[:500000]
    all_colors_array_sample = np.apply_along_axis(HSVHelpers.project_hsv_to_hsv_cone, 1, all_colors_array_sample)

    print ("Quantization")
    kmeans = KMeans(n_clusters=num_quantized_colors, max_iter=100,random_state=0).fit(all_colors_array_sample)
    # Quantize each image
    year_to_counts_of_each_color_in_image_dict= {}
    for year in range(1800,2020):
        if year in year_to_pixels_dict:
            list_of_colors =  kmeans.predict(np.apply_along_axis(HSVHelpers.project_hsv_to_hsv_cone, 1, fnum_to_pixels_dict[fnum]))
            counts_of_each_color_in_image = [0]*num_quantized_colors
            for color_ind in list_of_colors:
                counts_of_each_color_in_image[color_ind] +=1
            year_to_counts_of_each_color_in_image_dict[year] = counts_of_each_color_in_image

    centroids = kmeans.cluster_centers_
    centroids = [HSVHelpers.hsv_cone_coords_to_hsv(col) for col in centroids]
    centroids = [colorsys.hsv_to_rgb(col[0]/255,col[1]/255,col[2]/255) for col in centroids]
    centroids = [[col[0]*255,col[1]*255,col[2]*255] for col in centroids]
    quantized_years_breakdown = QuantizedImageBreakdown(centroids,year_to_counts_of_each_color_in_image_dict)
    pickle.dump(quantized_years_breakdown,open("%s/data/quantized_years_breakdown_Q%d_hsv_including_monochrome.p"%(DATA_PATH,num_quantized_colors),"wb"))

def sort_colors_lists(rgb_colors_list):
    hue_colors_list = [colorsys.rgb_to_hsv(c[0],c[1],c[2])[0] for c in rgb_colors_list]
    rgb_colors_list = [x for _,x in sorted(zip(hue_colors_list,rgb_colors_list))]
    return rgb_colors_list

def get_hsv_color_from_hex(hex_color):
    hex_color = hex_color[1:]
    rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    hsv_color = colorsys.rgb_to_hsv(rgb_color[0]/255,rgb_color[1]/255,rgb_color[2]/255)[0]

def get_ordered_string_from_hex_colors_proportions_dict(hex_colors_proportions_dict):
    my_tuples = []
    for hex_color in hex_colors_proportions_dict.keys():
        hue = get_hsv_color_from_hex(hex_color)[0]
        my_tuples.append([hex_color,hue,hex_colors_proportions_dict[hex_color]])
    my_tuples = sorted(my_tuples,key=lambda x: x[1])
    my_str = "{"
    for tup in my_tuples:
        my_str += "%s:%s, "%(tup[0],tup[2])
    my_str = my_str[:-2]+"}"
    return my_str

def make_yearly_colors_list_for_react(num_colors=10,num_quantized_colors=20):
    year_to_color_palettes_dict=pickle.load(open("%s/data/year_to_%d_color_palettes_dict.p"%(DATA_PATH,num_colors),"rb"))

    my_str = "yearly_colors:["
    for curr_year in sorted(list(year_to_color_palettes_dict.keys())):
        curr_colors =  sort_colors_lists(year_to_color_palettes_dict[curr_year])
        my_str += "{year:%d, colors: %s},\n"%(curr_year,rgb_list_to_hex_list(curr_colors))
    my_str = my_str[:-2] + "],\n"

    quantized_years_breakdown = pickle.load(open(open("%s/data/quantized_years_breakdown_Q%d_hsv_including_monochrome.p"%(DATA_PATH,num_quantized_colors),"rb"))
    year_to_hex_colors_proportions_dict = quantized_years_breakdown.get_year_to_hex_colors_proportions_dict()
    my_str += "yearly_quantized_colors: [\n"
    for curr_year in range(1800,2020):
        if curr_year not in year_to_hex_colors_proportions_dict:
            continue
        my_str += "{year:%d, colors: %s},\n"%(curr_year, get_ordered_string_from_hex_colors_proportions_dict(year_to_hex_colors_proportions_dict[curr_year]))
    my_str = my_str[:-2] + "],\n"

    text_file = open("%s/data/react-codes/react_yearly_colors_for_colors_slides.txt"%DATA_PATH, "w")
    text_file.write(my_str)
    text_file.close()
    print ("Done making yearly colors react codes")

#make_yearly_color_palettes(num_colors=10)
make_yearly_quantization_based_color_palettes()
make_yearly_colors_list_for_react(num_colors=10)
