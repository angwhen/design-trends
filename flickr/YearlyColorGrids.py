from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os, re, pickle, math, cv2
from ColorThiefModified import ColorThief
import colorsys
from collections import deque
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from collections import Counter
from functools import reduce
import HSVHelpers,  GetPixelsHelpers, QuantizationHelper

try:
    DATA_PATH  = open("data_location.txt", "r").read().strip()
except:
    DATA_PATH = "."

def rgb_list_to_hex_list(rgb_list):
    return ["#%02x%02x%02x"%(int(c[0]),int(c[1]),int(c[2])) for c in rgb_list]

def make_yearly_dom_color_palettes(num_colors=10,color_rep="hsv",remove_monochrome=True, remove_heads = False, remove_skin=False):
    info_str = GetPixelsHelpers.get_pixels_dict_info_string(color_rep=color_rep, remove_monochrome=remove_monochrome, remove_heads=remove_heads,remove_skin=remove_skin)
    year_to_pixels_dict =  GetPixelsHelpers.get_year_to_pixels_dict(color_rep=color_rep, remove_monochrome=remove_monochrome, remove_heads=remove_heads,remove_skin=remove_skin)
    year_to_color_palettes_dict = {}
    for year in year_to_pixels_dict.keys():
        year_to_color_palettes_dict[year] = ColorThief(year_to_pixels_dict[year]).get_palette(color_count=num_colors)
    pickle.dump(year_to_color_palettes_dict,open("%s/data/year_to_%d_color_palettes_dict%s.p"%(DATA_PATH,num_colors,info_str),"wb"))

def make_yearly_quantization_based_color_palettes(Q=20,color_rep="hsv",remove_monochrome=True, remove_heads = False, remove_skin=False):
    #all_colors, year_to_pixels_dict = get_all_colors_and_year_to_pixels_dict()
    info_str = GetPixelsHelpers.get_pixels_dict_info_string(color_rep=color_rep, remove_monochrome=remove_monochrome, remove_heads=remove_heads,remove_skin=remove_skin)
    _, all_colors = GetPixelsHelpers.get_fnum_to_pixels_dict_and_all_colors(color_rep=color_rep,remove_monochrome=remove_monochrome, remove_heads=remove_heads,remove_skin=remove_skin)
    year_to_pixels_dict =  GetPixelsHelpers.get_year_to_pixels_dict(color_rep=color_rep, remove_monochrome=remove_monochrome, remove_heads=remove_heads,remove_skin=remove_skin)

    all_colors_array_sample = np.array(all_colors)
    if color_rep == "hsv":
        all_colors_array_sample = np.apply_along_axis(HSVHelpers.project_hsv_to_hsv_cone, 1, all_colors_array_sample)

    quantized_years_breakdown = QuantizationHelper.quantize(Q,color_rep,all_colors_array_sample,year_to_pixels_dict)
    pickle.dump(quantized_years_breakdown,open("%s/data/quantized_years_breakdown_Q%d%s.p"%(DATA_PATH,Q,info_str),"wb"))

def sort_rgb_colors_lists(rgb_colors_list):
    hue_colors_list = [colorsys.rgb_to_hsv(c[0]/255,c[1]/255,c[2]/255)[0] for c in rgb_colors_list]
    return [x for _,x in sorted(zip(hue_colors_list,rgb_colors_list))]

def get_hue_color_from_hex(hex_color):
    hex_color = hex_color[1:]
    rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return colorsys.rgb_to_hsv(rgb_color[0]/255,rgb_color[1]/255,rgb_color[2]/255)[0]

def get_ordered_string_from_hex_colors_proportions_dict(hex_colors_proportions_dict):
    my_tuples = []
    for hex_color in hex_colors_proportions_dict.keys():
        hue = get_hue_color_from_hex(hex_color)
        my_tuples.append([hex_color,hue,hex_colors_proportions_dict[hex_color]])
    my_tuples = sorted(my_tuples,key=lambda x: x[1])
    my_str = "{"
    for tup in my_tuples:
        my_str += "\'%s\':%s, "%(tup[0],tup[2])
    my_str = my_str[:-2]+"}"
    return my_str

def make_yearly_colors_list_for_react(num_colors,num_quantized_colors, info_str):
    year_to_color_palettes_dict=pickle.load(open("%s/data/year_to_%d_color_palettes_dict%s.p"%(DATA_PATH,num_colors,info_str),"rb"))
    my_str = "yearly_colors:["
    for curr_year in sorted(list(year_to_color_palettes_dict.keys())):
        curr_colors =  sort_rgb_colors_lists(year_to_color_palettes_dict[curr_year])
        my_str += "{year:%d, colors: %s},\n"%(curr_year,rgb_list_to_hex_list(curr_colors))
    my_str = my_str[:-2] + "],\n"

    quantized_years_breakdown = pickle.load(open("%s/data/quantized_years_breakdown_Q%d%s.p"%(DATA_PATH,num_quantized_colors,info_str),"rb"))
    year_to_hex_colors_proportions_dict = quantized_years_breakdown.get_year_to_hex_colors_proportions_dict()
    my_str += "yearly_quantized_colors: [\n"
    for curr_year in range(1800,2020):
        if curr_year not in year_to_hex_colors_proportions_dict:
            continue
        my_str += "{year:%d, colors: %s},\n"%(curr_year, get_ordered_string_from_hex_colors_proportions_dict(year_to_hex_colors_proportions_dict[curr_year]))
    my_str = my_str[:-2] + "],\n"

    text_file = open("%s/data/react-codes/react_yearly_colors_grids%s.txt"%(DATA_PATH,info_str), "w")
    text_file.write(my_str)
    text_file.close()
    print ("Done making yearly colors grids react codes")

def yearly_grids(num_dom_colors=5, Q= 10, color_rep="hsv",remove_monochrome=True, remove_heads = False, remove_skin=False):
    make_yearly_quantization_based_color_palettes(Q=Q,color_rep=color_rep, remove_monochrome=remove_monochrome, remove_heads=remove_heads,remove_skin=remove_skin)
    make_yearly_dom_color_palettes(num_colors=num_dom_colors,color_rep=color_rep, remove_monochrome=remove_monochrome, remove_heads=remove_heads,remove_skin=remove_skin)
    info_str = GetPixelsHelpers.get_pixels_dict_info_string(color_rep=color_rep, remove_monochrome=remove_monochrome, remove_heads=remove_heads,remove_skin=remove_skin)
    make_yearly_colors_list_for_react(num_colors=num_dom_colors,num_quantized_colors=Q,info_str=info_str)
"""
make_yearly_dom_color_palettes(num_colors=10)
make_yearly_quantization_based_color_palettes()
make_yearly_colors_list_for_react(num_colors=10)


def get_all_colors_and_year_to_pixels_dict():
    fnums_list = pickle.load(open("%s/data/basics/fnums_list.p"%DATA_PATH,"rb"))
    year_to_fnums_dict=pickle.load(open("%s/data/basics/year_to_fnums_dict.p"%DATA_PATH,"rb"))
    all_colors = []
    year_to_pixels_dict = {}
    for year in year_to_fnums_dict.keys():
        all_pixels_curr = get_pixels_in_fnums(year_to_fnums_dict[year],1)
        all_colors.extend(all_pixels_curr)
        if len(all_pixels_curr) != 0:
            year_to_pixels_dict[year] = all_pixels_curr
    return all_colors, year_to_pixels_dict"""
