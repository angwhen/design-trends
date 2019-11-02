# get k groups of images
# https://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
import pandas as pd
import pickle
from time import time
import cv2
import random
from collections import Counter
from functools import reduce

DATA_PATH = "."
try:
    f=open("data_location.txt", "r")
    DATA_PATH  = f.read().strip()
except:
    print ("data is right here")

def rgb_list_to_hex_list(rgb_list):
    return ["#%02x%02x%02x"%(int(c[0]),int(c[1]),int(c[2])) for c in rgb_list]

def get_pixels_in_file(fnum,every_few = 10):
    try:
        res = pickle.load(open("%s/data/images/mask_rcnn_results/res_%d.p"%(DATA_PATH,fnum),"rb"))
    except:
        return []
    masks = res[1]
    ids = res[2]

    people_indices = []
    for i in range(0,masks.shape[0]): #the masks we have for people
        if ids[i] == 0: # 0 means person
            people_indices.append(i)
    if len(people_indices) == 0:
        return []

    im = cv2.imread("%s/data/images/smaller_images/%d.jpg"%(DATA_PATH,fnum))
    if (im.shape[0] != masks.shape[1] or im.shape[1] != masks.shape[2]):
        print ("some dimensional problem")
        return []

    my_pixels = []
    not_greys = 0
    for ind in people_indices:
        curr_mask =  masks[ind]
        for row in range(0,curr_mask.shape[0]):
            for col in range(0,curr_mask.shape[1]):
                my_pixels.append(im[row][col])
                # if it is grayscale do not return
                if abs(im[row][col][0] - im[row][col][1]) > 5 or abs(im[row][col][0] - im[row][col][2]) > 5 or  abs(im[row][col][1] - im[row][col][2]) > 5:
                    not_greys +=1
    # if it is grayscale do not return
    if not_greys < 100: #allow for a few non grays in case photo bad or something
        return []
    return shuffle(my_pixels, random_state=0)[:max(36000,int(len(my_pixels)/every_few))] #dont let any image return too many pixels

class QuantizedImageBreakdown():
    def __init__(self,colors_definitions,fnum_to_counts_of_each_color_in_image_dict):
        self.colors_definitions = rgb_list_to_hex_list(colors_definitions)
        self.fnum_to_counts_of_each_color_in_image_dict = fnum_to_counts_of_each_color_in_image_dict
    def get_fnums_to_hex_colors_proportions_dict(self):
        fnums_to_hex_colors_proportions_dict = {}
        for fnum in self.fnum_to_counts_of_each_color_in_image_dict.keys():
            total_pixels = sum(self.fnum_to_counts_of_each_color_in_image_dict[fnum])
            hex_colors_proportions_dict = {}
            for i,color in enumerate(self.colors_definitions):
                hex_colors_proportions_dict[color] = self.fnum_to_counts_of_each_color_in_image_dict[fnum][i]/total_pixels
            fnums_to_hex_colors_proportions_dict[fnum] = hex_colors_proportions_dict
        return fnums_to_hex_colors_proportions_dict

def make_clusters(num_quantized_colors = 5, num_clusters = 7):
    fnums_list = pickle.load(open("%s/data/basics/fnums_list.p"%DATA_PATH,"rb"))

    print ("Loading people pixels from images")
    all_colors = []
    fnum_to_pixels_dict = {}
    for fnum in fnums_list[:1]:
        all_pixels_curr = get_pixels_in_file(fnum)
        all_colors.extend(all_pixels_curr)
        if len(all_pixels_curr) != 0:
            fnum_to_pixels_dict[fnum] = all_pixels_curr
    all_colors_array_sample = shuffle(np.array(all_colors), random_state=0)[:500000]

    print ("Quantization")
    kmeans = KMeans(n_clusters=num_quantized_colors, max_iter=100,random_state=0).fit(all_colors_array_sample)
    # Quantize each image
    fnum_to_counts_of_each_color_in_image_dict= {}
    for fnum in fnums_list:
        if fnum in fnum_to_pixels_dict:
            list_of_colors =  kmeans.predict(fnum_to_pixels_dict[fnum])
            counts_of_each_color_in_image = [0]*num_quantized_colors
            for color_ind in list_of_colors:
                counts_of_each_color_in_image[color_ind] +=1
            fnum_to_counts_of_each_color_in_image_dict[fnum] = counts_of_each_color_in_image

    centroids = kmeans.cluster_centers_
    print (centroids)
    quantized_images_breakdown = QuantizedImageBreakdown(centroids,fnum_to_counts_of_each_color_in_image_dict)
    print (quantized_images_breakdown.get_fnums_to_hex_colors_proportions_dict())
    pickle.dump(quantized_images_breakdown,open("%s/data/quantized_images_breakdown.p"%DATA_PATH,"wb"))

    # Clustering images
    print ("Clustering on those counts per color labels")
    fnums_in_order_list = list(fnum_to_counts_of_each_color_in_image_dict.keys())
    counts_of_each_color_in_image_list = [fnum_to_counts_of_each_color_in_image_dict[fnum] for fnum in fnums_in_order_list]
    kmeans = KMeans(n_clusters=num_clusters,random_state=0).fit(counts_of_each_color_in_image_list)
    clusters = kmeans.predict(counts_of_each_color_in_image_list)
    fnum_to_cluster_dict, cluster_to_fnums_dict = {}, {}
    for i in range(0,len(fnums_in_order_list)):
        fnum_to_cluster_dict[fnums_in_order_list[i]] = clusters[i]
        if clusters[i] not in cluster_to_fnums_dict:
            cluster_to_fnums_dict[clusters[i]] = []
        cluster_to_fnums_dict[clusters[i]].append(fnums_in_order_list[i])
    pickle.dump(fnum_to_cluster_dict,open("%s/data/fnum_to_cluster_number_dict.p"%DATA_PATH,"wb"))
    pickle.dump(cluster_to_fnums_dict,open("%s/data/cluster_number_to_fnum_dict.p"%DATA_PATH,"wb"))

    make_dict_of_cluster_and_year()

def make_dict_of_cluster_and_year():
    fnum_to_cluster_dict = pickle.load(open("%s/data/fnum_to_cluster_number_dict.p"%DATA_PATH,"rb"))
    cluster_to_fnums_dict = pickle.load(open("%s/data/cluster_number_to_fnum_dict.p"%DATA_PATH,"rb"))
    fnums_to_year_dict = pickle.load(open("%s/data/basics/fnum_to_year_dict.p"%DATA_PATH,"rb"))

    # Find all years associated with each color cluster
    color_cluster_to_years_dict = {}
    for cc in cluster_to_fnames_dict.keys():
        fnum_list = cluster_to_fnames_dict[cc]
        color_cluster_to_years_dict[cc] = [fnums_to_year_dict[fn] for fn in fnum_list]
    pickle.dump(color_cluster_to_years_dict,open("%s/data/color_cluster_number_to_years_dict.p"%DATA_PATH,"wb"))

    # Find most common color cluster for a given year
    years_to_most_common_cluster_dict = {}
    for fnum in fnums_to_year_dict.keys():
        year = fnums_to_year_dict[fnum]
        if fnum not in fnum_to_cluster_dict:
            continue
        curr_cluster = fnum_to_cluster_dict[fnum]
        if year not in years_to_most_common_cluster_dict:
            years_to_most_common_cluster_dict[year] = []
        years_to_most_common_cluster_dict[year].append(curr_cluster)
    for year in years_to_most_common_cluster_dict:
        years_to_most_common_cluster_dict[year] = Counter(years_to_most_common_cluster_dict[year]).most_common(1)[0][0]
    pickle.dump(years_to_most_common_cluster_dict,open("%s/data/years_to_most_common_color_cluster_dict.p"%DATA_PATH,"wb"))

def make_react_codes(num_clusters=7):
    print ("Starting making React codes")
    fnum_to_url_dict = pickle.load(open("%s/data/basics/fnum_to_flickr_url_dict.p"%DATA_PATH,"rb"))
    cluster_to_fnums_dict = pickle.load(open("%s/data/cluster_number_to_file_num_dict.p"%DATA_PATH,"rb"))
    for i in range(0,num_clusters):
        random.shuffle(cluster_to_fnums_dict[i])
    min_len = reduce((lambda x, y: max(x,y)), [len(fnums) for fnums in cluster_to_fnums_dict.values()])

    # IMAGES
    my_str = "images: [\n"
    for i in range(0, min_len):
        my_str += "["
        for k in range(0,num_clusters):
            curr_fnum = cluster_to_fnums_dict[k][i]
            my_str += "'%s',"%fnum_to_url_dict[curr_fnum]
        my_str = my_str[:-1]+"],\n"
    my_str = my_str[:-2]+"\n],\n"

    # DOM COLORS
    fnum_to_palettes_dict = pickle.load(open("%s/data/5_color_fnum_to_palettes_dict.p"%DATA_PATH,"rb"))
    my_str += "dom_colors: [\n"
    for i in range(0, min_len):
        my_str += "["
        for k in range(0,num_clusters):
            curr_fnum = cluster_to_fnums_dict[k][i]
            if curr_fnum in fnum_to_palettes_dict:
                my_str += "%s,"%rgb_list_to_hex_list(fnum_to_palettes_dict[curr_fnum])
            else:
                my_str += "[],"
        my_str = my_str[:-1]+"],\n"
    my_str = my_str[:-2]+"\n],\n"

    # QUANTIZED COLORS: each one gets a dict from hex color to proportion

    text_file = open("%s/data/react-codes/react_color_clustering_page_codes.txt"%DATA_PATH, "w")
    text_file.write(my_str)
    text_file.close()
    print ("Done with React codes")

num_clusters = 7
make_clusters(num_clusters=num_clusters)
#make_react_codes(num_clusters=num_clusters)
