# get k groups of images
# https://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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
    masks, ids = res[1], res[2]

    people_indices = [i for i in range(0,masks.shape[0]) if ids[i] == 0]
    if len(people_indices) == 0:
        return []

    im = cv2.imread("%s/data/images/smaller_images/%d.jpg"%(DATA_PATH,fnum))
    if (im.shape[0] != masks.shape[1] or im.shape[1] != masks.shape[2]):
        print ("Dimensional problem on %d, image:%d, %d vs masks: %d, %d"%(fnum, im.shape[0],im.shape[1],masks.shape[1],masks.shape[2]))
        return []

    sum_mask = masks[people_indices[0]]
    for ind in people_indices[1:]:
        sum_mask += masks[ind]
    my_pixels = im[sum_mask!=0]
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

def get_all_colors_and_fnum_to_pixels_dict():
    fnums_list = pickle.load(open("%s/data/basics/fnums_list.p"%DATA_PATH,"rb"))
    monochrome_list = set(pickle.load(open("%s/data/monochrome_list_%s.p"%(DATA_PATH,"hsv"),"rb")))
    print ("Loading people pixels from images")
    all_colors = []
    fnum_to_pixels_dict = {}
    for fnum in fnums_list:
        if fnum in monochrome_list:
            continue
        all_pixels_curr = get_pixels_in_file(fnum)
        all_colors.extend(all_pixels_curr)
        if len(all_pixels_curr) != 0:
            fnum_to_pixels_dict[fnum] = all_pixels_curr
    return all_colors, fnum_to_pixels_dict

def make_clusters(num_quantized_colors = 5, num_clusters = 7, all_colors = None, fnum_to_pixels_dict = None):
    fnums_list = pickle.load(open("%s/data/basics/fnums_list.p"%DATA_PATH,"rb"))
    if all_colors == None or fnum_to_pixels_dict == None:
        all_colors, fnum_to_pixels_dict = get_all_colors_and_fnum_to_pixels_dict()
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
    quantized_images_breakdown = QuantizedImageBreakdown(centroids,fnum_to_counts_of_each_color_in_image_dict)
    pickle.dump(quantized_images_breakdown,open("%s/data/quantized_images_breakdown_Q%d.p"%(DATA_PATH,num_quantized_colors),"wb"))

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
    pickle.dump(fnum_to_cluster_dict,open("%s/data/fnum_to_cluster_number_dict_Q%d_K%d.p"%(DATA_PATH,num_quantized_colors,num_clusters),"wb"))
    pickle.dump(cluster_to_fnums_dict,open("%s/data/cluster_number_to_fnum_dict_Q%d_K%d.p"%(DATA_PATH,num_quantized_colors,num_clusters),"wb"))

    make_dict_of_cluster_and_year(Q=num_quantized_colors,K=num_clusters)

def make_dict_of_cluster_and_year(Q=5,K=7):
    fnum_to_cluster_dict = pickle.load(open("%s/data/fnum_to_cluster_number_dict_Q%d_K%d.p"%(DATA_PATH,Q,K),"rb"))
    cluster_to_fnums_dict = pickle.load(open("%s/data/cluster_number_to_fnum_dict_Q%d_K%d.p"%(DATA_PATH,Q,K),"rb"))
    fnums_to_year_dict = pickle.load(open("%s/data/basics/fnum_to_year_dict_Q%d_K%d.p"%(DATA_PATH,Q,K),"rb"))

    # Find all years associated with each color cluster
    color_cluster_to_years_dict = {}
    for cc in cluster_to_fnums_dict.keys():
        fnum_list = cluster_to_fnums_dict[cc]
        color_cluster_to_years_dict[cc] = [fnums_to_year_dict[fn] for fn in fnum_list]
    pickle.dump(color_cluster_to_years_dict,open("%s/data/color_cluster_number_to_years_dict_Q%d_K%d.p"%(DATA_PATH,Q,K),"wb"))

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
    pickle.dump(years_to_most_common_cluster_dict,open("%s/data/years_to_most_common_color_cluster_dict_Q%d_K%d.p"%(DATA_PATH,Q,K),"wb"))

def make_react_codes(Q = 5, K=7):
    print ("Starting making React codes")
    fnum_to_url_dict = pickle.load(open("%s/data/basics/fnum_to_flickr_url_dict.p"%DATA_PATH,"rb"))
    cluster_to_fnums_dict = pickle.load(open("%s/data/cluster_number_to_fnum_dict_Q%d_K%d.p"%(DATA_PATH,Q,K),"rb"))
    for i in range(0,num_clusters):
        random.shuffle(cluster_to_fnums_dict[i])
    min_len = reduce((lambda x, y: min(x,y)), [len(fnums) for fnums in cluster_to_fnums_dict.values()])

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
    quantized_images_breakdown = pickle.load(open("%s/data/quantized_images_breakdown_Q%d.p"%(DATA_PATH,Q),"rb"))
    fnums_to_hex_colors_proportions_dict = quantized_images_breakdown.get_fnums_to_hex_colors_proportions_dict()
    my_str += "quantized_colors: [\n"
    for i in range(0, min_len):
        my_str += "["
        for k in range(0,num_clusters):
            curr_fnum = cluster_to_fnums_dict[k][i]
            if curr_fnum in fnums_to_hex_colors_proportions_dict:
                my_str += "%s,"%fnums_to_hex_colors_proportions_dict[curr_fnum]
            else:
                my_str += "[],"
        my_str = my_str[:-1]+"],\n"
    my_str = my_str[:-2]+"\n],\n"

    text_file = open("%s/data/react-codes/react_color_clustering_page_codes_Q%d_K%d.txt"%(DATA_PATH,Q,K), "w")
    text_file.write(my_str)
    text_file.close()
    print ("Done with React codes")

#num_clusters = 7
#num_quantized_colors = 5
all_colors, fnum_to_pixels_dict = get_all_colors_and_fnum_to_pixels_dict()
for num_clusters in [5,7,10]:
    for num_quantized_colors in [5,7,10,20]:
        print ("working on Q=%d, K = %d"%num_quantized_colors,num_clusters)
        make_clusters(num_quantized_colors =num_quantized_colors,num_clusters=num_clusters,all_colors=all_colors, fnum_to_pixels_dict=fnum_to_pixels_dict)
        make_react_codes(Q =num_quantized_colors,K=num_clusters)
