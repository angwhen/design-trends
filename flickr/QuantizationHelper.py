import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from time import time
from collections import Counter
from functools import reduce
import pickle, math, random, cv2, colorsys
import HSVHelpers


def rgb_list_to_hex_list(rgb_list):
    return ["#%02x%02x%02x"%(int(c[0]),int(c[1]),int(c[2])) for c in rgb_list]

class QuantizedImageBreakdown():
    def __init__(self,colors_definitions,im_to_counts_of_each_color_in_image_dict):
        self.colors_definitions = rgb_list_to_hex_list(colors_definitions)
        self.im_to_counts_of_each_color_in_image_dict = im_to_counts_of_each_color_in_image_dict
    def get_ims_to_hex_colors_proportions_dict(self):
        ims_to_hex_colors_proportions_dict = {}
        for im in self.im_to_counts_of_each_color_in_image_dict.keys():
            total_pixels = sum(self.im_to_counts_of_each_color_in_image_dict[im])
            hex_colors_proportions_dict = {}
            for i,color in enumerate(self.colors_definitions):
                hex_colors_proportions_dict[color] = self.im_to_counts_of_each_color_in_image_dict[im][i]/total_pixels
            ims_to_hex_colors_proportions_dict[im] = hex_colors_proportions_dict
        return ims_to_hex_colors_proportions_dict

def quantize(Q,color_rep,all_colors_array_sample,fnum_to_pixels_dict):
    print ("Quantization")
    kmeans = KMeans(n_clusters=Q, max_iter=100,random_state=0).fit(all_colors_array_sample)
    # Quantize each image
    fnum_to_counts_of_each_color_in_image_dict= {}
    for fnum in fnum_to_pixels_dict:
        if color_rep == "hsv":
            list_of_colors =  kmeans.predict(np.apply_along_axis(HSVHelpers.project_hsv_to_hsv_cone, 1, fnum_to_pixels_dict[fnum]))
        else:
            list_of_colors =  kmeans.predict(fnum_to_pixels_dict[fnum])
        color_ind_to_count = Counter(list_of_colors)
        counts_of_each_color_in_image = [0]*Q
        for color_ind in color_ind_to_count.keys():
            counts_of_each_color_in_image[color_ind] = color_ind_to_count[color_ind]
        fnum_to_counts_of_each_color_in_image_dict[fnum] = counts_of_each_color_in_image

    centroids = kmeans.cluster_centers_
    if color_rep == "hsv": #convert back to rgb for saving purposes
        centroids = [HSVHelpers.hsv_cone_coords_to_rgb(col) for col in centroids]
    quantized_images_breakdown = QuantizedImageBreakdown(centroids,fnum_to_counts_of_each_color_in_image_dict)
    return quantized_images_breakdown
