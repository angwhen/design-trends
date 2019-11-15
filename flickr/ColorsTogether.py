import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from time import time
from collections import Counter
from functools import reduce
import pickle, math, random, cv2, colorsys
import HSVHelpers, GetPixelsHelpers

try:
    DATA_PATH  = open("data_location.txt", "r").read().strip()
except:
    DATA_PATH = "."

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

def make_color_clusters(Q, K, color_rep="rgb",remove_monochrome=False, remove_predom_faces = False, remove_skin=False):
    info_str = GetPixelsHelpers.get_pixels_dict_info_string(color_rep=color_rep, remove_monochrome=remove_monochrome, remove_predom_faces=remove_predom_faces,remove_skin=remove_skin)
    fnums_list = pickle.load(open("%s/data/basics/fnums_list.p"%DATA_PATH,"rb"))
    fnum_to_pixels_dict,all_colors = GetPixelsHelpers.get_fnum_to_pixels_dict_and_all_colors(color_rep=color_rep,remove_monochrome=remove_monochrome, remove_predom_faces=remove_predom_faces,remove_skin=remove_skin)

    all_colors_array_sample = np.array(all_colors)
    if color_rep == "hsv":
        all_colors_array_sample = np.apply_along_axis(HSVHelpers.project_to_hsv_cone, 1, all_colors_array_sample)

    print ("Quantization")
    kmeans = KMeans(n_clusters=Q, max_iter=100,random_state=0).fit(all_colors_array_sample)
    # Quantize each image
    fnum_to_counts_of_each_color_in_image_dict= {}
    for fnum in fnum_to_pixels_dict:
        if color_rep == "hsv":
            list_of_colors =  kmeans.predict(np.apply_along_axis(HSVHelpers.project_to_hsv_cone, 1, fnum_to_pixels_dict[fnum]))
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
    pickle.dump(quantized_images_breakdown,open("%s/data/quantized_images_breakdown_Q%d%s.p"%(DATA_PATH,Q,info_str),"wb"))

    print ("Clustering on those counts per color labels")
    fnums_in_order_list = list(fnum_to_counts_of_each_color_in_image_dict.keys())
    counts_of_each_color_in_image_list = [fnum_to_counts_of_each_color_in_image_dict[fnum] for fnum in fnums_in_order_list]
    kmeans = KMeans(n_clusters=K,random_state=0).fit(counts_of_each_color_in_image_list)
    clusters = kmeans.predict(counts_of_each_color_in_image_list)
    fnum_to_cluster_dict, cluster_to_fnums_dict,fnum_to_distance_to_cluster_dict = {}, {}, {}
    for i in range(0,len(fnums_in_order_list)):
        fnum_to_cluster_dict[fnums_in_order_list[i]] = clusters[i]
        fnum_to_distance_to_cluster_dict[fnum] = numpy.linalg.norm(kmeans.cluster_centers_[i]-counts_of_each_color_in_image_list[i])
        if clusters[i] not in cluster_to_fnums_dict:
            cluster_to_fnums_dict[clusters[i]] = []
        cluster_to_fnums_dict[clusters[i]].append(fnums_in_order_list[i])
    cluster_to_hex_colors_proportions_list = [0]*len(kmeans.cluster_centers_)
    for ind,cluster in enumerate(kmeans.cluster_centers_):
        total_pixels = sum(cluster)
        cluster_to_hex_colors_proportions_list[ind] = {quantized_images_breakdown.colors_definitions[i]:(qcol_count/total_pixels) for i,qcol_count in enumerate(cluster)}

    pickle.dump(cluster_to_hex_colors_proportions_list,open("%s/data/cluster_to_hex_colors_proportions_list_Q%d_K%d%s.p"%(DATA_PATH,Q,K,info_str),"wb"))
    pickle.dump(fnum_to_cluster_dict,open("%s/data/fnum_to_cluster_number_dict_Q%d_K%d%s.p"%(DATA_PATH,Q,K,info_str),"wb"))
    pickle.dump(fnum_to_distance_to_cluster_dict,open("%s/data/fnum_to_distance_to_cluster_dict_Q%d_K%d%s.p"%(DATA_PATH,Q,K,info_str),"wb"))
    pickle.dump(cluster_to_fnums_dict,open("%s/data/cluster_number_to_fnum_dict_Q%d_K%d%s.p"%(DATA_PATH,Q,K,info_str),"wb"))
    make_react_codes_for_cluster_area_charts(Q,K,color_rep=color_rep, remove_monochrome=remove_monochrome, remove_predom_faces=remove_predom_faces,remove_skin=remove_skin)
    make_clusters_react_codes(Q,K,color_rep=color_rep, remove_monochrome=remove_monochrome, remove_predom_faces=remove_predom_faces,remove_skin=remove_skin)

def make_dict_of_year_to_cluster_prop(Q, K, info_str):
    fnum_to_cluster_dict = pickle.load(open("%s/data/fnum_to_cluster_number_dict_Q%d_K%d%s.p"%(DATA_PATH,Q,K, info_str),"rb"))
    clusters_list = list(pickle.load(open("%s/data/cluster_number_to_fnum_dict_Q%d_K%d%s.p"%(DATA_PATH,Q,K, info_str),"rb")).keys())
    year_to_fnums_dict = pickle.load(open("%s/data/basics/year_to_fnums_dict.p"%(DATA_PATH),"rb"))
    year_to_cluster_prop_dict = {}
    for year in year_to_fnums_dict.keys():
        fnums = [fnum for fnum in year_to_fnums_dict[year] if fnum in fnum_to_cluster_dict]
        year_to_cluster_prop_dict[year] = {}
        for cluster in clusters_list:
            year_to_cluster_prop_dict[year][cluster] = 0
        for fnum in fnums:
            cluster = fnum_to_cluster_dict[fnum]
            year_to_cluster_prop_dict[year][cluster]+=1/len(fnums)
    return year_to_cluster_prop_dict

def get_ordered_list_of_clusters(Q, K, info_str):
    cluster_to_fnums_dict = pickle.load(open("%s/data/cluster_number_to_fnum_dict_Q%d_K%d%s.p"%(DATA_PATH,Q,K, info_str),"rb"))
    tup_list = [[cluster,len(cluster_to_fnums_dict[cluster])] for cluster in cluster_to_fnums_dict.keys()]
    return [tup[0] for tup in sorted(tup_list, key = lambda x: x[1])]

def make_react_codes_for_cluster_area_charts(Q, K, color_rep="rgb",remove_monochrome=False, remove_predom_faces = False, remove_skin=False):
    print ("Starting React Codes for Cluster Area Charts")
    info_str = GetPixelsHelpers.get_pixels_dict_info_string(color_rep=color_rep, remove_monochrome=remove_monochrome, remove_predom_faces=remove_predom_faces,remove_skin=remove_skin)
    year_to_cluster_props_dict = make_dict_of_year_to_cluster_prop(Q=Q,K=K,info_str=info_str)
    clusters_list = get_ordered_list_of_clusters(Q=Q,K=K,info_str=info_str)

    years_sum_so_far_dict = {} #react does not stack itself, so manually stacking
    my_str = "color_clustering_data:[ \n"
    count = 1
    for cluster in clusters_list:
        my_str += "\t  {\"name\":\"Color Cluster %d\",\"data\": {\n"%count
        count +=1
        for year in range(1852,2020):
            if year not in year_to_cluster_props_dict:
                continue
            current_prop = year_to_cluster_props_dict[year][cluster]
            if year not in years_sum_so_far_dict:
                 years_sum_so_far_dict[year] = 0
            years_sum_so_far_dict[year] += current_prop
            my_str += " '%d': %f ,"%(year,years_sum_so_far_dict[year])
        my_str = my_str[:-1]+"\t}},\n"
    my_str = my_str[:-2]+"\n],\n"

    cluster_to_hex_colors_proportions_list = pickle.load(open("%s/data/cluster_to_hex_colors_proportions_list_Q%d_K%d%s.p"%(DATA_PATH,Q,K,info_str),"rb"))
    new_cluster_to_hex_colors_proportions_list = {clusters_list.index(cluster):cluster_to_hex_colors_proportions_list[cluster] for cluster in cluster_to_hex_colors_proportions_list.keys()}
    my_str += "cluster_to_hex:  %s ,\n"%new_cluster_to_hex_colors_proportions_list

    text_file = open("%s/data/react-codes/react_color_clustering_area_chart_codes_Q%d_K%d%s.txt"%(DATA_PATH,info_str), "w")
    text_file.write(my_str)
    text_file.close()
    print ("Done with Area Chart Color React codes")

def make_clusters_react_codes(Q,K, color_rep="rgb",remove_monochrome=False, remove_predom_faces = False, remove_skin=False):
    print ("Starting making Clustering React codes")
    info_str = GetPixelsHelpers.get_pixels_dict_info_string(color_rep=color_rep, remove_monochrome=remove_monochrome, remove_predom_faces=remove_predom_faces,remove_skin=remove_skin)

    fnum_to_url_dict = pickle.load(open("%s/data/basics/fnum_to_flickr_url_dict.p"%DATA_PATH,"rb"))
    cluster_to_fnums_dict = pickle.load(open("%s/data/cluster_number_to_fnum_dict_Q%d_K%d%s.p"%(DATA_PATH,Q,K,info_str),"rb"))
    fnum_to_distance_to_cluster_dict = pickle.load(open("%s/data/fnum_to_distance_to_cluster_dict_Q%d_K%d%s.p"%(DATA_PATH,Q,K,info_str),"rb"))
    for i in range(0,num_clusters):
        cluster_to_fnums_dict[i].sort(key = lambda x: fnum_to_distance_to_cluster_dict[x])
    min_len = reduce((lambda x, y: min(x,y)), [len(fnums) for fnums in cluster_to_fnums_dict.values()])
    # CLUSTER LABELS
    cluster_to_hex_colors_proportions_list = pickle.load(open("%s/data/cluster_to_hex_colors_proportions_list_Q%d_K%d%s.p"%(DATA_PATH,Q,K,info_str),"rb"))
    my_str = "labels:  %s ,\n"%cluster_to_hex_colors_proportions_list

    # IMAGES
    my_str += "images: [\n"
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
                my_str += "%s,"%bgr_list_to_hex_list(fnum_to_palettes_dict[curr_fnum]) #TODO this is from old code ... should update in the future
            else:
                my_str += "[],"
        my_str = my_str[:-1]+"],\n"
    my_str = my_str[:-2]+"\n],\n"

    # QUANTIZED COLORS: each one gets a dict from hex color to proportion
    quantized_images_breakdown = pickle.load(open("%s/data/quantized_images_breakdown_Q%d%s.p"%(DATA_PATH,Q,info_str),"rb"))
    fnums_to_hex_colors_proportions_dict = quantized_images_breakdown.get_ims_to_hex_colors_proportions_dict()
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

    text_file = open("%s/data/react-codes/react_color_clustering_page_codes_Q%d_K%d%s.txt"%(DATA_PATH,Q,K,info_str), "w")
    text_file.write(my_str)
    text_file.close()
    print ("Done with Clustering React codes")
