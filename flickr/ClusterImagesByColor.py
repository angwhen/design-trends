# get k groups of images
# https://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html
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

DATA_PATH = "."
try:
    f=open("data_location.txt", "r")
    DATA_PATH  = f.read().strip()
except:
    print ("data is right here")

def rgb_list_to_hex_list(rgb_list):
    return ["#%02x%02x%02x"%(int(c[0]),int(c[1]),int(c[2])) for c in rgb_list]

#https://stackoverflow.com/questions/35113979/calculate-distance-between-colors-in-hsv-space
def project_to_hsv_cone(p): #input out of 255, input should already be in hsv
    return (p[1]/255*p[2]/255*math.sin(p[0]/255*2*math.pi), p[1]/255*p[2]/255*math.cos(p[0]/255*2*math.pi), p[2]/255)

def hsv_cone_coords_to_hsv(p): #returns out of 255
    val = p[2]
    if val != 0:
        sat = math.sqrt((p[0]*p[0]+p[1]*p[1])/(val*val))
    else:
        sat = 0
    if sat!=0 and val != 0:
        hue1 = math.asin(p[0]/sat/val) #figure out which result based on cosine
        hue2 = math.acos(p[1]/sat/val)
        if hue1 == hue2:
            hue = hue1
        elif hue1 < 0:
            if hue2 <= math.pi/2:
                hue = 2*math.pi+hue1
            else:
                hue = math.pi-hue1
        else:
            hue = hue2
    else:
        hue = 0
    return (hue*255/(2*math.pi),sat*255,val*255)

def get_pixels_in_file(fnum,every_few = 10,use_hsv=False):
    try:
        res = pickle.load(open("%s/data/images/mask_rcnn_results/res_%d.p"%(DATA_PATH,fnum),"rb"))
    except:
        return []
    masks, ids = res[1], res[2]

    people_indices = [i for i in range(0,masks.shape[0]) if ids[i] == 0]
    if len(people_indices) == 0:
        return []

    im = cv2.imread("%s/data/images/smaller_images/%d.jpg"%(DATA_PATH,fnum))
    if use_hsv:
        im = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
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

def get_all_colors_and_fnum_to_pixels_dict(use_hsv):
    fnums_list = pickle.load(open("%s/data/basics/fnums_list.p"%DATA_PATH,"rb"))
    monochrome_list = set(pickle.load(open("%s/data/monochrome_list_%s.p"%(DATA_PATH,"hsv"),"rb")))
    print ("Loading people pixels from images")
    all_colors = []
    fnum_to_pixels_dict = {}
    for fnum in fnums_list:
        if fnum in monochrome_list:
            continue
        all_pixels_curr = get_pixels_in_file(fnum,use_hsv=use_hsv)
        all_colors.extend(all_pixels_curr)
        if len(all_pixels_curr) != 0:
            fnum_to_pixels_dict[fnum] = all_pixels_curr
    return all_colors, fnum_to_pixels_dict

def make_clusters(num_quantized_colors = 5, num_clusters = 7, all_colors = None, fnum_to_pixels_dict = None, use_hsv = False):
    hsv_add_str = ""
    if use_hsv:
        hsv_add_str = "_hsv"
    fnums_list = pickle.load(open("%s/data/basics/fnums_list.p"%DATA_PATH,"rb"))
    if all_colors == None or fnum_to_pixels_dict == None:
        all_colors, fnum_to_pixels_dict = get_all_colors_and_fnum_to_pixels_dict(use_hsv)

    all_colors_array_sample = shuffle(np.array(all_colors), random_state=0)[:500000]
    if use_hsv:
        all_colors_array_sample = np.apply_along_axis(project_to_hsv_cone, 1, all_colors_array_sample)

    print ("Quantization")
    kmeans = KMeans(n_clusters=num_quantized_colors, max_iter=100,random_state=0).fit(all_colors_array_sample)
    # Quantize each image
    fnum_to_counts_of_each_color_in_image_dict= {}
    for fnum in fnums_list:
        if fnum in fnum_to_pixels_dict:
            if use_hsv:
                list_of_colors =  kmeans.predict(np.apply_along_axis(project_to_hsv_cone, 1, fnum_to_pixels_dict[fnum]))
            else:
                list_of_colors =  kmeans.predict(fnum_to_pixels_dict[fnum])
            counts_of_each_color_in_image = [0]*num_quantized_colors
            for color_ind in list_of_colors:
                counts_of_each_color_in_image[color_ind] +=1
            fnum_to_counts_of_each_color_in_image_dict[fnum] = counts_of_each_color_in_image

    centroids = kmeans.cluster_centers_
    if use_hsv:
        centroids = [hsv_cone_coords_to_hsv(col) for col in centroids]
        centroids = [colorsys.hsv_to_rgb(col[0]/255,col[1]/255,col[2]/255) for col in centroids]
        centroids = [[col[0]*255,col[1]*255,col[2]*255] for col in centroids]
    quantized_images_breakdown = QuantizedImageBreakdown(centroids,fnum_to_counts_of_each_color_in_image_dict)
    pickle.dump(quantized_images_breakdown,open("%s/data/quantized_images_breakdown_Q%d%s.p"%(DATA_PATH,num_quantized_colors,hsv_add_str),"wb"))

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
    cluster_to_hex_colors_proportions_list = [0]*len(kmeans.cluster_centers_)
    for ind,cluster in enumerate(kmeans.cluster_centers_):
        total_pixels = sum(cluster)
        print (cluster)
        print (total_pixels)
        print (cluster[0]/total_pixels)
        print (quantized_images_breakdown.colors_definitions[0])
        cluster_to_hex_colors_proportions_list[ind] = {quantized_images_breakdown.colors_definitions[i]:(qcol_count/total_pixels) for i,qcol_count in enumerate(cluster)}
    pickle.dump(cluster_to_hex_colors_proportions_list,open("%s/data/cluster_to_hex_colors_proportions_list_Q%d_K%d%s.p"%(DATA_PATH,num_quantized_colors,num_clusters,hsv_add_str),"wb"))
    pickle.dump(fnum_to_cluster_dict,open("%s/data/fnum_to_cluster_number_dict_Q%d_K%d%s.p"%(DATA_PATH,num_quantized_colors,num_clusters,hsv_add_str),"wb"))
    pickle.dump(cluster_to_fnums_dict,open("%s/data/cluster_number_to_fnum_dict_Q%d_K%d%s.p"%(DATA_PATH,num_quantized_colors,num_clusters,hsv_add_str),"wb"))

    make_dict_of_cluster_and_year(Q=num_quantized_colors,K=num_clusters,use_hsv=use_hsv)

def make_dict_of_cluster_and_year(Q=5,K=7,use_hsv=False):
    hsv_add_str = ""
    if use_hsv:
        hsv_add_str = "_hsv"
    fnum_to_cluster_dict = pickle.load(open("%s/data/fnum_to_cluster_number_dict_Q%d_K%d%s.p"%(DATA_PATH,Q,K,hsv_add_str),"rb"))
    cluster_to_fnums_dict = pickle.load(open("%s/data/cluster_number_to_fnum_dict_Q%d_K%d%s.p"%(DATA_PATH,Q,K,hsv_add_str),"rb"))
    fnums_to_year_dict = pickle.load(open("%s/data/basics/fnum_to_year_dict.p"%(DATA_PATH),"rb"))

    # Find all years associated with each color cluster
    color_cluster_to_years_dict = {}
    for cc in cluster_to_fnums_dict.keys():
        fnum_list = cluster_to_fnums_dict[cc]
        color_cluster_to_years_dict[cc] = [fnums_to_year_dict[fn] for fn in fnum_list]
    pickle.dump(color_cluster_to_years_dict,open("%s/data/color_cluster_number_to_years_dict_Q%d_K%d%s.p"%(DATA_PATH,Q,K,hsv_add_str),"wb"))

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
    pickle.dump(years_to_most_common_cluster_dict,open("%s/data/years_to_most_common_color_cluster_dict_Q%d_K%d%s.p"%(DATA_PATH,Q,K,hsv_add_str),"wb"))

def make_react_codes(Q = 5, K=7,use_hsv=False):
    print ("Starting making React codes")
    hsv_add_str = ""
    if use_hsv:
        hsv_add_str = "_hsv"
    fnum_to_url_dict = pickle.load(open("%s/data/basics/fnum_to_flickr_url_dict.p"%DATA_PATH,"rb"))
    cluster_to_fnums_dict = pickle.load(open("%s/data/cluster_number_to_fnum_dict_Q%d_K%d%s.p"%(DATA_PATH,Q,K,hsv_add_str),"rb"))
    for i in range(0,num_clusters):
        random.shuffle(cluster_to_fnums_dict[i])
    min_len = reduce((lambda x, y: min(x,y)), [len(fnums) for fnums in cluster_to_fnums_dict.values()])

    # CLUSTER LABELS
    cluster_to_hex_colors_proportions_list = pickle.load(open("%s/data/cluster_to_hex_colors_proportions_list_Q%d_K%d%s.p"%(DATA_PATH,Q,K,hsv_add_str),"rb"))
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
                my_str += "%s,"%rgb_list_to_hex_list(fnum_to_palettes_dict[curr_fnum])
            else:
                my_str += "[],"
        my_str = my_str[:-1]+"],\n"
    my_str = my_str[:-2]+"\n],\n"

    # QUANTIZED COLORS: each one gets a dict from hex color to proportion
    quantized_images_breakdown = pickle.load(open("%s/data/quantized_images_breakdown_Q%d%s.p"%(DATA_PATH,Q,hsv_add_str),"rb"))
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

    text_file = open("%s/data/react-codes/react_color_clustering_page_codes_Q%d_K%d%s.txt"%(DATA_PATH,Q,K,hsv_add_str), "w")
    text_file.write(my_str)
    text_file.close()
    print ("Done with React codes")

def make_dict_of_year_to_cluster_prop(Q=5,K=7,use_hsv=False):
    hsv_add_str = ""
    if use_hsv:
        hsv_add_str = "_hsv"
    fnum_to_cluster_dict = pickle.load(open("%s/data/fnum_to_cluster_number_dict_Q%d_K%d%s.p"%(DATA_PATH,Q,K,hsv_add_str),"rb"))
    clusters_list = list(pickle.load(open("%s/data/cluster_number_to_fnum_dict_Q%d_K%d%s.p"%(DATA_PATH,Q,K,hsv_add_str),"rb")).keys())
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

def get_ordered_list_of_clusters(Q=5,K=7,use_hsv=False):
    hsv_add_str = ""
    if use_hsv:
        hsv_add_str = "_hsv"
    cluster_to_fnums_dict = pickle.load(open("%s/data/cluster_number_to_fnum_dict_Q%d_K%d%s.p"%(DATA_PATH,Q,K,hsv_add_str),"rb"))
    tup_list = []
    for cluster in cluster_to_fnums_dict.keys():
        tup_list.append([cluster,len(cluster_to_fnums_dict[cluster])])
    return [tup[0] for tup in sorted(tup_list, key = lambda x: x[1])]

def make_react_codes_for_cluster_area_charts():
    print ("Starting React Codes for Cluster Area Charts")
    year_to_cluster_props_dict = make_dict_of_year_to_cluster_prop(Q=5,K=7,use_hsv=True)
    clusters_list = get_ordered_list_of_clusters(Q=5,K=7,use_hsv=True)

    years_sum_so_far_dict = {} #react does not stack itself, so manually stacking
    my_str = "color_clustering_data:[ \n"
    count = 1
    for cluster in clusters_list:
        my_str += "\t  {\"name\":\"Color Cluster %d\",\"data\": {\n"%count
        count +=1
        for year in range(1800,2020):
            if year not in year_to_cluster_props_dict:
                continue
            current_prop = year_to_cluster_props_dict[year][cluster]
            if year not in years_sum_so_far_dict:
                 years_sum_so_far_dict[year] = 0
            years_sum_so_far_dict[year] += current_prop
            my_str += " '%d': %f ,"%(year,years_sum_so_far_dict[year])
        my_str = my_str[:-1]+"\t}},\n"

    my_str = my_str[:-2]+"],\n"
    text_file = open("%s/data/react-codes/react_color_clustering_area_chart_codes.txt"%(DATA_PATH), "w")
    text_file.write(my_str)
    text_file.close()
    print ("Done with Area Chart Color React codes")

make_react_codes_for_cluster_area_charts()

"""use_hsv = True
all_colors, fnum_to_pixels_dict = get_all_colors_and_fnum_to_pixels_dict(use_hsv)
for num_clusters in [7]:
    for num_quantized_colors in [5,7,20]:
        print ("working on Q=%d, K = %d"%(num_quantized_colors,num_clusters))
        make_clusters(num_quantized_colors =num_quantized_colors,num_clusters=num_clusters,all_colors=all_colors, fnum_to_pixels_dict=fnum_to_pixels_dict,use_hsv=use_hsv)
        make_react_codes(Q =num_quantized_colors,K=num_clusters,use_hsv=use_hsv)"""
