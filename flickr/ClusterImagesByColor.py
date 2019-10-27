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

DATA_PATH = "."
try:
    f=open("data_location.txt", "r")
    DATA_PATH  = f.read().strip()
except:
    print ("data is right here")

def get_pixels_in_file(fname_num,every_few = 20):
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
    print (im.shape)
    print (fname_num)
    my_pixels = []
    for ind in people_indices:
        curr_mask =  masks[ind]
        for row in range(0,curr_mask.shape[0]):
            for col in range(0,curr_mask.shape[1]):
                my_pixels.append(im[row][col])
    return shuffle(my_pixels, random_state=0)[:max(36000,int(len(my_pixels)/every_few))] #dont let any image return too many pixels

def make_clusters(num_clusters=7):
    n_colors = 21
    # Load all of my "dom_col_images"
    df =  pd.read_csv("%s/data/url_title_and_file_data.csv"%DATA_PATH)
    fnames_list = df[["file_name"]].values.tolist()

    all_colors = []
    for fname in fnames_list:
        fname_num = fname[0].split("/")[-1]
        fname_num = (int) (fname_num.split(".jpg")[0])
        all_colors.extend(get_pixels_in_file(fname_num))
    all_colors = np.array(all_colors)
    image_array = all_colors
    print (image_array)
    print("Fitting model on a small sub-sample of the data")
    t0 = time()
    image_array_sample = shuffle(image_array, random_state=0)[:500000]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
    print("done in %0.3fs." % (time() - t0))

    # Get labels for all points
    print("Predicting color indices on each image")
    fname_nums_in_order_list = []
    color_labels_one_hots= []
    for fname in fnames_list:
        fname_num = fname[0].split("/")[-1]
        fname_num = (int) (fname_num.split(".jpg")[0])
        if fname_num in palettes:
            list_ver =  kmeans.predict(palettes[fname_num])
            one_hot_ver = [0]*n_colors
            for el in list_ver:
                one_hot_ver[el] +=1
            fname_nums_in_order_list.append(fname_num)
            color_labels_one_hots.append(one_hot_ver)

    pickle.dump(color_labels_one_hots,open("%s/data/per_image_color_labels_one_hots.p"%DATA_PATH,"wb"))
    print ("Clustering on those counts per color labels")

    # random categorical data
    #data = np.random.choice(20, (100, 10))
    #print (data)
    kmeans = KMeans(n_clusters=num_clusters,random_state=0).fit(color_labels_one_hots)

    clusters = kmeans.predict(color_labels_one_hots)
    fname_to_cluster_dict = {}
    cluster_to_fnames_dict = {}
    for i in range(0,len(fname_nums_in_order_list)):
        fname_to_cluster_dict[fname_nums_in_order_list[i]] = clusters[i]
        if clusters[i] not in cluster_to_fnames_dict:
            cluster_to_fnames_dict[clusters[i]] = []
        cluster_to_fnames_dict[clusters[i]].append(fname_nums_in_order_list[i])
    pickle.dump(fname_to_cluster_dict,open("%s/data/file_num_to_cluster_number_dict.p"%DATA_PATH,"wb"))
    pickle.dump(cluster_to_fnames_dict,open("%s/data/cluster_number_to_file_num_dict.p"%DATA_PATH,"wb"))
    # Print the cluster centroids
    #print(km.cluster_centroids_)

    #make dictionary of cluster number to related years
    year_and_fname = df[["year","file_name"]].values.tolist()
    fname_nums_to_year_dict = {}
    for el in fnames_list:
        year = int(fname[0])
        fname_num = fname[1].split("/")[-1]
        fname_num = (int) (fname_num.split(".jpg")[0])
        fname_nums_to_year_dict[fname_num] = year
    color_cluster_to_year_dict = {}
    for cc in color_cluster_to_year_dict.keys():
        fname_num_list = color_cluster_to_year_dict[year]
        color_cluster_to_year_dict[cc] = [fname_nums_to_year_dict[fn] for fn in fnames_num_list]
    pickle.dump(cluster_to_years_dict,open("%s/data/color_cluster_number_to_years_dict.p"%DATA_PATH,"wb"))
    years_to_most_common_cluster_dict = {}
    for el in fnames_list:
        year = int(fname[0])
        fname_num = fname[1].split("/")[-1]
        fname_num = (int) (fname_num.split(".jpg")[0])
        curr_cluster = fname_to_cluster_dict[fname_num]
        if year not in years_to_most_common_cluster_dict:
            years_to_most_common_cluster_dict[year] = []
        years_to_most_common_cluster_dict[year].append(curr_cluster)
    for year in years_to_most_common_cluster_dict:
        years_to_most_common_cluster_dict[year] = Counter(years_to_most_common_cluster_dict[year]).most_common(1)[0][0]
    pickle.dump(years_to_most_common_cluster_dict,open("%s/data/years_to_most_common_color_cluster_dict.p"%DATA_PATH,"wb"))

def make_react_codes(num_clusters=7):
    # Make as many lists of 7 images (one for each cluster) as we can to show
    df =  pd.read_csv("%s/data/url_title_and_file_data.csv"%DATA_PATH)
    my_urls_list = df[["url","year","file_name"]].values.tolist()
    # Make filename num to url dict
    fnum_to_url_dict = {}
    for el in my_urls_list:
        url = el[0]
        p1 = url.split("_o")[0]
        url = p1+"_n.jpg"
        fname_num = el[2].split("/")[-1]
        fname_num = (int) (fname_num.split(".jpg")[0])
        fnum_to_url_dict[fname_num]=url
    # Make react code
    cluster_to_fnames_dict = pickle.load(open("%s/data/cluster_number_to_file_num_dict.p"%DATA_PATH,"rb"))
    min_len = min(len(cluster_to_fnames_dict[0]),len(cluster_to_fnames_dict[1]),len(cluster_to_fnames_dict[2]),len(cluster_to_fnames_dict[3]),len(cluster_to_fnames_dict[4]),len(cluster_to_fnames_dict[5]),len(cluster_to_fnames_dict[6]))
    my_str = "images: [\n"
    for i in range(0, min_len):
        my_str += "['%s','%s','%s','%s','%s','%s','%s'],\n"%(fnum_to_url_dict[cluster_to_fnames_dict[0][i]],fnum_to_url_dict[cluster_to_fnames_dict[1][i]],
        fnum_to_url_dict[cluster_to_fnames_dict[2][i]],fnum_to_url_dict[cluster_to_fnames_dict[3][i]],fnum_to_url_dict[cluster_to_fnames_dict[4][i]],
        fnum_to_url_dict[cluster_to_fnames_dict[5][i]],fnum_to_url_dict[cluster_to_fnames_dict[6][i]]) # need to make number of these variable based on num clusters
    my_str = my_str[:-2]+"\n],"
    text_file = open("%s/data/react-codes/react_color_clustering_page_codes.txt"%DATA_PATH, "w")
    text_file.write(my_str)
    text_file.close()


make_clusters()
make_react_codes()
