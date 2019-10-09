# get k groups of images, with the 10 most dominant colors being the criteria for clustering
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

n_colors = 20

# Load all of my "dom_col_images"
df =  pd.read_csv("data/url_title_and_file_data.csv")
fnames_list = df[["file_name"]].values.tolist()
palettes = pickle.load(open("data/color_palettes.p","rb"))

all_colors = []
for fname in fnames_list:
    fname_num = fname[0].split("/")[-1]
    fname_num = (int) (fname_num.split(".jpg")[0])
    if fname_num in palettes:
        all_colors.extend(palettes[fname_num])
all_colors = np.array(all_colors)
image_array = all_colors

print("Fitting model on a small sub-sample of the data")
t0 = time()
image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
print("done in %0.3fs." % (time() - t0))

# Get labels for all points
print("Predicting color indices on each image")
fnames_in_order_list = []
color_labels_one_hots= []
for fname in fnames_list:
    fname_num = fname[0].split("/")[-1]
    fname_num = (int) (fname_num.split(".jpg")[0])
    if fname_num in palettes:
        list_ver =  kmeans.predict(palettes[fname_num])
        one_hot_ver = [0]*n_colors
        for el in list_ver:
            one_hot_ver[el] +=1
        fnames_in_order_list.append(fname_num)
        color_labels_one_hots.append(one_hot_ver)

print ("Clustering on those one hot labels")
from kmodes.kmodes import KModes

# random categorical data
#data = np.random.choice(20, (100, 10))
#print (data)
kmodes = KModes(n_clusters=7, init='Huang', n_init=5, verbose=0)

clusters = kmodes.fit_predict(color_labels_one_hots)
fname_to_cluster_dict = {}
for i in range(0,len(fnames_in_order_list)):
    fname_to_cluster_dict[fnames_in_order_list[i]] = clusters[i]

pickle.dump(fname_to_cluster_dict,open("data/file_num_to_cluster_number_dict.p","wb"))
# Print the cluster centroids
#print(km.cluster_centroids_)
