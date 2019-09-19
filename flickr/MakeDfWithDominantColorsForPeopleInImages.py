from matplotlib import pyplot as plt
import pandas as pd
import pickle
import os
from ColorThiefModified import ColorThief

#df =  pd.read_csv("data/url_title_and_file_data.csv")
#fnames_list = df[["file_name"]].values.tolist()
fnames_list = [["images/14.jpg"]]
results = []
count = 0
for fname in fnames_list:
    fname_num = fname[0].split("/")[-1]
    fname_num = (int) (fname_num.split(".jpg")[0])
    print (fname_num)
    res = pickle.load(open("data/images/mask_rcnn_results/res_%d.p"%fname_num,"rb"))
    orig_img = res[0]
    masks = res[1]
    ids = res[2]
    scores = res[3]

    people_indices = []
    for i in range(0,masks.shape[0]): #the masks we have for people
        if ids[i] == 0:
            people_indices.append(i)

    if len(people_indices) == 0:
        continue

    my_pixels = []
    for ind in people_indices:
        curr_mask =  masks[ind]
        for row in range(0,curr_mask.shape[0]):
            for col in range(0,curr_mask.shape[1]):
                my_pixels.append(orig_img[row][col])
    ct = ColorThief(my_pixels)
    color_list = ct.get_palette()
    results.append([fname_num,color_list])

    if count % 20 == 0: #save frequently to avoid having to rerun too often
        df=pd.DataFrame(results)
        urls_df.columns = ["fname_num","colors_list"]
        urls_df.to_csv("data/color_palettes.csv")
    count +=1

df=pd.DataFrame(results)
urls_df.columns = ["fname_num","colors_list"]
urls_df.to_csv("data/color_palettes.csv")
