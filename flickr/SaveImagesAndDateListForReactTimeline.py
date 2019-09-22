import pandas as pd
import pickle
import os

def get_images_code_for_react():
    df =  pd.read_csv("data/url_title_and_file_data.csv")
    my_list = df[["url","year","file_name"]].values.tolist()
    my_list.sort(key=lambda x: x[1])


    my_str = ""
    my_str += "  images: [\n"
    for im in my_list:
        #also filter out images that do not have people (TODO: keep clothes only)
        fname_num = im[2].split("/")[-1]
        fname_num = (int) (fname_num.split(".jpg")[0])
        mask_file_name ="data/images/mask_rcnn_results/res_%d.p"%fname_num
        if not os.path.exists(mask_file_name):
            continue
        res = pickle.load(open(mask_file_name,"rb"))
        masks = res[1]
        ids = res[2]
        hasPerson = False
        for i in range(0,masks.shape[0]): #the masks we have for people
            if ids[i] == 0:
                hasPerson = True
                break
        if not hasPerson:
            continue
        # regular add to string if it's fine
        url = im[0]
        p1 = url.split("_")[0]
        url = p1+"_n.jpg" #small size image http://joequery.me/code/flickr-api-image-search-python/
        my_str += "['%s','%s'],\n"%(url,im[1])
    my_str = my_str[:-2]+"\n"
    my_str += "],"

    text_file = open("data/react_default_images_timeline_code_lists_and_imports.txt", "w")
    text_file.write(my_str)
    text_file.close()

get_images_code_for_react()
