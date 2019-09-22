from matplotlib import pyplot as plt
import pandas as pd
import pickle
import os
from ColorThiefModified import ColorThief
import re

def make_df():
    df =  pd.read_csv("data/url_title_and_file_data.csv")
    fnames_list = df[["file_name"]].values.tolist()

    df =  pd.read_csv("data/color_palettes.csv")
    done_nums = set([el[0] for el in df[["fname_num"]].values.tolist()])

    results = []
    count = 1
    for fname in fnames_list:
        fname_num = fname[0].split("/")[-1]
        fname_num = (int) (fname_num.split(".jpg")[0])
        if fname_num in done_nums:
            print ("already done with %d"%fname_num)
            continue
        print (fname_num)
        try:
            res = pickle.load(open("data/images/mask_rcnn_results/res_%d.p"%fname_num,"rb"))
        except:
            continue
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

        if count % 5 == 0: #save frequently to avoid having to rerun too often
            df=pd.DataFrame(results)
            df.columns = ["fname_num","colors_list"]
            df.to_csv("data/color_palettes.csv")
            print ("current part saved")
        count +=1

    if len(results) != 0:
        df=pd.DataFrame(results)
        df.columns = ["fname_num","colors_list"]
        df.to_csv("data/color_palettes.csv")

def convert_df_into_list_for_react():
    df =  pd.read_csv("data/url_title_and_file_data.csv")
    years_list = df[["file_name","year"]].values.tolist()

    df =  pd.read_csv("data/color_palettes.csv")
    done_nums = {el[0]:el[1] for el in df[["fname_num","colors_list"]].values.tolist()}

    my_str = "["
    for el in years_list:
        fname_num = int(el[0].split(".")[0].split("/")[-1])
        year = el[1]
        if fname_num in done_nums:
            #https://stackoverflow.com/questions/3380726/converting-a-rgb-color-tuple-to-a-six-digit-code-in-python
            my_str += "["
            #print (len( done_nums[fname_num].split("),"))) #not sure why only 9
            print  (done_nums[fname_num])
            for color in done_nums[fname_num].split("),"):
                r1 = re.findall(r"[0-9]+",color)
                #print (r1)
                my_str+="'#%02x%02x%02x'," %(int(r1[0]),int(r1[1]),int(r1[2]))
            my_str +="%d"%year
            my_str += "],\n"
    my_str = my_str[:-2] + "],"
    text_file = open("data/react_colors_list_for_colors_slides.txt", "w")
    text_file.write(my_str)
    text_file.close()
convert_df_into_list_for_react()
