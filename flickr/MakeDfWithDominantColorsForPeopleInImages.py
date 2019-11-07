from matplotlib import pyplot as plt
import pandas as pd
import os, re, pickle,math
from ColorThiefModified import ColorThief
import colorsys
from collections import deque
import numpy as np
import cv2
from sklearn.utils import shuffle

DATA_PATH = ""
try:
    f=open("data_location.txt", "r")
    DATA_PATH  = f.read().strip()
except:
    print ("data is right here")

def get_pixels_in_fnums(fnums,sample_amount):
    all_pixels = []
    for fnum in fnums:
        try:
            res = pickle.load(open("%s/data/images/mask_rcnn_results/res_%d.p"%(DATA_PATH,fnum),"rb"))
        except:
            continue
        masks, ids = res[1], res[2]
        people_indices = [i for i in range(0,masks.shape[0]) if ids[i] == 0]
        if len(people_indices) == 0:
            continue

        im = cv2.imread("%s/data/images/smaller_images/%d.jpg"%(DATA_PATH,fnum))
        if (im.shape[0] != masks.shape[1] or im.shape[1] != masks.shape[2]):
            print ("Dimensional problem on %d, image:%d, %d vs masks: %d, %d"%(fnum, im.shape[0],im.shape[1],masks.shape[1],masks.shape[2]))
            continue

        sum_mask = masks[people_indices[0]]
        for ind in people_indices[1:]:
            sum_mask += masks[ind]
        my_pixels = im[sum_mask!=0]
        all_pixels.extend(shuffle(my_pixels, random_state=0)[:int(len(my_pixels)/sample_amount)])

    return shuffle(all_pixels,random_state=0)[:360000]

def make_yearly_color_palettes(num_colors=10,sample_amount=5):
    year_to_fnums_dict=pickle.load(open("%s/data/basics/year_to_fnums_dict.p"%DATA_PATH,"rb"))
    year_to_color_palettes_dict = {}
    for year in year_to_fnums_dict.keys():
        if year in year_to_color_palettes_dict:
            continue
        year_pixels = get_pixels_in_fnums(year_to_fnums_dict[year],sample_amount)
        if len(year_pixels) == 0:
            print ("year %d has no valid pixels to use"%year)
            continue
        ct = ColorThief(year_pixels)
        color_list = ct.get_palette(color_count=num_colors)
        year_to_color_palettes_dict[year] = color_list
    pickle.dump(year_to_color_palettes_dict,open("%s/data/year_to_%d_color_palettes_dict.p"%(DATA_PATH,num_colors),"wb"))

def make_color_palettes(num_colors=10,output_fname="color_palettes",sample_amount=1):
    df =  pd.read_csv("%s/data/url_title_and_file_data.csv"%DATA_PATH)
    fnames_list = df[["file_name"]].values.tolist()

    try:
        palettes = pickle.load(open("%s/data/%s.p"%(DATA_PATH,output_fname),"rb"))
    except:
        palettes = {}

    count = 0
    for fname in fnames_list:
        fname_num = fname[0].split("/")[-1]
        fname_num = (int) (fname_num.split(".jpg")[0])
        if fname_num in palettes:
            #print ("already done with %d"%fname_num)
            continue
        try:
            res = pickle.load(open("%s/data/images/mask_rcnn_results/res_%d.p"%(DATA_PATH,fname_num),"rb"))
        except:
            continue
        masks = res[1]
        ids = res[2]
        scores = res[3]

        people_indices = []
        for i in range(0,masks.shape[0]): #the masks we have for people
            if ids[i] == 0:
                people_indices.append(i)

        if len(people_indices) == 0:
            continue

        print (fname_num)
        my_pixels = []
        inner_count = 0
        im = cv2.imread("%s/data/images/smaller_images/%d.jpg"%(DATA_PATH,fname_num))
        if (im.shape[0] != masks.shape[1] or im.shape[1] != masks.shape[2]):
            print ("some dimensional problem with the mask and image for this one: %d"%fname_num)
            continue
        for ind in people_indices:
            curr_mask =  masks[ind]
            for row in range(0,curr_mask.shape[0]):
                for col in range(0,curr_mask.shape[1]):
                    if inner_count % sample_amount == 0: #sample
                        my_pixels.append(im[row][col])
                    inner_count +=1
        ct = ColorThief(my_pixels)
        color_list = ct.get_palette(color_count=num_colors)
        palettes[fname_num] = color_list

        if count % 5 == 0: #save frequently to avoid having to rerun too often
            pickle.dump(palettes,open("%s/data/%s.p"%(DATA_PATH,output_fname),"wb"))
            print ("current part saved")
        count +=1

    if len(palettes) != 0:
        pickle.dump(palettes,open("%s/data/%s.p"%(DATA_PATH,output_fname),"wb"))
    print ("Done")


# STUFF RELATED TO REACT CODE

def diff_score(prev_row,curr_row):
    sum_dist = 0
    curr_row =  [colorsys.rgb_to_hsv(c[0],c[1],c[2]) for c in curr_row]
    prev_row =  [colorsys.rgb_to_hsv(c[0],c[1],c[2]) for c in prev_row]
    for i in range(0,len(curr_row)):
        sum_dist += math.sqrt(math.pow((prev_row[i][0]/255.0-curr_row[i][0]/255.0),2)+math.pow((prev_row[i][1]/255.0-curr_row[i][1]/255.0),2)+math.pow((prev_row[i][2]/255.0-curr_row[i][2]/255.0),2))
        # diagonals
        sum_dist += math.sqrt(math.pow((prev_row[i][0]/255.0-curr_row[i-1][0]/255.0),2)+math.pow((prev_row[i][1]/255.0-curr_row[i-1][1]/255.0),2)+math.pow((prev_row[i-1][2]/255.0-curr_row[i][2]/255.0),2))/2.0
        sum_dist += math.sqrt(math.pow((prev_row[i-1][0]/255.0-curr_row[i][0]/255.0),2)+math.pow((prev_row[i-1][1]/255.0-curr_row[i][1]/255.0),2)+math.pow((prev_row[i-1][2]/255.0-curr_row[i][2]/255.0),2))/2.0

    return sum_dist

def rotate_until_most_contig(prev_row,curr_row):
    d=deque(curr_row)
    min_diff_score = 999999999999
    best_rot = 0
    for i in range(0,len(curr_row)):
        d.rotate(1)
        my_diff_score = diff_score(prev_row,curr_row)
        if my_diff_score < min_diff_score:
            min_diff_score= my_diff_score
            best_rot = i+1
    d = deque(curr_row)
    d.rotate(best_rot)
    return list(d)
    #return curr_row

def sort_colors_lists(all_colors_list):
    new_all_colors_list = []
    for rgb_colors_list in all_colors_list:
        hue_colors_list = [colorsys.rgb_to_hsv(c[0],c[1],c[2])[0] for c in rgb_colors_list]
        rgb_colors_list = [x for _,x in sorted(zip(hue_colors_list,rgb_colors_list))]
        if len(new_all_colors_list) == 0:
            new_all_colors_list.append(rgb_colors_list)
        else:
            #rotate until rgb colors list matches the prev as best as possible
            new_all_colors_list.append( rotate_until_most_contig(new_all_colors_list[len(new_all_colors_list)-1],rgb_colors_list))
    return new_all_colors_list

def make_colors_list_for_react():
    df =  pd.read_csv("%s/data/url_title_and_file_data.csv"%DATA_PATH)
    years_list = df[["file_name","year"]].values.tolist()
    years_list.sort(key=lambda x: x[1])
    palettes = pickle.load(open("%s/data/color_palettes.p"%DATA_PATH,"rb"))

    used_years_list = []
    all_colors_list = []
    years_start_and_end = {}
    i = 0
    for el in years_list:
        fname_num = int(el[0].split(".")[0].split("/")[-1])
        year = el[1]
        if fname_num in palettes:
            used_years_list.append(year)
            all_colors_list.append(palettes[fname_num])
            if year not in years_start_and_end:
                years_start_and_end[year] = (i,i+1)
            else:
                years_start_and_end[year] = (years_start_and_end[year][0],i+1)
            i+=1

    # sort rows within the same year
    # sort by average hue of the row
    for year in years_start_and_end.keys():
        avg_hues_list = []
        for  row in  all_colors_list[years_start_and_end[year][0]:years_start_and_end[year][1]]:
            #ct = ColorThief(row)
            #dom_col = ct.get_color(quality=100)
            #avg_hues_list.append(colorsys.rgb_to_hsv(dom_col[0],dom_col[1],dom_col[2])[0])
            avg_hues_list.append(np.mean([colorsys.rgb_to_hsv(c[0],c[1],c[2])[0] for c in row]))
        all_colors_list[years_start_and_end[year][0]:years_start_and_end[year][1]] =  [x for _,x in sorted(zip(avg_hues_list,all_colors_list[years_start_and_end[year][0]:years_start_and_end[year][1]]))]

    # sort within row
    all_colors_list = sort_colors_lists(all_colors_list)

    # MAKE STRING
    my_str = "colors:["
    for i in range(0,len(all_colors_list)):
        curr_colors = all_colors_list[i]
        curr_year = used_years_list[i]
        my_str += "["
        for c in curr_colors:
            my_str+="'#%02x%02x%02x'," %(c[0],c[1],c[2])
        my_str +="%d"%curr_year
        my_str += "],\n"
    my_str = my_str[:-2] + "],\n"

    text_file = open("%s/data/react_colors_list_for_colors_slides.txt"%DATA_PATH, "w")
    text_file.write(my_str)
    text_file.close()

def make_yearly_colors_list_for_react(num_colors=10):
    year_to_color_palettes_dict=pickle.load(open("%s/data/year_to_%d_color_palettes_dict.p"%(DATA_PATH,num_colors),"rb"))

    my_str = "yearly_colors:["
    for curr_year in sorted(list(year_to_color_palettes_dict.keys())):
        curr_colors = year_to_color_palettes_dict[curr_year]
        my_str += "["
        for i in range(0,len(curr_colors)):
            c = curr_colors[i]
            my_str+="'#%02x%02x%02x'," %(c[0],c[1],c[2])
        my_str +="%d"%curr_year
        my_str += "],\n"
    my_str = my_str[:-2] + "],"

    text_file = open("%s/data/react-codes/react_yearly_colors_for_colors_slides.txt"%DATA_PATH, "w")
    text_file.write(my_str)
    text_file.close()
    print ("Done making yearly colors react codes")

#make_color_palettes(num_colors=5,output_fname="5_color_fnum_to_palettes_dict")
#make_colors_list_for_react()
#make_yearly_color_palettes(num_colors=10)
make_yearly_colors_list_for_react(num_colors=10)
