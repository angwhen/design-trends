import pandas as pd
import pickle
import cv2
import numpy as np

try:
    DATA_PATH  = open("data_location.txt", "r").read().strip()
except:
    DATA_PATH = "."

#https://www.quora.com/What-is-the-most-reliable-algorithm-to-detect-if-an-RGB-Image-is-monochrome
def save_monochrome_fnums_list(method = "hsv"):
    fnums_list = pickle.load(open("%s/data/basics/fnums_list.p"%DATA_PATH,"rb"))
    monochrome_list = []
    for fnum in fnums_list:
        im = cv2.imread("%s/data/images/smaller_images/%d.jpg"%(DATA_PATH,fnum))
        if method == "hsv":
            hsv_im = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
            hsv_im = hsv_im.reshape(hsv_im.shape[0]*hsv_im.shape[1],3)
            good_hues = [col[0] for col in hsv_im if col[2] > 50]
            my_std = np.std(good_hues)
            if (my_std < 10):#threshold
                print ("fnum %d is monochrome with std %.2f" %(fnum,my_std))
                monochrome_list.append(fnum)
            else:
                print ("fnum %d is colored with std %.2f" %(fnum,my_std))
    pickle.dump(monochrome_list,open("%s/data/monochrome_list_%s.p"%(DATA_PATH,method),"wb"))
    return monochrome_list

def make_react_code_for_prop_monochrome_over_time(method = "hsv"):
    monochrome_list = pickle.load(open("%s/data/monochrome_list_%s.p"%(DATA_PATH,method),"rb"))
    year_to_fnums_dict=pickle.load(open("%s/data/basics/year_to_fnums_dict.p"%DATA_PATH,"rb"))
    fnum_to_year_dict=pickle.load(open("%s/data/basics/fnum_to_year_dict.p"%DATA_PATH,"rb"))
    year_to_monochrome_prop_dict = {}
    for fnum in monochrome_list:
        year = fnum_to_year_dict[fnum]
        if year not in year_to_monochrome_prop_dict:
            year_to_monochrome_prop_dict[year] = 0
        year_to_monochrome_prop_dict[year] +=1/float(len(year_to_fnums_dict))

    my_str = "data:[["
    for year in range(1800,2020):
        if year not in year_to_fnums_dict:
            continue
        proportion = 0
        if year in year_to_monochrome_prop_dict:
            proportion = year_to_monochrome_prop_dict[year]
        my_str += "{ x: %d, y: %f },"%(year,proportion)
    my_str += "]]\n"
    text_file = open("%s/data/react-codes/react_monochrome_proportion_chart_%s.txt"%(DATA_PATH,method), "w")
    text_file.write(my_str)
    text_file.close()
    print ("Done monochrome over time")

def make_react_code(method = "hsv"):
    fnum_to_url_dict = pickle.load(open("%s/data/basics/fnum_to_flickr_url_dict.p"%DATA_PATH,"rb"))
    fnums_list = pickle.load(open("%s/data/basics/fnums_list.p"%DATA_PATH,"rb"))
    if method == "hsv":
        monochrome_list = pickle.load(open("%s/data/monochrome_list_%s.p"%(DATA_PATH,method),"rb"))
        my_str = "export function makeHSVMonochromeData() {\n"
        my_str += "return "
    else:
        return

    colored_list = [fnum for fnum in fnums_list if fnum not in set(monochrome_list)]
    my_str += "["
    for i in range(0,min(len(colored_list),len(monochrome_list))):
        if monochrome_list[i] in fnum_to_url_dict and colored_list[i] in fnum_to_url_dict:
            my_str += "\t['%s', '%s'],\n"%(fnum_to_url_dict[monochrome_list[i]],fnum_to_url_dict[colored_list[i]])
    my_str = my_str[:-2]+"]\n}\n"

    text_file = open("%s/data/react-codes/react_monochrome_detection_%s.txt"%(DATA_PATH,method), "w")
    text_file.write(my_str)
    text_file.close()
    print ("Done with React codes")

make_react_code_for_prop_monochrome_over_time(method = "hsv")
