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
            if (np.std(good_hues) < 2):#threshold
                print ("fnum %d is monochrome" %fnum)
                monochrome_list.append(fnum)
            else:
                print ("fnum %d is colored" %fnum)
    pickle.dump(monochrome_list,open("%s/data/monochrom_list_%s.p"%(DATA_PATH,method),"wb"))

def make_react_code(method = "hsv"):
    fnum_to_url_dict = pickle.load(open("%s/data/basics/fnum_to_flickr_url_dict.p"%DATA_PATH,"rb"))
    fnums_list = pickle.load(open("%s/data/basics/fnums_list.p"%DATA_PATH,"rb"))
    if method == "hsv":
        monochrome_list = pickle.load(open("%s/data/monochrom_list_%s.p"%(DATA_PATH,method),"rb"))
        my_str = "export function makeHSVMonochromeData() {\n"
        my_str += "return "
    else:
        return

    colored_list = [fnum for fnum in fnums_list if fnum not in set(monochrome_list)]
    my_str += "["
    for i in range(0,min(len(colored_list),len(monochrome_list))):
        if monochrome_list[i] in fnum_to_url_dict and colored_list[i] in fnum_to_url_dict:
        my_str += "['%s', '%s'],\n"%(fnum_to_url_dict[monochrome_list[i]],fnum_to_url_dict[colored_list[i]])
    my_str = my_str[:-2]+"]\n}\n"

    text_file = open("%s/data/react-codes/react_monochrome_detection_%s.txt"%(DATA_PATH,method), "w")
    text_file.write(my_str)
    text_file.close()
    print ("Done with React codes")

#save_monochrome_fnums_list()
make_react_code()
