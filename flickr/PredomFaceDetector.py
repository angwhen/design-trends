import pandas as pd
import pickle
import cv2
import numpy as np

try:
    DATA_PATH  = open("data_location.txt", "r").read().strip()
except:
    DATA_PATH = "."

def save_predom_faces_fnums_list():
    fnums_list = pickle.load(open("%s/data/basics/fnums_list.p"%DATA_PATH,"rb"))
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    predom_faces_list = set([])
    for fnum in fnums_list:
        img = cv2.imread("%s/data/images/smaller_images/%d.jpg"%(DATA_PATH,fnum))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # face cascade needs grayscale
        faces = face_cascade.detectMultiScale(gray, 1.1, 10)
        for (x, y, w, h) in faces:
            if h > 0.75*img.shape[0] || h > 0.75*img.shape[0]:
                predom_faces_list.add(fnum)
                break
    pickle.dump(predom_faces_list,open("%s/data/predom_faces_list.p"%(DATA_PATH),"wb"))

def make_react_code():
    fnum_to_url_dict = pickle.load(open("%s/data/basics/fnum_to_flickr_url_dict.p"%DATA_PATH,"rb"))
    fnums_list = pickle.load(open("%s/data/basics/fnums_list.p"%DATA_PATH,"rb"))
    predom_faces_list = pickle.load(open("%s/data/predom_faces_list.p"%(DATA_PATH),"rb"))
    my_str = "export function makePredomFacesData() {\n"
    my_str += "return "

    non_facey_list = [fnum for fnum in fnums_list if fnum not in set(predom_faces_list)]
    my_str += "["
    for i in range(0,min(len(non_facey_list),len(predom_faces_list))):
        if predom_faces_list[i] in fnum_to_url_dict and non_facey_list[i] in fnum_to_url_dict:
            my_str += "\t['%s', '%s'],\n"%(fnum_to_url_dict[predom_faces_list[i]],fnum_to_url_dict[non_facey_list[i]])
    my_str = my_str[:-2]+"]\n}\n"

    text_file = open("%s/data/react-codes/react_predom_faces_list_stuff_%s.txt"%(DATA_PATH), "w")
    text_file.write(my_str)
    text_file.close()
    print ("Done with React codes")
    
save_predom_faces_fnums_list()
make_react_code()
