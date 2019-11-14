import pandas as pd
import pickle, os

try:
    DATA_PATH  = open("data_location.txt", "r").read().strip()
except:
    DATA_PATH = "."

def get_images_code_for_react_skinless():
    fnums_list = pickle.load(open("%s/data/basics/fnums_list.p"%DATA_PATH,"rb"))
    fnum_to_url_dict = pickle.load(open("%s/data/basics/fnum_to_flickr_url_dict.p"%DATA_PATH,"rb"))
    monochrome_list = set(pickle.load(open("%s/data/monochrome_list_%s.p"%(DATA_PATH,"hsv"),"rb")))
    already_uploaded_skinless = pickle.load(open("../aws/people_segmented_images_uploaded_to_aws_fnums_without_skin.p","rb"))

    my_str = ""
    my_str += "  images: [\n"
    for fnum in fnums_list:
        if not fnum in already_uploaded_skinless:
            continue
        if fnum in monochrome_list:
            continue
        url = fnum_to_url_dict[fnum]
        aws_seg_url = "https://design-trends-bucket.s3.us-east-2.amazonaws.com/people_seg_results_%d.png"%fnum
        aws_seg_without_skin_url = "https://design-trends-bucket.s3.us-east-2.amazonaws.com/people_seg_results_without_skin_%d.png"%fnum
        my_str += "['%s','%s','%s'],\n"%(url,aws_seg_url,aws_seg_without_skin_url)
    my_str = my_str[:-2]+"\n"
    my_str += "],"

    text_file = open("%s/data/react-codes/react_for_skinless.txt"%DATA_PATH, "w")
    text_file.write(my_str)
    text_file.close()

get_images_code_for_react_skinless()
