import pandas as pd
import pickle
import os
import boto3
import csv

# script to upload all segmented_images with proper date to aws
try:
    DATA_PATH  = open("../flickr/data_location.txt", "r").read().strip()
except:
    DATA_PATH = "../flickr"

def contains_person(fnum):
    mask_file_name ="%s/data/images/mask_rcnn_results/res_%d.p"%(DATA_PATH,fnum)
    if not os.path.exists(mask_file_name):
        return False
    res = pickle.load(open(mask_file_name,"rb"))
    masks = res[1]
    ids = res[2]
    hasPerson = False
    for i in range(0,masks.shape[0]): #the masks we have for people
        if ids[i] == 0:
            hasPerson = True
            break
    return  hasPerson

def upload_segmented_images_to_aws(skinless=False):
    s3 = boto3.client('s3')
    #df =  pd.read_csv("%s/data/url_title_and_file_data.csv"%DATA_PATH)
    #my_list = df[["url","year","file_name"]].values.tolist()

    fnums_list = pickle.load(open("%s/data/basics/fnums_list.p"%DATA_PATH,"rb"))

    skinless_str = ""
    if skinless:
        skinless_str = "_without_skin"
    finished_peopled_ims =set(os.listdir("%s/data/images/mask_rcnn_results/people_seg_images%s/"%(DATA_PATH,skinless_str)))
    #try:
    #    already_uploaded = pickle.load(open("people_segmented_images_uploaded_to_aws_fnums%s.p"%skinless_str,"rb"))
    #except:
    #    already_uploaded = set([])
    already_uploaded = set([])
    #for im in my_list:
    #    fnum = im[2].split("/")[-1]
    #    fnum = (int) (fnum.split(".jpg")[0])
    for fnum in fnums_list:
        if contains_person(fnum) and not fnum in already_uploaded and "%d.png"%fnum in finished_peopled_ims:
            # want to change to get the person segmented only version
            filename = "%s/data/images/mask_rcnn_results/people_seg_images%s/%d.png"%(DATA_PATH,skinless_str,fnum)
            bucket_name = 'design-trends-bucket'
            objectname = "people_seg_results%s_%d.png"%(skinless_str,fnum)
            print (objectname)

            s3.upload_file(filename, bucket_name, objectname)
            already_uploaded.add(fnum)
            pickle.dump(already_uploaded,open("people_segmented_images_uploaded_to_aws_fnums%s.p"%skinless_str,"wb")) # save freq in case stuff breaks


upload_segmented_images_to_aws(skinless=True)


"""with open("credentials.csv") as csv_file:
    csv_reader = csv.reader(csv_file,delimiter=",")
    line_count = 0
    for row in csv_reader:
        if line_count == 1:
            ACCESS_ID = row[2]
            ACCESS_KEY = row[3]
        line_count +=1"""
