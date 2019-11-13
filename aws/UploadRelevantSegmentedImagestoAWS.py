import pandas as pd
import pickle
import os
import boto3
import os

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

def upload_segmented_images_to_aws():
    s3 = boto3.client('s3')
    df =  pd.read_csv("%s/data/url_title_and_file_data.csv"%DATA_PATH)
    my_list = df[["url","year","file_name"]].values.tolist()
    finished_peopled_ims =set(os.listdir("%s/data/images/mask_rcnn_results/people_seg_images/"%DATA_PATH))
    try:
        already_uploaded = pickle.load(open("people_segmented_images_uploaded_to_aws_fnums.p","rb"))
    except:
        already_uploaded = set([])
    for im in my_list:
        fnum = im[2].split("/")[-1]
        fnum = (int) (fnum.split(".jpg")[0])

        if contains_person(fnum) and not fnum in already_uploaded and "%d.png"%fnum in finished_peopled_ims:
            # want to change to get the person segmented only version
            filename = "%s/data/images/mask_rcnn_results/people_seg_images/%d.png"%(DATA_PATH,fnum)
            bucket_name = 'design-trends-bucket'
            objectname = "people_seg_results_%d.png"%fnum
            print (objectname)

            s3.upload_file(filename, bucket_name, objectname)
            already_uploaded.add(fnum)
            pickle.dump(already_uploaded,open("people_segmented_images_uploaded_to_aws_fnums.p","wb")) # save freq in case stuff breaks

upload_segmented_images_to_aws()
