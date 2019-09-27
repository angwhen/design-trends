import pandas as pd
import pickle
import os
import boto3
import os

# script to upload all segmented_images with proper date to aws

def contains_person(fname_num):
    mask_file_name ="../flickr/data/images/mask_rcnn_results/res_%d.p"%fname_num
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
    if not hasPerson:
        return False
    return True

def upload_segmented_images_to_aws():
    s3 = boto3.client('s3')
    df =  pd.read_csv("../flickr/data/url_title_and_file_data.csv")
    my_list = df[["url","year","file_name"]].values.tolist()
    finished_peopled_ims =set(os.listdir("../flickr/data/images/mask_rcnn_results/people_seg_images/"))
    try:
        already_uploaded = pickle.load(open("people_segmented_images_uploaded_to_aws_fnums.p","rb"))
    except:
        already_uploaded = set([])
    for im in my_list:
        fname_num = im[2].split("/")[-1]
        fname_num = (int) (fname_num.split(".jpg")[0])

        if contains_person(fname_num) and not fname_num in already_uploaded and "%d.png"%fname_num in finished_peopled_ims:
            # want to change to get the person segmented only version
            filename = "../flickr/data/images/mask_rcnn_results/people_seg_images/%d.png"%fname_num
            bucket_name = 'design-trends-bucket'
            objectname = "people_seg_results_%d.png"%fname_num
            print (objectname)

            s3.upload_file(filename, bucket_name, objectname)
            already_uploaded.add(fname_num)
            pickle.dump(already_uploaded,open("people_segmented_images_uploaded_to_aws_fnums.p","wb")) # save freq in case stuff breaks

upload_segmented_images_to_aws()
