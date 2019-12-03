from matplotlib import pyplot as plt
#from gluoncv import model_zoo, data, utils
import numpy as np
import pickle, os, cv2, math, boto3
# make tiny 70x70 px crop images of the center of people
# string them together vertically by year
# to make height: 700, width 70 images
# upload one of those for each year

# also make texture trends
# make metric for texture
# for each 70x70 crop image give it a number
# make a col for lowest to highest (make 20 cols) texture amounts
# to indicate how well it works
# upload  those 20 700x70 images too

# using that metric show how  texture varies over time in the whole people image
try:
    DATA_PATH  = open("data_location.txt", "r").read().strip()
except:
    DATA_PATH = "."

s3 = boto3.client('s3')

def get_center_of_mask(mask):
    # https://answers.opencv.org/question/204175/how-to-get-boundry-and-center-information-of-a-mask/
    im, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    M = cv2.moments(contours[0])
    if (M['m00'] == 0 or M['m00']):
        return 0,0
    return round(M['m10'] / M['m00']), round(M['m01'] / M['m00'])

def get_sample_of_image(fnum):
    # for that image
    im = cv2.imread("%s/data/images/smaller_images/%d.jpg"%(DATA_PATH,fnum))
    # take the first person
    try:
        res = pickle.load(open("%s/data/images/mask_rcnn_results/res_%d.p"%(DATA_PATH,fnum),"rb"))
    except:
        #print ("no such mask rcnn result for fnum %d"%fnum)
        return None, None, None

    masks, ids = res[1], res[2]
    people_indices = [i for i in range(0,masks.shape[0]) if ids[i] == 0]
    if len(people_indices) == 0:
        return None, None, None

    which_mask = 0
    while which_mask < len(people_indices):
        my_mask = masks[people_indices[which_mask]]
        center_x, center_y = get_center_of_mask(my_mask)
        if (center_y-35 < 0 or center_y +35 > im.shape[0] or center_x-35 < 0 or center_x +35 > im.shape[1]):
            #print ("The center that you got is too on the edge")
            if which_mask == len(people_indices) - 1:
                return None, None, None
        else:
            break

    sample_im = im[center_y-35:center_y+35, center_x-35:center_x+35]
    cv2.imwrite("%s/data/images/samples/%d.png"%(DATA_PATH,fnum), sample_im)
    return sample_im, center_x, center_y

def save_years_samples(upload_to_aws = False):
    # make the 700x70 image with fnums of that year
    # if not enough fnums, just make the image shorter
    year_to_fnums_dict = pickle.load(open("%s/data/basics/year_to_fnums_dict.p"%(DATA_PATH),"rb"))
    for year in year_to_fnums_dict.keys():
        fnums = year_to_fnums_dict[year]
        sample_images = []
        for fnum in fnums:
            sample_im, _, _  = get_sample_of_image(fnum)
            if sample_im is not None:
                sample_images.append(sample_im)
        if len(sample_images) == 0:
            print ("no valid sample images for year ", year)
            continue
        year_image = sample_images[0]
        for i in range(1,min(10,len(sample_images))):
            year_image= np.concatenate((year_image, sample_images[i]), axis=0)
        cv2.imwrite("%s/data/images/samples/year_%d.png"%(DATA_PATH,year), year_image)

        filename = "%s/data/images/samples/year_%d.png"%(DATA_PATH,year)
        bucket_name = 'design-trends-bucket'
        objectname = "samples_from_year_%d.png"%(year)
        s3.upload_file(filename, bucket_name, objectname)

#return texture amount in whole image, return texture amount in sample
def identify_texture_amount_in_fnum_and_sample(fnum):
    sample_im, center_x, center_y = get_sample_of_image(fnum)
    # filter whole image
    # check out fnum
    return None, None

def save_texture_amounts_dict(fnum):
    # fnum to texture amounts dict
    # save samples to texture amounts dict
    fnums_list = pickle.load(open("%s/data/basics/fnums_list.p"%DATA_PATH,"rb"))

    # year to average texture amounts dict
    year_to_fnums_dict = pickle.load(open("%s/data/basics/year_to_fnums_dict.p"%(DATA_PATH),"rb"))

def save_texture_amounts_samples(upload_to_aws = False):
    # load samples to_texture_amounts_dict
    # reorganize into texture amounts buckets to fnums list
    # save images for those

    return None

save_years_samples()

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
    print ("DONE")
