from matplotlib import pyplot as plt
#from gluoncv import model_zoo, data, utils
import numpy as np
import pickle, os, cv2, math, boto3, time
from skimage import filters
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
    _,contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    M = cv2.moments(contours[0])
    if (M['m00'] == 0 or M['m00'] == 0):
        return 0,0
    print (round(M['m10'] / M['m00']), round(M['m01'] / M['m00']))
    return round(M['m10'] / M['m00']), round(M['m01'] / M['m00'])

def get_sample_of_image(fnum):
    # for that image
    im = cv2.imread("%s/data/images/smaller_images/%d.jpg"%(DATA_PATH,fnum))
    # take the first person
    try:
        res = pickle.load(open("%s/data/images/mask_rcnn_results/res_%d.p"%(DATA_PATH,fnum),"rb"))
    except:
        print ("no such mask rcnn result for fnum %d"%fnum)
        return None, None, None

    masks, ids = res[1], res[2]
    people_indices = [i for i in range(0,masks.shape[0]) if ids[i] == 0]
    if len(people_indices) == 0:
        print ("no people found")
        return None, None, None

    which_mask = 0
    while which_mask < len(people_indices):
        my_mask = masks[people_indices[which_mask]]
        center_x, center_y = get_center_of_mask(my_mask)
        if (center_y-35 < 0 or center_y +35 > im.shape[0] or center_x-35 < 0 or center_x +35 > im.shape[1]):
            #print ("The center that you got is too on the edge")
            if which_mask == len(people_indices) - 1:
                print ("too off center")
                return None, None, None
            else:
                which_mask += 1
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
            year_image = np.concatenate((year_image, sample_images[i]), axis=0)
        cv2.imwrite("%s/data/images/samples/year_%d.png"%(DATA_PATH,year), year_image)
        if upload_to_aws:
            filename = "%s/data/images/samples/year_%d.png"%(DATA_PATH,year)
            bucket_name = 'design-trends-bucket'
            objectname = "samples_from_year_%d.png"%(year)
            s3.upload_file(filename, bucket_name, objectname)

def make_react_code_for_years():
    # show as many columns
    my_str = ""
    my_str += "  yearly_samples_data: [\n"
    for year in range(1803,2020):
        # will have some non-existing years
        samples_url = "https://design-trends-bucket.s3.us-east-2.amazonaws.com/samples_from_year_%d.png"%year
        my_str += "['%s','%s'],\n"%(year,samples_url)
    my_str = my_str[:-2]+"\n"
    my_str += "],"

    text_file = open("%s/data/react-codes/react_for_samples_per_year_data.txt"%DATA_PATH, "w")
    text_file.write(my_str)
    text_file.close()

#return texture amount in whole image, return texture amount in sample
def identify_texture_amount_in_fnum_and_sample(fnum):
    sample_im, center_x, center_y = get_sample_of_image(fnum)
    im = cv2.imread("%s/data/images/smaller_images/%d.jpg"%(DATA_PATH,fnum))
    im_sobel = filters.sobel(im[..., 0]) + filters.sobel(im[..., 1]) + filters.sobel(im[..., 2])
    total_score = np.mean(im_sobel)
    if sample_im is not None:
        sample_score = np.mean(im_sobel[center_y-35:center_y+35, center_x-35:center_x+35])
    else:
        sample_score = None
    return total_score,sample_score

def save_texture_amounts_dicts():
    # fnum to texture amounts dict
    # save samples to texture amounts dict
    fnums_list = pickle.load(open("%s/data/basics/fnums_list.p"%DATA_PATH,"rb"))
    fnum_to_year_dict = pickle.load(open("%s/data/basics/fnum_to_year_dict.p"%DATA_PATH,"rb"))
    fnum_to_score_dict, fnum_sample_to_score_dict  = {}, {}
    year_to_score_sum_dict, year_to_total_num_dict  = {}, {}
    for fnum in fnums_list:
        total_score,sample_score = identify_texture_amount_in_fnum_and_sample(fnum)

        if sample_score != None:
            fnum_sample_to_score_dict[fnum] = sample_score

        fnum_to_score_dict[fnum] = total_score

        year = fnum_to_year_dict[fnum]
        if year not in year_to_score_sum_dict:
            year_to_score_sum_dict[year] = 0
            year_to_total_num_dict[year] = 0
        year_to_score_sum_dict[year]+=total_score
        year_to_total_num_dict[year]+=1

    pickle.dump(fnum_sample_to_score_dict, open("%s/data/fnum_sample_to_texture_score_dict.p"%DATA_PATH,"wb"))
    pickle.dump(fnum_to_score_dict, open("%s/data/fnum_to_texture_score_dict.p"%DATA_PATH,"wb"))

    year_to_avg_score_dict = {}
    for year in year_to_score_sum_dict:
        year_to_avg_score_dict[year] = year_to_score_sum_dict[year]/year_to_total_num_dict[year]
    pickle.dump(year_to_avg_score_dict, open("%s/data/year_to_avg_texture_score_dict.p"%DATA_PATH,"wb"))

def save_texture_amounts_samples(upload_to_aws = False):
    # load samples to_texture_amounts_dict
    fnum_sample_to_score_dict = pickle.load(open("%s/data/fnum_sample_to_texture_score_dict.p"%DATA_PATH,"rb"))
    # reorganize into texture amounts buckets to fnums list
    fnums = []
    sample_scores = []
    for fnum in fnum_sample_to_score_dict.keys():
        fnums.append(fnum)
        sample_scores.append(fnum_sample_to_score_dict[fnum])
    sample_scores,fnums = zip(*sorted(zip(sample_scores,fnums)))

    amount_in_each_bin = int(len(fnums)/20)
    bin_num = 0
    for i in range(0,len(fnums),amount_in_each_bin):
        all_bin_images = []
        for j in range(i,min(len(fnums),i+amount_in_each_bin)):
            bin_image,_,_ =  get_sample_of_image(fnums[j])
            if bin_image is not None:
                all_bin_images.append(bin_image)

        bin_image  = all_bin_images[0]
        for j in range(1,len(all_bin_images)):
            bin_image = np.concatenate((bin_image, all_bin_images[j]), axis=0)

        cv2.imwrite("%s/data/images/samples/texture_samples_bin_%d.png"%(DATA_PATH,bin_num), bin_image)

        if upload_to_aws:
            filename = "%s/data/images/samples/texture_samples_bin_%d.png"%(DATA_PATH,bin_num)
            bucket_name = 'design-trends-bucket'
            objectname = "samples_from_texture_amounts_bin_%d.png"%(bin_num)
            s3.upload_file(filename, bucket_name, objectname)
        bin_num += 1

print ("starting")
save_texture_amounts_dicts()
save_texture_amounts_samples(upload_to_aws=False)
#make_react_code_for_years()
#save_years_samples(upload_to_aws=True)
print ("ending")
