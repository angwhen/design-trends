# save images to flickr data images,but
from bs4 import BeautifulSoup
import requests, pickle
import pandas as pd
from os import listdir
from os.path import isfile, join

try:
    DATA_PATH  = open("data_location.txt", "r").read().strip()
except:
    DATA_PATH = "."
try:
    FLICKR_DATA_PATH  = open("../flickr/data_location.txt", "r").read().strip()
except:
    FLICKR_DATA_PATH = "../flickr"

def save_decades_to_pages_dict():
    decade_to_page_content_dict = {}
    for decade in range(1800,2000,10):
        page = requests.get("https://vintagefashionguild.org/fashion-timeline/%d-to-%d/"%(decade,decade+10))
        decade_to_page_content_dict[decade] = page.content
    pickle.dump(decade_to_page_content_dict,open("%s/data/decade_to_page_content_dict.p"%DATA_PATH,"wb"))

def save_images_to_flickr_location():
    decade_to_page_content_dict= pickle.load(open("%s/data/decade_to_page_content_dict.p"%DATA_PATH,"rb"))
    df_list = [] #  image num, title, year, url
    for decade in range(1800,1810,10):#,2000,10):
        page_content = decade_to_page_content_dict[decade]
        soup = BeautifulSoup(page_content, 'html.parser')
        images_list = soup.findAll('img')
        for image in images_list:
            if image["alt"] == 'Vintage Fashion Guild':
                continue


    #urls = pd.DataFrame(images_and_urls)
    #urls.columns = ["url","file_name"]
    # save a dataframe of image numbers, year, and link here, save title
    # also save a list of all fnums

#save_decades_to_pages_dict()
#save_images_to_flickr_location()
mypath = "%s/data/images/"%FLICKR_DATA_PATH
onlyfiles = [int(f.split(".")[0]) for f in listdir(mypath) if isfile(join(mypath, f))]
print (onlyfiles)
# LATER:
# save mask data using flickr code
# save monochrome together
# just add in usage of images to clustering and color charts automatically
# rerun color trends too
# make a double bar  chart showing how much data from each source (flickr vs vfg) on each year
