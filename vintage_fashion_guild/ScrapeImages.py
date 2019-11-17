# save images to flickr data images,but
from bs4 import BeautifulSoup
import requests, pickle, re, shutil
import pandas as pd
from os import listdir
from os.path import isfile, join
import urllib.request

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

    mypath = "%s/data/images/"%FLICKR_DATA_PATH
    existing_images_nums = [int(f.split(".")[0]) for f in listdir(mypath) if isfile(join(mypath, f)) if f[-3:] == 'jpg']
    max_num = max(existing_images_nums)
    curr_num = max_num +1

    df_list = [] #  image num, title, year, url
    all_fnums = []
    year_to_fnums_dict = {}
    fnum_to_year_dict = {}
    fnum_to_url_dict = {}
    for decade in range(1810,1820,10):#2000,10):
        page_content = decade_to_page_content_dict[decade]
        soup = BeautifulSoup(page_content, 'html.parser')
        images_list = soup.findAll('img')
        for image in images_list:
            if image["alt"] == 'Vintage Fashion Guild':
                continue
            try:
                title = image["alt"]
                print ("title:%s"%title)
                url = "https://vintagefashionguild.org%s"%image["src"]
                # extract year
                possible_years = re.findall(r"[1|2][0|1|8|9][0-9][0-9]",title)
                if len(possible_years) == 0:
                    print ("no year")
                    continue # no date
                extracted_year = possible_years[0]
                # download image
                print ("downloading")
                fname = "%s/data/images/%d.jpg"%(FLICKR_DATA_PATH,curr_num)
                #urllib.request.urlretrieve(url,fname)
                r = requests.get(url, stream=True)
                with open(fname,"wb") as f:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw,f)
                df_list.append([curr_num,title,extracted_year,url])
                all_fnums.append(curr_num)
                if extracted_year not in year_to_fnums_dict:
                    year_to_fnums_dict[extracted_year] = []
                year_to_fnums_dict[extracted_year].append(curr_num)
                fnum_to_year_dict[curr_num] = extracted_year
                fnum_to_url_dict[curr_num] = url
                curr_num +=1
            except:
                curr_num +=0
                url = "https://vintagefashionguild.org%s"%image["src"]
                print ("couldnt download %s to %s/data/images/%d.jpg"%(url,FLICKR_DATA_PATH,curr_num) )
    pickle.dump(df_list,open("%s/data/temp.p"%DATA_PATH,"wb")) #saving in case something goes wrong

    df = pd.DataFrame(df_list)
    df.columns = ["fnum","title","year","url"]
    df.to_csv("%s/data/vintage_fashion_guild_images_df.csv"%DATA_PATH)
    # save a dataframe of image numbers, year, and link here, save title
    # also save a list of all fnums
    pickle.dump(all_fnums,open("%s/data/vfg_fnums_list.p"%DATA_PATH,"wb"))
    pickle.dump(year_to_fnums_dict,open("%s/data/vfg_year_to_fnums_dict.p"%DATA_PATH,"wb"))
    pickle.dump(fnum_to_year_dict,open("%s/data/vfg_fnum_to_year_dict.p"%DATA_PATH,"wb"))
    pickle.dump(fnum_to_url_dict,open("%s/data/vfg_fnum_to_url_dict.p"%DATA_PATH,"wb"))

save_decades_to_pages_dict()
save_images_to_flickr_location()


# LATER:
# run save smaller images, no code needs to be changed
#rerun make basic dicts , run everythign else directly
# save mask data using flickr code
# save monochrome together
# rerun color trends too directly
# make a double bar  chart showing how much data from each source (flickr vs vfg) on each year
