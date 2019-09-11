import pandas as pd
import csv
from os import walk
import re
import urllib.request

# save the images
# and also a dataframe from url to image name
def later_additions():
    images_and_urls_so_far = pd.read_csv("data/images/url_to_image_file.csv")
    urls_so_far = set([])
    max_image = 0
    for i,row in images_and_urls_so_far.iterrows():
        urls_so_far.add(row['url'])
        fname_num= re.findall(r'\d+', row["file_name"])[0]
        max_image = max(max_image,int(fname_num))

    print (urls_so_far)
    print (max_image)

    data = None
    for keyword in open("data/keywords.txt"):
        keyword = keyword.strip()
        if data is None:
            data = pd.read_csv("data/%s_urls.csv"%keyword)
        else:
            data = data.append(pd.read_csv("data/%s_urls.csv"%keyword))


    images_and_urls = []
    count = max_image+1
    for i,row in data.iterrows():
        url = row['url']
        if url in urls_so_far: # do not repeat
            continue
        print ("%s %d"%(url,count))
        try:
            fname = "data/images/%d.jpg"%count
            urllib.request.urlretrieve(url,fname)
            images_and_urls.append([url,fname])
            count +=1
        except:
            count +=0

        #save data as we go in case the program fails before reaching the end
        pickle.dump(images_and_urls,open("data/images/url_to_image_file_temp.p","wb"))

    urls = pd.DataFrame(images_and_urls)
    urls.columns = ["url","file_name"]
    urls = urls.append(images_and_urls_so_far[["url","file_name"]])
    urls.to_csv("data/images/url_to_image_file.csv")


def first_time():
    data = None
    for keyword in open("data/keywords.txt"):
        keyword = keyword.strip()
        if data is None:
            data = pd.read_csv("data/%s_urls.csv"%keyword)
        else:
            data = data.append(pd.read_csv("data/%s_urls.csv"%keyword))


    images_and_urls = []
    count = 0
    for i,row in data.iterrows():
        url = row['url']
        print ("%s %d"%(url,count))
        try:
            fname = "data/images/%d.jpg"%count
            urllib.request.urlretrieve(url,fname)
            images_and_urls.append([url,fname])
            count +=1
        except:
            count +=0
    urls = pd.DataFrame(images_and_urls)
    urls.columns = ["url","file_name"]
    urls.to_csv("data/images/url_to_image_file.csv")

later_additions()
