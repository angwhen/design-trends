import pandas as pd
import csv
from os import walk
import urllib.request

# save the images
# and also a dataframe from url to image name

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
urls.to_csv("data/images/url_to_image_file.csv")
