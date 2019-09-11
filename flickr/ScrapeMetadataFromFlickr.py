#!/usr/bin/env python3
from flickrapi import FlickrAPI
import pandas as pd
import sys
import re


def get_urls(image_tag,MAX_COUNT):
    api_key = open("flickr_api_key.txt").read().strip()
    api_secret = open("flickr_api_secret.txt").read().strip()
    flickr = FlickrAPI(api_key, api_secret)
    photos = flickr.walk(text=image_tag,
                            tag_mode='all',
                            tags=image_tag,
                            extras='url_o',
                            per_page=50,
                            sort='relevance')

    count=0
    urls=[]
    for photo in photos:
        curr_photo = []
        if count< MAX_COUNT:
            print("Fetching data for image number {}".format(count))
            try:
                url=photo.get('url_o')
                if url == None:
                    continue
                title= photo.get('title')
                possible_years = re.findall(r"[1|2][0|1|8|9][0-9][0-9]",title)
                if len(possible_years) == 0:
                    continue # no date
                year = possible_years[0]
                curr_photo = [url,title,year]
                urls.append(curr_photo)
                count +=1
            except:
                print("Data for image number {} could not be fetched".format(count))
        else:
            break

    urls=pd.DataFrame(urls)
    urls.columns = ["url","title","year"]
    urls.to_csv("data/"+image_tag+"_urls.csv")

keywords_file = open("data/keywords.txt","a")
for decade in range(1850,2030,10):
    name = "%ds fashion"%decade
    keywords_file.write(name+"\n")
    get_urls(name,10000)
keywords_file.close()
