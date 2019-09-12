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
                            per_page=500,
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
        # save before finishing in case breaks midway
        if count> 100 and count%100 == 0:
            urls_df=pd.DataFrame(urls)
            urls_df.columns = ["url","title","year"]
            urls_df.to_csv("data/"+image_tag+"_urls.csv")

    if len(urls) == 0:
        return False
    urls_df=pd.DataFrame(urls)
    urls_df.columns = ["url","title","year"]
    urls_df.to_csv("data/"+image_tag+"_urls.csv")
    return True

name = "woman"
keywords_file = open("data/keywords.txt","a")
res = get_urls(name,10000)
if res:
    keywords_file.write(name+"\n")
keywords_file.close()
