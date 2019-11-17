# save images to flickr data images,but  save a dataframe of image numbers, year, and link here, save title
from bs4 import BeautifulSoup
import requests, pickle

def save_decades to_pages_dict():
    decade_to_page_content_dict = {}
    for decade in range(1800,2000,10):
        page = requests.get("https://vintagefashionguild.org/fashion-timeline/%d-to-%d/"%(decade,decade+10))
        decade_to_page_content_dict[decade] = page.content
    pickle.dump()
#soup = BeautifulSoup(page.content, 'html.parser')


# LATER:
# save mask data using flickr code
# save monochrome together
# just add in usage of images to clustering and color charts automatically
# rerun color trends too
# make a double bar  chart showing how much data from each source (flickr vs vfg) on each year
