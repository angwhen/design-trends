import pandas as pd
import os
from PIL import Image

directory = os.fsencode("data/images/")
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"):
        # RESIZE IMAGE
        img = Image.open("data/images/%s"%filename)
        # adjust width and height to your needs
        if img.size[0] < img.size[1]:
            basewidth = 600
            wpercent = (basewidth/float(img.size[0]))
            hsize = int((float(img.size[1])*float(wpercent)))
            # use one of these filter options to resize the image
            img = img.resize((basewidth,hsize), Image.ANTIALIAS)
            img.save('data/images/smaller_images/%s'%filename)
        else:
            baseheight = 600
            hpercent = (baseheight/float(img.size[1]))
            wsize = int((float(img.size[1])*float(hpercent)))
            # use one of these filter options to resize the image
            img = img.resize((baseheight,wsize), Image.ANTIALIAS)
            img.save('data/images/smaller_images/%s'%filename)
    else:
        continue
