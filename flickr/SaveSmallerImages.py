import pandas as pd
import os
from PIL import Image

#shorter size should be 600 px
# but bigger side should be less than 1000px
smaller_ims =set(os.listdir("data/images/smaller_images"))
directory = os.fsencode("data/images/")
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg") and filename not in smaller_ims:
        print (filename)
        # RESIZE IMAGE
        img = Image.open("data/images/%s"%filename).convert('RGB')
        if img.size[0] <= 1000 and img.size[1] <= 1000: #temp, because technically already did them, erase in future
            continue
        # adjust width and height to your needs
        if img.size[0] < img.size[1]:
            basewidth = 600
            wpercent = (basewidth/float(img.size[0]))
            hsize = int((float(img.size[1])*float(wpercent)))
            if hsize > 1000:
                shrinkage = 1000.0/hsize
                basewidth = int(shrinkage*600)
                hsize = 1000
            # use one of these filter options to resize the image
            img = img.resize((basewidth,hsize), Image.ANTIALIAS)
            img.save('data/images/smaller_images/%s'%filename)
        else:
            baseheight = 600
            hpercent = (baseheight/float(img.size[1]))
            wsize = int((float(img.size[1])*float(hpercent)))
            if wsize > 1000:
                shrinkage = 1000.0/wsize
                baseheight = int(shrinkage*600)
                wsize = 1000
            # use one of these filter options to resize the image
            img = img.resize((baseheight,wsize), Image.ANTIALIAS)
            img.save('data/images/smaller_images/%s'%filename)
    else:
        continue
