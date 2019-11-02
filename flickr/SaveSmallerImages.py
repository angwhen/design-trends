import pandas as pd
import os
from PIL import Image

DATA_PATH = "."
try:
    f=open("data_location.txt", "r")
    DATA_PATH  = f.read().strip()
except:
    print ("data is right here")

#shorter size should be 600 px
# but bigger side should be less than 1000px
smaller_ims =set(os.listdir("%s/data/images/smaller_images"%DATA_PATH))
directory = os.fsencode("%s/data/images/smaller_images"%DATA_PATH) #"%s/data/images/"%DATA_PATH)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"): # and filename not in smaller_ims: #ADD BACK LATER
        print (filename)
        # RESIZE IMAGE
        img = Image.open("%s/data/images/smaller_images/%s"%(DATA_PATH,filename)).convert('RGB')#Image.open("%s/data/images/%s"%(DATA_PATH,filename)).convert('RGB')
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
            img.save('%s/data/images/smaller_images/%s'%(DATA_PATH,filename))
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
            img.save('%s/data/images/smaller_images/%s'%(DATA_PATH,filename))
    else:
        continue

print ("Done resizing")
