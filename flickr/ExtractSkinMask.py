from matplotlib import pyplot as plt
#from gluoncv import model_zoo, data, utils
import pandas as pd
import numpy as np
import pickle
import os, cv2, math
#from jeanCVModified import skinDetector
#import cv2

try:
    DATA_PATH  = open("data_location.txt", "r").read().strip()
except:
    DATA_PATH = "."

def get_people_cutout(fnum):
    try:
        res = pickle.load(open("%s/data/images/mask_rcnn_results/res_%d.p"%(DATA_PATH,fnum),"rb"))
    except:
        return None
    masks, ids,scores= res[1], res[2], res[3]
    print (scores[:4])
    people_indices = [i for i in range(0,masks.shape[0]) if ids[i] == 0]
    if len(people_indices) == 0:
        return None

    im = cv2.imread("%s/data/images/smaller_images/%d.jpg"%(DATA_PATH,fnum))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if (im.shape[0] != masks.shape[1] or im.shape[1] != masks.shape[2]):
        print ("Dimensional problem on %d, image:%d, %d vs masks: %d, %d"%(fnum, im.shape[0],im.shape[1],masks.shape[1],masks.shape[2]))
        return None

    people_img = masks[people_indices[0]]
    for i in range(1,len(people_indices)):
        if scores[i] < 0.75:
            break
        people_img += masks[people_indices[i]]
    #im[people_img == 0] = [0, 0, 0]
    #return im
    print (len(people_indices))
    return people_img

#https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
def get_face_histograms_and_cutouts(fnum):#size of image, 0 where no face, 1 where is face
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread("%s/data/images/smaller_images/%d.jpg"%(DATA_PATH,fnum))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # face cascade needs grayscale
    faces = face_cascade.detectMultiScale(gray, 1.1, 10)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    histograms = []
    faces_cutout = np.zeros([img.shape[0],img.shape[1]],dtype=np.uint8)
    for (x, y, w, h) in faces:
        faces_cutout[y:y+h, x:x+w]= 1
        crop_face_before_blur = img[y+int(h/6):y+5*int(h/6), x+int(w/6):x+5*int(w/6)]  # crop abit more tightly
        crop_face_blurred = cv2.blur(crop_face_before_blur,(int(w/4),int(h/4)))

        face_features = crop_face_before_blur-crop_face_blurred
        face_features = cv2.cvtColor(face_features,cv2.COLOR_HSV2BGR)

        face_features_gray = cv2.cvtColor(face_features, cv2.COLOR_BGR2GRAY)
        #print (face_features_gray.shape)
        new = np.ones([face_features.shape[0],face_features.shape[1]],dtype=np.uint8)
        new[np.where(face_features_gray >20)] = 0

        crop_face_temp = cv2.cvtColor(crop_face_before_blur,cv2.COLOR_HSV2BGR)

        histr = cv2.calcHist([crop_face_blurred],[0,1], mask=new, histSize=[256, 256], ranges=[0, 256, 0, 256] )
        histograms.append(histr)

    return histograms,faces_cutout

#https://www.learnopencv.com/blob-detection-using-opencv-python-c/
#https://stackoverflow.com/questions/30369031/remove-spurious-small-islands-of-noise-in-an-image-python-opencv
def remove_small_blobs(mask):
    from skimage import morphology
    import skimage
    small_piece = mask.shape[0]*mask.shape[1]/600
    processed = morphology.remove_small_objects(mask.astype(bool), min_size=small_piece, connectivity=1).astype(np.uint8)
    return processed

# 1) https://nalinc.github.io/blog/2018/skin-detection-python-opencv/
# 1) tends to get much more pixels than is actually skin
# 2) #https://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/
# 2) Using the zebra stuff from this for the histogram
def convolve(B, r):
    D = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r,r))
    cv2.filter2D(B, -1, D, B)
    return B
def get_skin_cutout(fnum):
    im = cv2.imread("%s/data/images/smaller_images/%d.jpg"%(DATA_PATH,fnum))
    im = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    #print (im.shape)
    skin_cutout = np.zeros((im.shape[0],im.shape[1]),dtype=np.uint8)
    histograms,faces_cutout = get_face_histograms_and_cutouts(fnum)
    if len(histograms) == 0:
        return None
    for histr in histograms:
        B = cv2.calcBackProject([im], channels=[0,1], hist=histr, ranges=[0,256,0,256], scale=1)
        B = convolve(B, r=5)
        #print (B[100])
        skin_cutout += B

    skin_cutout[skin_cutout > 10] = 1
    textured_mask = get_textured_mask(fnum)
    #return faces_cutout
    skin_cutout  = cv2.bitwise_and(skin_cutout,skin_cutout, mask =textured_mask) #removing overly textured stuff from what is considered skin
    #skin_cutout  += faces_cutout
    skin_cutout[skin_cutout > 0] = 1
    #return skin_cutout
    return remove_small_blobs(skin_cutout)
def get_skin_mask(fnum):
    skin_cutout = get_skin_cutout(fnum)
    if skin_cutout is None:
        return None
    return  1-skin_cutout

#https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	return edged

def get_textured_mask(fnum):
    im = cv2.imread("%s/data/images/smaller_images/%d.jpg"%(DATA_PATH,fnum))
    textured_outlines = auto_canny(im,0.01)
    kernel = np.ones((2,2), np.uint8)
    textured_outlines = cv2.dilate(textured_outlines, kernel, iterations=1)

    textured_mask_starter = cv2.blur(textured_outlines,(10,10))
    #cv2.imshow("okay",textured_mask_starter)
    #cv2.waitKey(0)
    textured_mask = np.ones([textured_mask_starter.shape[0],textured_mask_starter.shape[1]],dtype=np.uint8)
    textured_mask[np.where(textured_mask_starter >10)] = 0

    return textured_mask #mask with textured areas zeroed out

def save_skin_masks_and_deskinned_people_images():
    fnums_list = pickle.load(open("%s/data/basics/fnums_list.p"%DATA_PATH,"rb"))
    finished_fnums =set(os.listdir("%s/data/images/mask_rcnn_results/people_seg_images_without_skin/"%(DATA_PATH)))
    for fnum in fnums_list:
        if fnum in finished_fnums:
            continue
        print ("working on %d"%fnum)
        skin_mask = get_skin_mask(fnum)
        if skin_mask is None:
            print ("no face...")
            continue
        people_cutout =  get_people_cutout(fnum)
        if people_cutout is None:
            continue
        pickle.dump(skin_mask,open("%s/data/images/mask_rcnn_results/skin_masks/%d.png"%(DATA_PATH,fnum),"wb"))

        #save image with skin darked out
        im = cv2.imread("%s/data/images/smaller_images/%d.jpg"%(DATA_PATH,fnum))
        people_without_skin_cutout = cv2.bitwise_and(people_cutout,skin_mask)
        sub = np.true_divide(im,5)
        im = im - sub
        im[people_without_skin_cutout == 0] = [0, 0, 0]
        im  = im + sub
        cv2.imwrite("%s/data/images/mask_rcnn_results/people_seg_images_without_skin/%d.png"%(DATA_PATH,fnum), im)
    print ("Done")

def magic_wand(fnum, small_skin_mask,people_cutout):
    #small_skin_cutout = 1- small_skin_mask
    if (small_skin_mask) is None:
        print ("NO SKIN MASK?")
        return None
    im = cv2.imread("%s/data/images/smaller_images/%d.jpg"%(DATA_PATH,fnum))
    #im = cv2.imread("seagull.jpg") #https://stackoverflow.com/questions/16705721/opencv-floodfill-with-mask
    from skimage.segmentation import flood, flood_fill
    from skimage import data, filters
    small_skin_mask_orig = small_skin_mask
    #small_skin_mask  = small_skin_mask
    hists, _ = get_face_histograms_and_cutouts(fnum)
    B = None
    hists_sum = None
    for histr in hists:
        Bcurr = cv2.calcBackProject([im], channels=[0,1], hist=histr, ranges=[0,256,0,256], scale=1)
        Bcurr = convolve(Bcurr, r=50)
        #print (Bcurr)
        if B is None:
            B = Bcurr
            hists_sum = histr
        else:
            B = cv2.bitwise_or(B,Bcurr)
            hists_sum += histr
        #cv2.imshow("Back",255*Bcurr)
        #cv2.waitKey(0)
    #if B is None:
    #    Bsum = 0
    #else:
    #    Bsum = len(B[B != 0])

    kernel = np.ones((10,10), np.uint8)
    #small_skin_mask = cv2.dilate((1-small_skin_mask), kernel, iterations=1)
    small_skin_mask = (small_skin_mask)
    #cv2.imshow("skin mask",small_skin_mask*255)
    #cv2.waitKey(0)
    if len(small_skin_mask[small_skin_mask==1]) == 0 or len(small_skin_mask[small_skin_mask==0]) == 0:
        print ("NO", fnum)
        return small_skin_mask_orig
    #cv2.imshow("skin",small_skin_mask*255)
    #cv2.waitKey(0)

    connectivity = 4
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(small_skin_mask , connectivity , cv2.CV_32S)
    print (centroids)

    im_sobel = filters.sobel(im[..., 0]) + filters.sobel(im[..., 1]) + filters.sobel(im[..., 2])
    im_sobel = cv2.dilate(im_sobel,np.ones((10,10), np.uint8), iterations=1)

    #cv2.imshow("people cutout",people_cutout*255)
    #cv2.waitKey(0)
    #loose_people_cutout = cv2.dilate(people_cutout,kernel,iterations = 1)
    #cv2.imshow("people_cutout",loose_people_cutout*255)
    #cv2.waitKey(0)
    skin_mask_total = None
    for i in range(0,centroids.shape[0]):
        x = int(centroids[i][0])#keypoints[i].pt[0] #i is the index of the blob you want to get the position
        y = int(centroids[i][1])#keypoints[i].pt[1]

        if people_cutout[y][x] == False:
            """fig, ax = plt.subplots(nrows=2, figsize=(10, 20))
            ax[0].imshow(im)
            ax[0].plot(int(x),int(y), 'wo',color='blue')  # seed point
            ax[0].set_title('spot not in person')
            ax[0].axis('off')

            fig.tight_layout()
            plt.show()"""
            print ("PEOPLE",people_cutout[y][x] )
            continue

        print (stats[i])
        if small_skin_mask_orig[y][x] != 0:
            print ( small_skin_mask_orig[y][x])
            #continue

        im_skin_curr = flood(im_sobel, (int(y),int(x)), tolerance=0.05).astype(np.uint8)

        # color correctness
        im_temp =  cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        histr = cv2.calcHist([im_temp],[0,1], mask=cv2.bitwise_and(1-im_skin_curr,people_cutout), histSize=[256, 256], ranges=[0, 256, 0, 256] )
        correl = cv2.compareHist(histr, hists_sum,cv2.HISTCMP_CORREL)
        print ("COMPARISON",correl)
        if correl < 0.02:
            cv2.imshow("bad color match part",im_skin_curr*255)
            cv2.waitKey(0)
            continue

        if skin_mask_total is None:
            skin_mask_total = im_skin_curr
        else:
            skin_mask_total =cv2.bitwise_or(im_skin_curr,skin_mask_total)

        print (x,y)
        """fig, ax = plt.subplots(nrows=3, figsize=(10, 20))

        ax[0].imshow(im)
        ax[0].set_title('Original')
        ax[0].axis('off')

        ax[1].imshow(im_sobel)
        ax[1].set_title('Sobel filtered')
        ax[1].axis('off')

        ax[2].imshow(im)
        ax[2].imshow(skin_mask_total,  alpha=0.9)
        ax[2].plot(int(x),int(y), 'wo',color='blue')  # seed point
        ax[2].set_title('Segmented skin part`')
        ax[2].axis('off')

        fig.tight_layout()
        plt.show()"""
    if skin_mask_total is None:
        return small_skin_mask_orig
    return cv2.bitwise_and(1-skin_mask_total,small_skin_mask_orig)
#save_skin_masks_and_deskinned_people_images()
#im = get_image_with_non_people_blacked_out(5)
#get_face_histograms_and_cutouts(154)
fnum = 27#136#1649#14#1649 #26 27,,5, 154, 136
im = cv2.imread("%s/data/images/smaller_images/%d.jpg"%(DATA_PATH,fnum))

for fnum in [9,99,987,196,200,2,1,27,136,154,1649]:
    skin_mask = get_skin_mask(fnum)
    people_cutout =  get_people_cutout(fnum)
    skin_mask2 = magic_wand(fnum, skin_mask,people_cutout)
    im = cv2.imread("%s/data/images/smaller_images/%d.jpg"%(DATA_PATH,fnum))

    new_im =cv2.bitwise_and(im,im, mask = people_cutout)
    new_im =cv2.bitwise_and(new_im,new_im, mask = skin_mask)
    cv2.imshow("new image1",new_im)
    cv2.waitKey(0)
    new_im =cv2.bitwise_and(im,im, mask = people_cutout)
    new_im =cv2.bitwise_and(new_im,new_im, mask = skin_mask2)
    cv2.imshow("new image2",new_im)
    cv2.waitKey(0)

#new_im =cv2.bitwise_and(im,im, mask = cv2.bitwise_and(skin_mask_very_general,people_cutout))
#cv2.imshow("new image2",new_im)
#cv2.waitKey(0)

"""
image = cv2.imread("%s/data/images/smaller_images/%d.jpg"%(DATA_PATH,fnum))
detector = skinDetector(image,get_people_cutout(fnum))
skin_cutout = detector.find_skin()
new_im =cv2.bitwise_and(image,image, mask = people_cutout)
new_im =cv2.bitwise_and(new_im,new_im, mask = skin_cutout)
cv2.imshow("new image2",new_im)
cv2.waitKey(0)"""
