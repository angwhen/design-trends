from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
import pandas as pd
import pickle
import os

DATA_PATH = ""
try:
    f=open("data_location.txt", "r")
    DATA_PATH  = f.read().strip()
except:
    print ("data is right here")

SAVE_SPACE = True
df =  pd.read_csv("%s/data/url_title_and_file_data.csv"%DATA_PATH)
fnames_list = df[["file_name"]].values.tolist()
finished_ims =set(os.listdir("%s/data/images/mask_rcnn_results"%DATA_PATH))

net = model_zoo.get_model('mask_rcnn_resnet50_v1b_coco', pretrained=True)

for fname in fnames_list:
    fname_num = fname[0].split("/")[-1]
    fname_num = (int) (fname_num.split(".jpg")[0])
    if "%d.png"%fname_num in finished_ims:
        print ("finished with %d previously"%fname_num)
        continue
    print ("working on %d"%fname_num)
    im_fname = "%s/data/images/smaller_images/%d.jpg"%(DATA_PATH,fname_num)
    x, orig_img = data.transforms.presets.rcnn.load_test(im_fname)
    ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in net(x)]

    # paint segmentation mask on images directly
    width, height = orig_img.shape[1], orig_img.shape[0]
    masks = utils.viz.expand_mask(masks, bboxes, (width, height), scores)
    orig_img = utils.viz.plot_mask(orig_img, masks)

    # identical to Faster RCNN object detection
    if not SAVE_SPACE:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax = utils.viz.plot_bbox(orig_img, bboxes, scores, ids,
                             class_names=net.classes, ax=ax)
        plt.savefig("%s/data/images/mask_rcnn_results/%d.png"%(DATA_PATH,fname_num), bbox_inches = 'tight', pad_inches = 0)
        plt.clf()
        plt.close(fig)
    if SAVE_SPACE:
        orig_img = [] #set it to empty, no need for it
        bboxes = []
    res = [orig_img, masks, ids, scores, bboxes]
    pickle.dump(res,open("%s/data/images/mask_rcnn_results/res_%d.p"%(DATA_PATH,fname_num),"wb"))
