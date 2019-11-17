from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
import pandas as pd
import pickle
import os

try:
    DATA_PATH  = open("data_location.txt", "r").read().strip()
except:
    DATA_PATH = "."

SAVE_SPACE = True
df =  pd.read_csv("%s/data/url_title_and_file_data.csv"%DATA_PATH)
fnums_list = pickle.load(open("%s/data/basics/fnums_list.p"%DATA_PATH,"rb"))
finished_ims =set(os.listdir("%s/data/images/mask_rcnn_results"%DATA_PATH))

net = model_zoo.get_model('mask_rcnn_resnet50_v1b_coco', pretrained=True)
for fnum in fnums_list:
    if "%d.png"%fnum in finished_ims:
        print ("finished with %d previously"%fnum)
        continue
    print ("working on %d"%fnum)
    im_fname = "%s/data/images/smaller_images/%d.jpg"%(DATA_PATH,fnum)
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
        plt.savefig("%s/data/images/mask_rcnn_results/%d.png"%(DATA_PATH,fnum), bbox_inches = 'tight', pad_inches = 0)
        plt.clf()
        plt.close(fig)
    if SAVE_SPACE:
        orig_img = [] #set it to empty, no need for it
        bboxes = []
    res = [orig_img, masks, ids, scores, bboxes]
    pickle.dump(res,open("%s/data/images/mask_rcnn_results/res_%d.p"%(DATA_PATH,fnum),"wb"))
