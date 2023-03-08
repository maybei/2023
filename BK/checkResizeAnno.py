import os
import glob
# import time
import numpy as np
import cv2
import json

# newfilename = './data/H_slab/WS_VALID_100.json'
train_info = './data/H_slab/WS/train/WS_TRAIN_500.json'
# train_info = 'D:\\2022\\3.YOLACT\\data\\H_slab\\WS\\train\\WS_TRAIN_100.json'


train_images = './data/H_slab/WS/train/images/*.jpg'
valid_info = './data/H_slab/WS/train/WS_TRAIN_500.json'
valid_images = './data/H_slab/WS/train/images/'

# with open(train_info, 'r') as f1:
# with open(train_info, encoding = 'utf8') as f1:
with open(str(train_info)) as f1:
    cocodata = json.load(f1)
    NCocoData = cocodata.copy()


pos = 0
for image_path in glob.glob(train_images):


    img = cv2.imread(image_path)

    txy_list = []
    xy_list = []
    txy_list = cocodata['annotations'][pos]['segmentation'][0]
    count = int(len(cocodata['annotations'][pos]['segmentation'][0]) / 2)

    pos1 = 0
    for data in range(count):
        pos1 = data * 2
        xy = [int(txy_list[pos1]), int(txy_list[pos1+1])]
        xy_list.append(xy)

    pts = xy_list


    mask = np.zeros(img.shape, np.uint8)
    points = np.array(pts, np.int32)
    points = points.reshape((-1, 1, 2))
    #
    mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
    mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255))  # for ROI
    mask3 = cv2.fillPoly(mask.copy(), [points], (0, 255, 0))  # for displaying images on the desktop

    show_image = cv2.addWeighted(src1=img, alpha=0.8, src2=mask3, beta=0.2, gamma=0)
    ROI = cv2.bitwise_and(mask2, img)

    pos +=1
    cv2.imshow("original", img)
    # cv2.imshow("mask2", mask2)
    cv2.imshow("mask3", mask3)
    cv2.waitKey(0)
    # cv2.imwrite(os.path.join( edit_path, image_file_names[x]), ROI)
    # print(image_file_names[x])


    #resize & save

    #cropped_img = image1[0: 640, 0: 640]
    # dst = cv2.resize(image1, dsize=(640, 640), interpolation=cv2.INTER_AREA)
    #
    # cv2.imwrite(origin, dst)

    #cv2.waitKey(0)
    print(image_path )
