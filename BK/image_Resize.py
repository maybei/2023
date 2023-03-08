import os
import glob
# import time
# import tensorflow as tf
import numpy as np
# import unet as unet
import cv2


# image_height = 800
# image_width = 1600
# number_of_color_channels = 3
# color = (255,255,255)
# pixel_array = np.full((image_height, image_width, number_of_color_channels), color, dtype=np.uint8)
# cv2.imwrite("D:\WHITE_IMAGE_1600_800.jpg", pixel_array)

# 1600 * 800 white image create




for image_path in glob.glob('D:\\2022\\3.YOLACT\\data\\H_slab\\DS_label\\TRAIN_INDEXED\\*.jpg'):

    image1 = cv2.imread(image_path)

    origin = image_path.replace('TRAIN_INDEXED', 'TRAIN_INDEXED\\resize')

    #resize & save

    #cropped_img = image1[0: 640, 0: 640]
    dst = cv2.resize(image1, dsize=(640, 640), interpolation=cv2.INTER_AREA)

    cv2.imwrite(origin, dst)

    #cv2.waitKey(0)
    print(image_path )
