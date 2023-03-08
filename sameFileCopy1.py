import os
import glob
# import time
import numpy as np
import cv2
import shutil

print("TEST")
print("TEST")
print("TEST")
print("TEST")


# PATH = './output_images_H_21_10_27_TOP1/NG/*.jpg'
#
# for image_file in glob.glob(PATH):
#
#
#     origin = image_file.replace('output_images_H_21_10_27_TOP1/NG', 'output_images_H_21_10_27_TOP2')
#     target = image_file.replace('NG', 'NG2')
#     shutil.copyfile(origin, target)
#
#     print(origin )

    #
    # image1 = cv2.imread(image_path)
    #
    # image2 = cv2.imread(origin)
    #
    # img9 = np.zeros((256,256,1), dtype=np.uint8)
    #
    # img9 = image1[:, :, 1]
    #
    # # image = tf.image.decode_image(
    # #     tf.io.read_file(image_path),
    # #     channels=3,
    # #     dtype=tf.dtypes.uint8
    # # )
    # #
    # # # img = np.zeros((height, width, bpp), np.uint8)
    # # zeros_channel = np.zeros( (256,256,3), np.uint8)
    # cv2.imshow("predict", image1)
    # cv2.imshow("original", image2)
    #
    #
    # # 이진화
    # ret, img9 = cv2.threshold(img9, 50, 255, cv2.THRESH_BINARY)
    # cv2.imshow("threshold", img9)
    #
    # cv2.waitKey(0)
    # print(image_path )