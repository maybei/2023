from data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools
from data import cfg, set_cfg, set_dataset
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from os import listdir
from os.path import isfile, join
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import datetime
import shutil

# filename = './data/slab/train/via_project_300_800_check_1.json'
# filename1 = './data/slab/train/train_old.json'
# 50,52,56,60,64
input_folder  = 'E:/H_slab/DS/12'
output_folder = 'E:/H_slab/DS_label/RAW'
perfolder_getcnt = 350

onlyfiles = [f for f in listdir(input_folder) if isfile(join(input_folder, f))]

totalcount = len(onlyfiles)
PITCH = int(totalcount / perfolder_getcnt)
cnt=0
copiedcnt=0

for src in Path(input_folder).glob('*'):

    cnt += 1
    if(cnt%PITCH!=0):
        continue
    copiedcnt +=1

    if(copiedcnt > perfolder_getcnt):
        break

    strsrc = str(src)
    dst = strsrc.replace("WS", "WS_label")
    shutil.copyfile(src, dst)

    print(dst)

    # print("--- %s seconds ---" % (time.time() - start_time))

    # time.sleep(3)
    # now = datetime.datetime.now()
    # print(now)