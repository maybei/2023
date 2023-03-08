import json
import os
import numpy as np


filename1 = './via/cocoFormat/train.json'
filename2 = './via/cocoFormat/val2.json'


with open(filename2, 'r') as f1:
    cocodata111 = json.load(f1)
    imgcnt = len(cocodata111['images'])
    for i1 in range(0, imgcnt):
        listcnt = len(cocodata111['annotations'][i1]['segmentation'])

        if(cocodata111['annotations'][i1]['area']<5):
            print('check area')

        print(listcnt  )

print('1111')