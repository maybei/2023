import json
import os
import numpy as np


'''
filename = 'D:/Project/2021_H/IMAGE/FROM_CHERRY/via_project_300_800_check_1.json'
filename1 = 'D:/Project/2021_H/IMAGE/FROM_CHERRY/train.json'
newfilename = 'D:/Project/2021_H/IMAGE/FROM_CHERRY/cocoformat.json'
'''
filename = './data/H_slab/via_project_WS_TRAIN_100.json'
filename1 = './data/slab/train/train_old.json'
newfilename = './data/H_slab/WS_TRAIN_100.json'

'''
with open(filename1, 'r') as f1:
    cocodata111 = json.load(f1)
    imgcnt = len(cocodata111['images'])
    for i1 in range(0, imgcnt):
        listcnt = len(cocodata111['annotations'][i1]['segmentation'])
        print(listcnt  )

print('1111')
'''

with open(filename1, 'r') as f1:
    cocodata = json.load(f1)
    NCocoData = cocodata.copy()

    for i in reversed(range(0,len(NCocoData['images']))):
        structimg = NCocoData['images'].pop(i)
        structAnno = NCocoData['annotations'].pop(i)

idCnt = 0
errcnt=0
with open(filename, 'r') as f:
    data = json.load(f)

    #via img metadata
    img_meta = data['_via_img_metadata']
    img_names = data['_via_image_id_list']

    for i in range(0,4900):
        img_m = img_meta[img_names[i]]

        if(len(img_m['regions'])==0):
            continue
        if(i==2672 or i==3441):
            continue

        str_img = structimg.copy()
        str_Anno = structAnno.copy()


        #i. Get File Name
        pfilename = img_m['filename']
        #2. Get X,Y coordinate
        xlist = img_m['regions'][0]['shape_attributes']['all_points_x']
        ylist = img_m['regions'][0]['shape_attributes']['all_points_y']

        #xlist min, max
        bboxlist =list()
        minxlist = min(xlist)
        maxxlist = max(xlist)
        minylist = min(ylist)
        maxylist = max(ylist)

        if(maxxlist>=512):
            print('xlist over')
            continue
        if(maxylist>=512):
            print('=ylist over')
            continue

        bboxlist.append(minxlist)
        bboxlist.append(minylist)
        bboxlist.append(maxxlist-minxlist)
        bboxlist.append(maxylist-minylist)

        xylist = list()
        for index in range(len(xlist)):
            xylist.append(xlist[index])
            xylist.append(ylist[index])

        if(len(xylist)<6 or (bboxlist[2] * bboxlist[3]<5)):
            print('8888')
            continue

        idCnt = idCnt + 1

        str_img['id'] = idCnt
        str_img['width'] = 640
        str_img['height'] = 640
        str_img['file_name'] = pfilename
        str_img['license'] = 0
        #str_img['data_captured'] = ''
        NCocoData['images'].append(str_img)

        #str Anno edit
        arr = np.array(xylist)
        #(a, 6)
        arr1 = arr.reshape((1,-1))
        xylist1 = arr1.tolist()

        str_Anno['segmentation'] = xylist1
        str_Anno['area'] = bboxlist[2] * bboxlist[3]
        str_Anno['bbox'] = bboxlist
        str_Anno['iscrowd'] = 0
        str_Anno['id'] = idCnt
        str_Anno['image_id'] = idCnt
        str_Anno['category_id'] = 1
        NCocoData['annotations'].append(str_Anno)


    with open(newfilename, "w") as jsonFile:
        json.dump(NCocoData, jsonFile)

    print('FINISHED')


