import json
import os
import numpy as np

# filename => via project file
# filename1 => for copy coco data structure
# newfilename => coco format data filename

filename = './data/H_slab/via_project_DS_valid_0100.json'
                        # via_project_DS_vaild_0100.json
filename1 = './data/slab/train/train_old.json'
newfilename = './data/H_slab/DS_VALID_100.json'

imagecnt = 100

raw_w = 1600
raw_h = 800

resize_w = 640
resize_h = 640

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

    for i in range(0,imagecnt):
        img_m = img_meta[img_names[i]]
        print(i)
        i1 = i+1
        # if(i1==3 or i1==5 or i1==127 or i1==128 or i1==131 or i1==233 or i1==244 or i1==302 or i1==311 or i1==325 or i1==231):
        #     continue
        if(i1==2 or i1==7):
            continue


        if(i==230):
            print('230')

        if(len(img_m['regions'])==0):
            continue

        if(len(img_m['regions'][0]['shape_attributes']['all_points_x'])==0):
            continue

        if(len(img_m['regions'][0]['shape_attributes']['all_points_y'])==0):
            continue

        str_img = structimg.copy()
        str_Anno = structAnno.copy()

        #i. Get File Name
        pfilename = img_m['filename']
        #2. Get X,Y coordinate
        xlist1 = img_m['regions'][0]['shape_attributes']['all_points_x']
        ylist1 = img_m['regions'][0]['shape_attributes']['all_points_y']
        xlist=[]
        ylist=[]
        for i in range(0,len(xlist1)):
            xlist.append( int(xlist1[i] * (640.0/1600.0)))
        for i in range(0,len(ylist1)):
            ylist.append( int(ylist1[i] * (640.0/800.0)) )

        #xlist min, max
        bboxlist =list()
        minxlist = min(xlist)
        maxxlist = max(xlist)
        minylist = min(ylist)
        maxylist = max(ylist)

        bboxlist.append(minxlist)
        bboxlist.append(minylist)
        bboxlist.append(maxxlist-minxlist)
        bboxlist.append(maxylist-minylist)

        xylist = list()
        for index in range(len(xlist)):
            xylist.append(xlist[index])
            xylist.append(ylist[index])

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


