import json
import os

#filename = './data/slab/train/train1.json'
#newfilename = './data/slab/train/train1_1.json'

#filename = './data/slab/val/val2_500_1024.json'
#newfilename = './data/slab/val/val2_1.json'


filename = './data/check_AIPC/val2_500_1024.json'
newfilename = './data/check_AIPC/val2_1.json'
#newfilename = './data/check_AIPC/val2_1.json'


with open(newfilename, 'r') as f:
    data = json.load(f)

    for i in range(0, len(data['images'])):
        if ( data['annotations'][i]['area'] < 5) :
            print('area under 4')
'''

#D:\TEST\yolact_20210718\data\check_AIPC

with open(filename, 'r') as f:
    data = json.load(f)

    print('DATA')

    anno = data['annotations']
    image = data['images']


    count = 0

    for x in reversed(range(0, len(data['images']))):
        data['images'][x]['width'] = 640
        count = count + 1

        # set image size
        #image[x]['width'] = 1024
        image[x]['id'] = count

        # set id and image_id equal
        anno[x]['id'] = count
        anno[x]['image_id'] = anno[x]['id']

        # if(anno[x]['area'] > 5 and anno[x]['area'] < 250):
        if(anno[x]['area'] <= 5):

            # remove annotations and images

            anno.pop(x)
            image.pop(x)

            
            #print('Numner:' + str(x) )
            #print(anno[x]['area'])
            #print('id : ' + str(anno[x]['id']))
            #print('image_id : '+str(anno[x]['image_id']))
            #continue
            

    for x in range(0, len(image)):
        count = count + 1

        # set image size
        #image[x]['width'] = 1024
        image[x]['id'] = count

        # set id and image_id equal
        anno[x]['id'] = count
        anno[x]['image_id'] = anno[x]['id']


    print('Total Count' + str(count) )

    with open(newfilename, "w") as jsonFile:
        json.dump(data, jsonFile)

    print('FINISHED')


'''