import sys
import os
# import re
import json
import numpy as np

# The output json format
#-----------
# annotations: id, image_id, bbox, category, area, ignore, iscrowd, segmentation
# categories: id, name, supercategory
# images: filename, height, width, id 

#input format is csv: path,width,height,left,top,right,bottom,label

#Usage: python convert_to_detectron.py train.txt test.txt
#writes results in same folder as input with .json extension

def get_category_id(label):
    for x in categories:
        if x['name'] == label:
            return x['id']


categories = []
categories_id = 1

labels_map = {}
def convert(input_file, training=True):
    global categories, categories_id
    annotations = []
    images = []

    annotations_id = 1
    images_id = 0
    prev = ''
    with open(input_file) as f:
        for line in f:
            path,width,height,left,top,right,bottom,label = line.rstrip().split(',')

            if path in BAD:
                print('Found bad images')
                continue
         
            left = float(left)
            top = float(top)
            right = float(right)
            bottom = float(bottom)

            area = (bottom - top) * (right - left)
            ratio = (bottom - top) / (right - left)
            min_side = min((bottom - top), (right - left))
            if(area <= 0 or (bottom - top <= 0) or (right - left <= 0)):
                print("BBOX ERROR %g,%g,%g,%g" %(left,top,right,bottom))
                continue

            #if(ratio > 3.0 or ratio < 1/3.0 or min_side < 16):
            #    continue

            #take full path (not relative)
            #actually take relative
            #filename = path
            filename = path.rsplit('/',1)[1]

            #update categories when we find a new one
            if training and not np.any([label in x['name'] for x  in categories]):
                new_category = {}
                new_category['id'] = categories_id
                categories_id += 1
                new_category['name'] = label
                new_category['supercategory'] = 'none'
                categories.append(new_category)

            #     labels_map[re.sub('[-_]', '', label)] = label

            # if not training and label in labels_map:
            #     label = labels_map[label]

            #update images when we find a new one
            #if not np.any([filename in x['file_name'] for x in images]):
            if filename != prev:
                images_id += 1
                new_image = {}
                new_image['file_name'] = filename
                new_image['width'] = int(width)
                new_image['height'] = int(height)
                new_image['id'] = images_id
                images.append(new_image)
          

            #update annotations
            new_annotation = {}
            new_annotation['id'] = annotations_id
            annotations_id += 1
            new_annotation['image_id'] = images_id
            new_annotation['bbox'] = [int(left),int(top),int(right - left),int(bottom - top)]
            new_annotation['category_id'] = get_category_id(label)


            new_annotation['area'] = (bottom - top) * (right - left)
            new_annotation['ignore'] = 0    #ignore nothing!
            new_annotation['iscrowd'] = 0   #no crowds
            new_annotation['segmentation'] = [] #no segmentation; hope this works
            annotations.append(new_annotation)
            prev = filename
    return annotations,images 

print("Converting training set...")
#also sets categories
annotations, images = convert(sys.argv[1])

print("writing train json")
final_json = {'annotations': annotations, 'categories': categories, 'images': images }
root,ext = sys.argv[1].rsplit('.',1)
with open(root + '.json','w') as f:
    json.dump(final_json, f)

print("Converting test set...")
#doesnt set categories
annotations, images = convert(sys.argv[2],training=False)

print("writing test json")
final_json = {'annotations': annotations, 'categories': categories, 'images': images }
root,ext = sys.argv[2].rsplit('.',1)
with open(root + '.json','w') as f:
    json.dump(final_json, f)

with open(os.path.join(root.rsplit('/',1)[0], 'labels.pbtxt'), 'wa+') as f:
    for val in categories:
        # print('>>>>>>>>>> val', val, val['id'], val['name'])
        # label_map_string = """
        #   item {
        #     id:{}
        #     name:{}
        #   }.format(val['id'],val['name'])
        # """
        f.write('item{\n\tid:' + str(val['id']) + '\n\tname:"' + val['name'] + '"\n}\n')
        # final_json = {'item': {'id': val['id'], 'name': val['name']}}
        # json.dump(final_json, f, separators=('',':'), indent=2)


