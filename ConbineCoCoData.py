import os
import json
import datetime
import numpy as np
import argparse


def CombineCoCoData(path):
    Privacyname = 'images'
    target_dirs = ['Virtualdata_V1', 'Virtualdata_V2', 'instances_train2017']
    Image_dirs = ['Virtualdata_V1', 'Virtualdata_V2',  'images']
    target_file = path

    Outputname = str()
    for target_dir in target_dirs:
       Outputname += target_dir + '_'

    target_file = os.path.join(target_file, Outputname + '.json')

    output_images = {}
    output_annotations = {}

    INFO = {
        "description": "Dataset",
        "url": "",
        "version": "0.1.0",
        "year": 2019,
        "contributor": "",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    LICENSES = [
        {
            "id": 1,
            "name": "",
            "url": ""
        }
    ]

    CATEGORIES = [
        {
            'id': 1,
            'name': 'person',
            'supercategory': 'person',
        }
    ]

    temp_id = 0
    anotation_id = 0
    for idx, target_dir in enumerate(target_dirs):
        target_json = os.path.join(path, target_dir + '.json')
        labels = json.load(open(target_json))
        if idx == 0:
            max_id = 0
            output_images = labels['images']
            output_annotations = labels['annotations']
            for i in range(len(output_images)):
                output_images[i]['file_name'] = os.path.join('..',Image_dirs[idx], output_images[i]['file_name'])
                output_images[i]['id'] = int(output_images[i]['id'])
                if output_images[i]['id'] > max_id:
                    max_id = output_images[i]['id']
            for i in range(len(output_annotations)):
                output_annotations[i]['image_id'] = int(output_annotations[i]['image_id'])
                output_annotations[i]['id'] = '{}'.format(anotation_id)
                anotation_id = anotation_id + 1
            temp_id += max_id
        else:
            max_id = 0
            temp_images = labels['images']
            temp_annotations = labels['annotations']
            for i in range(len(temp_images)):
                temp_images[i]['file_name'] = os.path.join('..', Image_dirs[idx], temp_images[i]['file_name'])
                temp_images[i]['id'] = int(temp_images[i]['id']) + temp_id + 1
                if temp_images[i]['id'] > max_id:
                    max_id = temp_images[i]['id']
            for i in range(len(temp_annotations)):
                temp_annotations[i]['image_id'] = int(temp_annotations[i]['image_id']) + temp_id + 1
                temp_annotations[i]['id'] = '{}'.format(anotation_id)
                anotation_id = anotation_id + 1
                # temp_annotations[i]['id'] = int(temp_annotations[i]['id']) + len(output_annotations)

            output_images.extend(temp_images)
            output_annotations.extend(temp_annotations)
            temp_id += max_id

    # check id is unique
    image_ids = []
    annotation_ids = []

    for i in range(len(output_images)):
        image_ids.append(output_images[i]['id'])
    for i in range(len(output_annotations)):
        annotation_ids.append(output_annotations[i]['id'])

    image_ids = np.array(image_ids)
    annotation_ids = np.array(annotation_ids)

    unique = False
    if len(image_ids) == len(np.unique(image_ids)):
        print('image_id is unique!')
        if len(annotation_ids) == len(np.unique(annotation_ids)):
            print('annotation_id is unique!')
            unique = True

    # save file
    output_json = {
        'info': INFO,
        'licenses': LICENSES,
        'categories': CATEGORIES,
        'images': output_images,
        'annotations': output_annotations
    }

    if unique:
        with open(target_file, 'w') as f:
            json.dump(output_json, f)
        print('save annotation!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--testfile', type=str, help='CoCo json file')
    opt = parser.parse_args()
    print(opt)

    CombineCoCoData(
        opt.testfile,
    )