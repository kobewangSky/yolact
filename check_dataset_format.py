import os
import json
import datetime
import numpy as np
import argparse


def CombineCoCoData(path):
    Privacyname = 'images'
    target_dirs = ['Virtualdata_V1', 'Virtualdata_V2', 'instances_train2017', 'instancesonly_filtered_gtFine_train', 'instancesonly_filtered_gtFine_val', 'voc_coco']
    Image_dirs = ['Virtualdata_V1', 'Virtualdata_V2',  'images', 'cityscapes', 'cityscapes', 'JPEGImages']
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
        max_id = 0
        output_images = labels['images']
        output_annotations = labels['annotations']
        for it in output_annotations:
            if it["segmentation"] == []:
                assert False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--testfile', type=str, help='CoCo json file')
    opt = parser.parse_args()
    print(opt)

    CombineCoCoData(
        opt.testfile,
    )