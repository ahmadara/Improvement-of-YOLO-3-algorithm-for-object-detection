import sys
sys.path.append("..")
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm


def parse_voc_annotation(data_path, file_type, anno_path,dataset_classes, use_difficult_bbox=False):

    classes = dataset_classes
    img_inds_file = os.path.join(data_path, 'ImageSets', 'Main', file_type+'.txt')
    with open(img_inds_file, 'r') as f:
        lines = f.readlines()
        image_ids = [line.strip() for line in lines]

    with open(anno_path, 'a') as f:
        for image_id in tqdm(image_ids):
            image_path = os.path.join(data_path, 'JPEGImages', image_id + '.jpg')
            annotation = image_path
            label_path = os.path.join(data_path, 'Annotations', image_id + '.xml')
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            for obj in objects:
                difficult = obj.find("difficult").text.strip()
                if (not use_difficult_bbox) and (int(difficult) == 1): # difficult 表示是否容易识别，0表示容易，1表示困难
                    continue
                bbox = obj.find('bndbox')
                class_id = classes.index(obj.find("name").text.lower().strip())
                xmin = bbox.find('xmin').text.strip()
                ymin = bbox.find('ymin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymax = bbox.find('ymax').text.strip()
                annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_id)])
            annotation += '\n'
            # print(annotation)
            f.write(annotation)
    return len(image_ids)