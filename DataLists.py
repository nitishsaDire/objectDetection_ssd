import pandas as pd
import xml.etree.ElementTree as ET
import torch
class_to_label = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                  'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', "bg"]


def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    difficulty = []
    bbox_xyxy = []
    labels = []
    for object in root.iter('object'):
        label = object.find('name').text.lower().strip()

        if label not in class_to_label:
            continue

        difficulty.append(int(object.find('difficult').text == '1'))

        xmin = int(float(object.find('bndbox').find('xmin').text)) - 1
        xmax = int(float(object.find('bndbox').find('xmax').text)) - 1
        ymin = int(float(object.find('bndbox').find('ymin').text)) - 1
        ymax = int(float(object.find('bndbox').find('ymax').text)) - 1

        bbox_xyxy.append([xmin, ymin, xmax, ymax])
        labels.append(label)

    return bbox_xyxy, labels, difficulty


all_images = {}
all_multi_labels = {}
all_multi_bboxes = {}
all_difficulties = {}


def get_all_images(isTrainData=True):
    if isTrainData:
        all_images = ['VOCdevkit/VOC2007/JPEGImages/{:06d}.jpg'.format(i) \
                      for i in list(pd.read_csv('VOCdevkit/VOC2007/ImageSets/Main/trainval.txt', header=None,
                                                names=['files']).files.values)]
        all_images += ['VOCdevkit/VOC2012/JPEGImages/{}.jpg'.format(i) for i in \
                       list(pd.read_csv('VOCdevkit/VOC2012/ImageSets/Main/trainval.txt', header=None,
                                        names=['files']).files.values)]
    else:
        all_images = ['VOCdevkit/VOC2012/JPEGImages/{}.jpg'.format(i) \
                      for i in list(
                pd.read_csv('VOCdevkit/VOC2012/ImageSets/Main/test.txt', header=None, names=['files']).files.values)]

    return all_images


def get_all_xml(isTrainData=True):
    if isTrainData:
        all_xml = ['VOCdevkit/VOC2007/Annotations/{:06d}.xml'.format(i) \
                   for i in list(pd.read_csv('VOCdevkit/VOC2007/ImageSets/Main/trainval.txt', header=None,
                                             names=['files']).files.values)]
        all_xml += ['VOCdevkit/VOC2012/Annotations/{}.xml'.format(i) \
                    for i in list(pd.read_csv('VOCdevkit/VOC2012/ImageSets/Main/trainval.txt', header=None,
                                              names=['files']).files.values)]
    else:
        all_xml = ['VOCdevkit/VOC2012/Annotations/{}.xml'.format(i) \
                   for i in list(
                pd.read_csv('VOCdevkit/VOC2012/ImageSets/Main/test.txt', header=None, names=['files']).files.values)]
    return all_xml


def get_all_multi_labels(isTrainData=True):
    all_multi_labels = []
    all_multi_bboxes = []
    all_difficulties = []

    xml_files = get_all_xml(isTrainData)

    for xml_file in xml_files:
        bbox_xyxy, labels, difficulty = parse_xml(xml_file)
        all_multi_bboxes.append(bbox_xyxy)
        all_multi_labels.append(labels)
        all_difficulties.append((difficulty))
    return all_multi_labels, all_multi_bboxes, all_difficulties


def call_on_load():
    global all_images, all_multi_labels, all_multi_bboxes

    all_images_tr_val = get_all_images()
    all_multi_labels_tr_val, all_multi_bboxes_tr_val, all_diff_tr_val = get_all_multi_labels()

    all_images['train'] = all_images_tr_val
    all_multi_bboxes['train'] = all_multi_bboxes_tr_val
    all_multi_labels['train'] = all_multi_labels_tr_val
    all_difficulties['train'] = all_diff_tr_val
