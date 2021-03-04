from fastai.vision import *
import json
from Util import get_largest_bbox_with_label
import numpy as np
import torch

path = Path('pascal_2007/')

annots = json.load(open(path/'train.json'))

train_images, train_lbl_bbox = get_annotations(path/'train.json')
val_images, val_lbl_bbox = get_annotations(path/'valid.json')


all_images = []
for images in train_images:
    all_images.append(path/'train'/images)

for images in val_images:
    all_images.append(path/'train'/images)

all_multi_labels = []
for idx in range(len(train_images)):
  all_multi_labels.append(train_lbl_bbox[idx][1])


for idx in range(len(val_images)):
  all_multi_labels.append(val_lbl_bbox[idx][1])


all_labels = []
for idx in range(len(train_images)):
  all_labels.append(get_largest_bbox_with_label(train_lbl_bbox[idx])[1][0])


for idx in range(len(val_images)):
  all_labels.append(get_largest_bbox_with_label(val_lbl_bbox[idx])[1][0])


all_bboxes = []
for idx in range(len(train_images)):
  l = get_largest_bbox_with_label(train_lbl_bbox[idx])[0][0]
  f = [float(i) for i in l]
  all_bboxes.append(f)


for idx in range(len(val_images)):
  l = get_largest_bbox_with_label(val_lbl_bbox[idx])[0][0]
  f = [float(i) for i in l]
  all_bboxes.append(f)

def convert_list_to_float(l):
  return [float(i) for i in l]

all_multi_bboxes = []
for idx in range(len(train_images)):
  l = train_lbl_bbox[idx][0]
  f = [convert_list_to_float(i) for i in l]
  all_multi_bboxes.append(f)


for idx in range(len(val_images)):
  l = val_lbl_bbox[idx][0]
  f = [convert_list_to_float(i) for i in l]
  all_multi_bboxes.append(f)

def denormalize(input):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return np.multiply(std,input) + mean

def one_zero(t, n):
  output = torch.zeros(n)
  for i in t:
    output[i]=1
  return output