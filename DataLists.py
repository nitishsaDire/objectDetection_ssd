from fastai.vision import *
import json

path = Path('pascal_2007/')

train_images, train_lbl_bbox = get_annotations(path/'train.json')
val_images, val_lbl_bbox = get_annotations(path/'valid.json')
test_images, test_lbl_bbox = get_annotations(path/'test.json')

all_images = {}
all_multi_labels = {}
all_multi_bboxes = {}

def get_all_images(isTrainData=True):
  all_images = []
  if isTrainData:
    for image in train_images+val_images:
      all_images.append(path / 'train' / image)
  else:
    for image in test_images:
      all_images.append(path / 'test' / image)

  return all_images

def get_all_multi_labels(isTrainData=True):
  all_multi_labels = []
  if isTrainData:
    for idx in range(len(train_images)):
      all_multi_labels.append(train_lbl_bbox[idx][1])
    for idx in range(len(val_images)):
      all_multi_labels.append(val_lbl_bbox[idx][1])

  else:
    for idx in range(len(test_images)):
      all_multi_labels.append(test_lbl_bbox[idx][1])

  return all_multi_labels


def convert_list_to_float(l):
  l =  [float(i) for i in l]
  l[0], l[1], l[2], l[3] = l[1], l[0], l[3], l[2]
  return l


def get_all_multi_bboxes(isTrainData=True):
  all_multi_bboxes = []

  if isTrainData:
    for idx in range(len(train_images)):
      l = train_lbl_bbox[idx][0]
      f = [convert_list_to_float(i) for i in l]
      all_multi_bboxes.append(f)

    for idx in range(len(val_images)):
      l = val_lbl_bbox[idx][0]
      f = [convert_list_to_float(i) for i in l]
      all_multi_bboxes.append(f)
  else:
    for idx in range(len(test_images)):
      l = test_lbl_bbox[idx][0]
      f = [convert_list_to_float(i) for i in l]
      all_multi_bboxes.append(f)

  return all_multi_bboxes


def call_on_load():
  global all_images, all_multi_labels, all_multi_bboxes

  all_images_tr_val, all_images_test = get_all_images(), get_all_images(isTrainData=False)
  all_multi_bboxes_tr_val, all_multi_bboxes_test = get_all_multi_bboxes(), get_all_multi_bboxes(isTrainData=False)
  all_multi_labels_tr_val, all_multi_labels_test = get_all_multi_labels(), get_all_multi_labels(isTrainData=False)

  all_images['train'] = all_images_tr_val
  all_images['test'] = all_images_test
  all_multi_bboxes['train'] = all_multi_bboxes_tr_val
  all_multi_bboxes['test'] = all_multi_bboxes_test
  all_multi_labels['train'] = all_multi_labels_tr_val
  all_multi_labels['test'] = all_multi_labels_test
