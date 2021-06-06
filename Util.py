import itertools
import numpy as np
import random
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
# from shapely.geometry import Polygon
import torchvision.transforms.functional as FT
from matplotlib import patches
use_cuda = torch.cuda.is_available()  # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
from DataLists import call_on_load
call_on_load()
from DataLists import all_images, all_multi_labels, all_multi_bboxes, all_difficulties
import os
grid_sizes_list = list(itertools.chain([0.25] * 144, [0.5] * 36, [1.] * 9))


transform = transforms.Compose([        transforms.Resize((300, 300)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])

class_to_label = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                  'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', "bg"]

label_to_class = {}
for idx, i in enumerate(class_to_label):
    label_to_class[i] = idx

distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']

def denormalize(input):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return np.multiply(std,input) + mean

from math import sqrt


def xyxy_to_yxyx(anchors):
    print(len(anchors), anchors[0])
    anchors = torch.Tensor(anchors)
    x1 = anchors[:, 0]
    y1 = anchors[:, 1]
    x2 = anchors[:, 2]
    y2 = anchors[:, 3]
    yx_diags = np.stack((y1,x1,y2,x2), axis=1)
    return yx_diags



def xyxy_to_xywh(anchors):
    x1 = anchors[:, 0].cpu()
    y1 = anchors[:, 1].cpu()
    x2 = anchors[:, 2].cpu()
    y2 = anchors[:, 3].cpu()

    return torch.Tensor(np.stack( ( (x2 + x1)/2., (y2 + y1)/2., x2-x1, y2-y1), axis=1))

def yxyx_to_xyxy(anchors):
    anchors = torch.Tensor(anchors)
    # print(anchors.shape)
    y1 = anchors[:, 0]
    x1 = anchors[:, 1]
    y2 = anchors[:, 2]
    x2 = anchors[:, 3]

    xy_diags = np.stack((x1, y1, x2, y2), axis=1)

    return xy_diags

def xywh_to_yxyx(anchors):
    x = anchors[:, 0]
    y = anchors[:, 1]
    w = anchors[:, 2]
    h = anchors[:, 3]
    x, y, w, h = x.cpu().detach().numpy(), y.cpu().detach().numpy(), w.cpu().detach().numpy(), h.cpu().detach().numpy()
    yx_diags = np.stack((y - h / 2., x - w / 2., y + h / 2., x + w / 2.), axis=1)
    return torch.Tensor(yx_diags)

def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):

    gcxgcy = gcxgcy.to(device)
    priors_cxcy = priors_cxcy.to(device)
    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h

def xywh_to_xyxy(box):
    x_y = box[:, :2]
    w_h = box[:, 2:]
    return torch.cat((x_y - w_h / 2., x_y + w_h / 2.), dim=1)

def get_offsets_coords(cxcy, priors_cxcy):
    cxcy = cxcy.to(device)
    priors_cxcy = priors_cxcy.to(device)
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


def create_priors_ssd300():

    print("-----------------------create priors-------------------")
    anc_grids = [38., 19., 10., 5., 3., 1.]

    # scales = [get_scale(i) for i in range(1, 7)]
    scales = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9]

    anc_ratios = [[1., 2., 0.5, ],
                  [1., 2., 3., 0.5, .333],
                  [1., 2., 3., 0.5, .333],
                  [1., 2., 3., 0.5, .333],
                  [1., 2., 0.5],
                  [1., 2., 0.5]
                  ]
    priors = []
    for idx, anc_grid in enumerate(anc_grids):
        for i in range(int(anc_grid)):
            for j in range(int(anc_grid)):
                cx, cy = j + 0.5, i + 0.5
                cx /= anc_grids[idx]
                cy /= anc_grids[idx]
                for a in anc_ratios[idx]:
                    priors.append([cx, cy, scales[idx] * sqrt(a), scales[idx] / sqrt(a)])
                    if a == 1.:
                        try:
                            scale = sqrt(scales[idx] * scales[idx + 1])
                        except IndexError:
                            scale = 1.
                        priors.append([cx, cy, scale, scale])
    priors = torch.FloatTensor(priors)
    priors.clamp_(0, 1)
    return priors


def get_scale(k):
    return round(0.2 + 0.7 * (k - 1) / 5.0, 2)


def create_ancs_xywh_zoom_ratio():
    # anc_grids = [4]
    anc_grids = [4, 2, 1]

    anc_zooms = [0.75, 1., 1.3]
    anc_ratios = [(1., 1.), (1., 0.5), (0.5, 1.)]

    # anc_zooms = [ 1.]
    # anc_ratios = [(1., 1.)]

    anchor_scales = [(anz * i, anz * j) for anz in anc_zooms for (i, j) in anc_ratios]
    k = len(anchor_scales)
    anc_offsets = [1 / (o * 2) for o in anc_grids]
    anc_x = np.concatenate([np.repeat(np.linspace(ao, 1 - ao, ag), ag) for ao, ag in zip(anc_offsets, anc_grids)])
    anc_y = np.concatenate([np.tile(np.linspace(ao, 1 - ao, ag), ag) for ao, ag in zip(anc_offsets, anc_grids)])
    anc_ctrs = np.repeat(np.stack([anc_x, anc_y], axis=1), k, axis=0)
    anc_sizes = np.concatenate([np.array([[o / ag, p / ag] for i in range(ag * ag) for o, p in anchor_scales]) for ag in anc_grids])
    grid_sizes = np.concatenate([np.array([1 / ag for i in range(ag * ag) for o, p in anchor_scales]) for ag in anc_grids])
    ancs = torch.Tensor(np.concatenate([anc_ctrs, anc_sizes], axis=1))
    # anchor_cnr = hw2corners(anchors[:,:2], anchors[:,2:])
    return torch.stack((ancs[:, 1], ancs[:, 0], ancs[:, 2], ancs[:, 3]), dim=1)



def draw_with_ancs(bboxes=None, scale=300, size=(4, 4)):
    _, ax = plt.subplots(figsize=(5, 5))

    for idx, bbox in enumerate(bboxes):
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)

    ax.set_ylim(1, 0)  # -1 is top, 1 is bottom
    for i, (x, y) in enumerate(zip(bboxes[:, 0] + (bboxes[:, 2] - bboxes[:, 0]) / 2,
                                   bboxes[:, 1] + (bboxes[:, 3] - bboxes[:, 1]) / 2)): ax.annotate(i, xy=(x, y))

    plt.show()


def show_anchors(ancs, size, scale=1):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    lower = 0
    ancs *= scale
    ax.set_xticks(np.linspace(lower, 1 * scale, size[1] + 1))
    ax.set_yticks(np.linspace(lower, 1 * scale, size[0] + 1))
    ax.grid()
    ax.scatter(ancs[:, 1], ancs[:, 0])  # y is first
    # ax.set_yticklabels([])
    # ax.set_xticklabels([])
    ax.set_xlim(lower, 1 * scale)
    ax.set_ylim(1 * scale, 0)  # -1 is top, 1 is bottom
    for i, (x, y) in enumerate(zip(ancs[:, 1], ancs[:, 0])): ax.annotate(i, xy=(x, y))
    return fig


def create_ancs_centre_coords(size=(4, 4)):
    "Create a grid of a given `size`."
    H, W = size
    grid = torch.Tensor(H, W, 2)
    lower = 0
    linear_points = torch.linspace(lower + 1 / (2 * W), 1 - 1 / (2 * W), W) if W > 1 else torch.Tensor([0.])
    grid[:, :, 1] = torch.ger(torch.ones(H), linear_points).expand_as(grid[:, :, 0])
    linear_points = torch.linspace(lower + 1 / (2 * H), 1 - 1 / (2 * H), H) if H > 1 else torch.Tensor([0.])
    grid[:, :, 0] = torch.ger(linear_points, torch.ones(W)).expand_as(grid[:, :, 1])
    return grid.view(-1, 2)


def get_ancs_diag_coords(ancs):
    diags = []
    side = ancs[1][1] - ancs[0][1]
    for anc in ancs:
        diags.append(torch.cat((anc - side / 2, anc + side / 2), dim=0).tolist())
    return diags


# for one
def get_ancs_coords_cw(diag_coords):
    d1y, d1x, d2y, d2x = diag_coords
    return [(d1y, d1x), (d1y, d2x), (d2y, d2x), (d2y, d1x)]


# for one
def get_img_sz(img_path):
    im = Image.open(img_path)
    return im.size


# for one
def calculate_iou(box_1, box_2):
    # expects box1, box2 coords in yxyx diags format
    poly_1 = Polygon(get_ancs_coords_cw(box_1))
    poly_2 = Polygon(get_ancs_coords_cw(box_2))
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou


def get_jaccard(index, diags_ancs_yxyx, scale=300, transform_img_sz=(300, 300)):
    diags_ancs_yxyx = diags_ancs_yxyx * scale
    # expects diags_ancs in yxyx format

    overlaps = []
    img_sz = get_img_sz(all_images_tr[index])
    bboxes_yxyx = scale_bboxes(standardize_bboxes(all_multi_bboxes_tr[index], img_sz), transform_img_sz)

    for bbox in bboxes_yxyx:
        overlaps.append([calculate_iou(anc, bbox) for anc in diags_ancs_yxyx])
    return torch.Tensor(overlaps)

def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)

def get_jaccard_tensor(index, box2_xyxy, phase='train'):

    box1_xyxy = torch.Tensor(all_multi_bboxes[phase][index]).to(device)
    w, h = get_img_sz(all_images[phase][index])
    box1_xyxy /= torch.FloatTensor([w, h, w, h]).unsqueeze(0).to(device)
    box1_xyxy *= torch.FloatTensor([300.,300.,300.,300.]).unsqueeze(0).to(device)

    box2_xyxy = box2_xyxy.to(device)
    # Find intersections
    intersection = find_intersection(box1_xyxy, box2_xyxy)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (box1_xyxy[:, 2] - box1_xyxy[:, 0]) * (box1_xyxy[:, 3] - box1_xyxy[:, 1])  # (n1)
    areas_set_2 = (box2_xyxy[:, 2] - box2_xyxy[:, 0]) * (box2_xyxy[:, 3] - box2_xyxy[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)

def get_jaccard_tensor1(box1_xyxy, box2_xyxy):
    # Find intersections
    box1_xyxy, box2_xyxy = box1_xyxy.to(device), box2_xyxy.to(device)
    intersection = find_intersection(box1_xyxy, box2_xyxy)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (box1_xyxy[:, 2] - box1_xyxy[:, 0]) * (box1_xyxy[:, 3] - box1_xyxy[:, 1])  # (n1)
    areas_set_2 = (box2_xyxy[:, 2] - box2_xyxy[:, 0]) * (box2_xyxy[:, 3] - box2_xyxy[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)

def get_jaccard_tensor11(box1_xyxy, box2_xyxy):
    # Find intersections
    # box1_xyxy, box2_xyxy = box1_xyxy.to(device), box2_xyxy.to(device)
    intersection = find_intersection(box1_xyxy, box2_xyxy)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (box1_xyxy[:, 2] - box1_xyxy[:, 0]) * (box1_xyxy[:, 3] - box1_xyxy[:, 1])  # (n1)
    areas_set_2 = (box2_xyxy[:, 2] - box2_xyxy[:, 0]) * (box2_xyxy[:, 3] - box2_xyxy[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)




def get_jaccard_old(index, ancs_sz=(4, 4), scale=300, transform_img_sz=(300, 300), phase = 'train'):
    diags_ancs = get_ancs_diag_coords(create_ancs_centre_coords(ancs_sz) * scale)

    overlaps = []
    img_sz = get_img_sz(all_images[phase][index])
    bboxes = scale_bboxes(standardize_bboxes(all_multi_bboxes[phase][index], img_sz), transform_img_sz)

    for bbox in bboxes:
        overlaps.append([calculate_iou(anc, bbox) for anc in diags_ancs])
    return torch.Tensor(overlaps)


def map_prior_to_bb(jacc, classes, threshold=0.5):
    # jacc has shape n_objects, total_priors
    jacc = jacc.to(device)

    # for each prior the object it has max overlap with, axis=0 means along x axis, i.e. along total_priors
    overlap_forEach_prior, obj_forEach_prior = jacc.max(axis=0)  # (189), (189)
    overlap_forEach_prior, obj_forEach_prior = overlap_forEach_prior.to(device), obj_forEach_prior.to(device)

    # for each obj, the value of overlap, prior it has max overlap with, axis=1 means along y axis, i.e. along n_objects
    _, prior_forEach_obj = jacc.max(axis=1)  #Num_objs
    prior_forEach_obj = prior_forEach_obj.to(device)

    obj_forEach_prior[prior_forEach_obj] = torch.LongTensor(range(jacc.shape[0])).to(device)
    overlap_forEach_prior[prior_forEach_obj] = 1.

    # classes = torch.Tensor([label_to_class[v] for v in labels])
    class_forEach_prior = classes[obj_forEach_prior]
    class_forEach_prior[overlap_forEach_prior < threshold] = 20

    return class_forEach_prior, obj_forEach_prior

def get_xywh_from_yxyx(diags):
    """
    yxyx --> xywh
    """
    cent_coords = []
    for diag in diags:
        d1y, d1x, d2y, d2x = diag

        w = d2x - d1x
        h = d2y - d1y
        cy = d1y + h / 2
        cx = d1x + w / 2
        # print(torch.stack((cx,cy,w,h)))
        cent_coords.append(torch.stack((cx, cy, w, h)).tolist())
    return torch.Tensor(cent_coords)


def remove_padding(bboxs):
    no_padding_bboxes = []
    for bbox in bboxs:
        if bbox[0] == bbox[1] == bbox[2] == bbox[3] == 0.:
            break
        no_padding_bboxes.append(bbox.tolist())
    return torch.Tensor(no_padding_bboxes)


def remove_padding_batch(bboxs):
    no_padding_bboxes = []
    for bbox in bboxs:
        no_padding_bboxes.append(remove_padding(bbox))
    return no_padding_bboxes


def standardize_bboxes(bboxes_xyxy, img_sz):
    w, h = img_sz
    return bboxes_xyxy.to(device) / torch.FloatTensor([w, h, w, h]).unsqueeze(0).to(device)

def scale_bboxes(bboxes_xyxy, scale_sz):
    w, h = scale_sz
    return bboxes_xyxy * torch.FloatTensor([w, h, w, h]).unsqueeze(0).to(device)


def get_p_bbox_labels(output, ancs_xywh, device, k=1, apply_softmax=False):
    soft = nn.Softmax2d()
    output = output.to(device)
    ancs_xywh = torch.Tensor(ancs_xywh).to(device)
    p_bbox = output[:, :4 * k, :, :].to(device)
    p_labels = output[:, 4 * k:, :, :].to(device)
    bs = p_bbox.shape[0]
    new_xy = torch.reshape(torch.tanh(p_bbox), (bs, 16, 4 * k))[:, :, :2] / 2 * 0.25 + ancs_xywh[:, :2]
    new_wh = (torch.reshape(torch.tanh(p_bbox), (bs, 16, 4 * k))[:, :, :2] / 2 + 1) * (ancs_xywh[:, 2:])
    if apply_softmax == True:
        p_labels = soft(p_labels)
    return torch.cat((new_xy, new_wh), dim=2), p_labels

def get_p_bbox_labels_za(output, ancs_xywh, device, k=9, apply_softmax=False):
    # soft = nn.Softmax(dim = 2)
    l, c = output
    l = l.to(device)  # (bs,16*k,25)
    c = c.to(device)
    ancs_xywh = ancs_xywh.to(device)

    p_bbox = l.to(device)
    p_labels = c.to(device)

    grid_sizes_list_t = torch.Tensor(grid_sizes_list).to(device)
    grid_sizes_list_st = torch.stack((grid_sizes_list_t, grid_sizes_list_t), dim=1)
    new_xy = torch.tanh(p_bbox)[:, :, :2] / 2 * grid_sizes_list_st + ancs_xywh[:, :2]
    new_wh = (torch.tanh(p_bbox)[:, :, 2:] / 2 + 1) * (ancs_xywh[:, 2:])

    return torch.cat((new_xy, new_wh), dim=2), p_labels
    # return ancs_xywh.repeat((output.shape[0],1,1)), p_labels
    # (torch.Size([bs, 16*k, 4]), torch.Size([bs, 21*k, 4, 4]))


def draw_image_with_ancs_xyxy(image_path, bboxes, labels, pred_prob=None):
    _, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(Image.open(image_path))
    classes = [label_to_class[i] for i in labels]

    if pred_prob != None:
        labels = [i + str(round(p,2)) for i,p in zip(labels, pred_prob.tolist())]
        for idx, label in enumerate(labels):
            print(str(idx) + "_" + label+", ", end =" ")

    for idx, bbox in enumerate(bboxes):
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=3,
                                 edgecolor=distinct_colors[classes[idx]], facecolor='none')
        ax.add_patch(rect)
        ax.text(bbox[0], bbox[1], str(idx) + "_" + labels[idx], verticalalignment='top', color='r',
                fontsize=10,weight='bold')

    plt.show()


def draw_image_with_labelled_ancs(image_path, labels=None, isTransform=True, scale=300, size=(4, 4)):
    img = Image.open(image_path)
    _, ax = plt.subplots(figsize=(5, 5))
    if isTransform == True:
        ax.imshow(denormalize(transform(img).permute(1, 2, 0)))
    else:
        ax.imshow(img)

    lower = 0
    ancs = create_ancs_centre_coords(size)
    ancs *= scale
    ax.set_xticks(np.linspace(lower, 1 * scale, size[1] + 1))
    ax.set_yticks(np.linspace(lower, 1 * scale, size[0] + 1))
    ax.grid()
    ax.scatter(ancs[:, 1], ancs[:, 0])  # y is first

    ax.set_xlim(lower, 1 * scale)
    ax.set_ylim(1 * scale, 0)  # -1 is top, 1 is bottom
    for i, (x, y) in enumerate(zip(ancs[:, 1], ancs[:, 0])): ax.annotate(i, xy=(x, y))
    topleft_corners = ancs - 28
    for idx, label in enumerate(labels):
        ax.text(topleft_corners[idx, 1], topleft_corners[idx, 0], str(idx) + "_" + label, verticalalignment='top',
                color="r", fontsize=10, weight='bold')

    plt.show()


def draw_image(image_path, bboxes=None, labels=None, isTransform=True):
    img = Image.open(image_path)
    print(img.size)
    _, ax = plt.subplots(figsize=(5, 5))
    if isTransform == True:
        ax.imshow(denormalize(transform(img).permute(1, 2, 0)))
    else:
        ax.imshow(img)

    if bboxes != None:
        for idx, bbox in enumerate(bboxes):
            rect = patches.Rectangle((bbox[1], bbox[0]), bbox[3] - bbox[1], bbox[2] - bbox[0], linewidth=1,
                                     edgecolor='w', facecolor='none')  # (x, y), width, height
            ax.add_patch(rect)
            if labels != None:
                ax.text(bbox[1], bbox[0], labels[idx], verticalalignment='top', color="white", fontsize=14,
                        weight='bold')
    elif labels != None:
        for idx, label in enumerate(labels):
            # print(labels, len(labels), idx)
            ax.text(0, idx * 30, label, verticalalignment='top', color="r", fontsize=14, weight='bold')
    plt.show()
    # return ax


def get_largest_bbox_with_label(bboxes_labels):
    d = {}
    bboxes, labels = bboxes_labels[0], bboxes_labels[1]
    for idx, bbox in enumerate(bboxes):
        d[idx] = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    x = [k for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)]
    return [[bboxes[x[0]]], [labels[x[0]]]]

def one_zero(t, n):
  output = torch.zeros(n)
  for i in t:
    output[i]=1
  return output

#
# nms(o, all_images[index], torch.cat(all_pred_bboxes), [class_to_label[i] for i in classes], torch.cat(all_pred_prob),
#     overlap_threshold=0.3)

def nms(image_path, pred_bboxes, pred_labels, pred_probs, overlap_threshold=0.5):
    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

    xyxy = xywh_to_xyxy(pred_bboxes)
    n_preds = xyxy.shape[0]
    pred_probs = torch.Tensor(pred_probs).contiguous().view((1, n_preds)).to(device)

    iou_val = get_jaccard_tensor1(xyxy,xyxy).to(device)
    iou_val.fill_diagonal_(0.01)

    pred_cl = torch.Tensor([label_to_class[i] for i in pred_labels]).contiguous().view(1,n_preds).to(device)
    iou = (iou_val>overlap_threshold).to(device)
    preds_Tr_if_same_cl = torch.zeros((n_preds,n_preds)).to(device)

    preds_Tr_if_same_cl[:, :] = (pred_cl - pred_cl.transpose(0, 1)).to(device)

    probs_Tr_if_iltj = torch.zeros((n_preds,n_preds)).to(device)

    probs_Tr_if_iltj[:, :] = ((pred_probs.transpose(0, 1) - pred_probs) < 0).bool().to(device)

    remove_boxes = []
    remove_boxes.append(torch.where((iou & (preds_Tr_if_same_cl == 0) & probs_Tr_if_iltj.bool()) == True)[0].tolist())
    remove_boxes.append(torch.where((iou & (preds_Tr_if_same_cl == 0) & (1 - probs_Tr_if_iltj).bool()) == True)[1].tolist())
    remove_bboxes_set = set([item for sublist in remove_boxes for item in sublist])

    valid_indxs = torch.ones(n_preds, dtype=torch.bool).to(device)

    valid_indxs[torch.Tensor(list(set(torch.Tensor(range(n_preds)).tolist()).difference(remove_bboxes_set))).long()] = False
    valid_indxs_mask = valid_indxs > 0.

    draw_image_with_ancs_xyxy(image_path,
                              torch.stack((xyxy[:, 1], xyxy[:, 0], xyxy[:, 3], xyxy[:, 2]), dim=1)[valid_indxs.bool()],
                              np.array(pred_labels)[valid_indxs_mask.cpu().numpy()].tolist(),
                              isTransform=True)

def subsampling(x, step):

    for d, s in enumerate(step):
        if s == None: continue
        x = x.index_select(dim=d, index=torch.arange(start=0, end=x.shape[d], step=s).long())
    return x


# Some augmentations are taken from
# https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection

def transform(image, boxes, labels):
    """
    Apply the transformations above.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :param split: one of 'TRAIN' or 'TEST', since different sets of transformations are applied
    :return: transformed image, transformed bounding box coordinates, transformed labels, transformed difficulties
    """

    # Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
    # see: https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]

    new_image = image
    new_boxes = boxes
    new_labels = labels
    # Skip the following operations for evaluation/testing
    # A series of photometric distortions in random order, each with 50% chance of occurrence, as in Caffe repo
    new_image = photometric_distort(new_image)

    # Convert PIL image to Torch tensor
    new_image = FT.to_tensor(new_image)

    # Expand image (zoom out) with a 50% chance - helpful for training detection of small objects
    # Fill surrounding space with the mean of ImageNet data that our base VGG was trained on
    if random.random() < .5:
        new_image, new_boxes = expand(new_image, boxes, filler=mean)

    # Randomly crop image (zoom in)
    new_image, new_boxes, new_labels = random_crop(new_image, new_boxes, new_labels)

    # Convert Torch tensor to PIL image
    new_image = FT.to_pil_image(new_image)

    # Flip image with a 50% chance
    if random.random() < .5:
        new_image, new_boxes = flip(new_image, new_boxes)

    return new_image, new_boxes, new_labels


def expand(image, boxes, filler):
    """
    Perform a zooming out operation by placing the image in a larger canvas of filler material.

    Helps to learn to detect smaller objects.

    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param filler: RBG values of the filler material, a list like [R, G, B]
    :return: expanded image, updated bounding box coordinates
    """
    # Calculate dimensions of proposed expanded (zoomed-out) image
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    # Create such an image with the filler
    filler = torch.FloatTensor(filler)  # (3)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)  # (3, new_h, new_w)
    # Note - do not use expand() like new_image = filler.unsqueeze(1).unsqueeze(1).expand(3, new_h, new_w)
    # because all expanded values will share the same memory, so changing one pixel will change all

    # Place the original image at random coordinates in this new image (origin at top-left of image)
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image

    # Adjust bounding boxes' coordinates accordingly
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(0)  # (n_objects, 4), n_objects is the no. of objects in this image

    return new_image, new_boxes


def random_crop(image, boxes, labels):
    """
    Performs a random crop in the manner stated in the paper. Helps to learn to detect larger and partial objects.

    Note that some objects may be cut out entirely.

    Adapted from https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :return: cropped image, updated bounding box coordinates, updated labels, updated difficulties
    """
    original_h = image.size(1)
    original_w = image.size(2)
    # Keep choosing a minimum overlap until a successful crop is made
    while True:
        # Randomly draw the value for minimum overlap
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])  # 'None' refers to no cropping

        # If not cropping
        if min_overlap is None:
            return image, boxes, labels

        # Try up to 50 times for this choice of minimum overlap
        # This isn't mentioned in the paper, of course, but 50 is chosen in paper authors' original Caffe repo
        max_trials = 50
        for _ in range(max_trials):
            # Crop dimensions must be in [0.3, 1] of original dimensions
            # Note - it's [0.1, 1] in the paper, but actually [0.3, 1] in the authors' repo
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            # Aspect ratio has to be in [0.5, 2]
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            # Crop coordinates (origin at top-left of image)
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom])  # (4)

            # Calculate Jaccard overlap between the crop and the bounding boxes
            overlap = get_jaccard_tensor11(crop.unsqueeze(0),
                                           boxes)  # (1, n_objects), n_objects is the no. of objects in this image
            overlap = overlap.squeeze(0)  # (n_objects)

            # If not a single bounding box has a Jaccard overlap of greater than the minimum, try again
            if overlap.max().item() < min_overlap:
                continue

            # Crop image
            new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)

            # Find centers of original bounding boxes
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.  # (n_objects, 2)

            # Find bounding boxes whose centers are in the crop
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                    bb_centers[:, 1] < bottom)  # (n_objects), a Torch uInt8/Byte tensor, can be used as a boolean index

            # If not a single bounding box has its center in the crop, try again
            if not centers_in_crop.any():
                continue

            # Discard bounding boxes that don't meet this criterion
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]

            # Calculate bounding boxes' new coordinates in the crop
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])  # crop[:2] is [left, top]
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])  # crop[2:] is [right, bottom]
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels


def flip(image, boxes):
    """
    Flip image horizontally.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: flipped image, updated bounding box coordinates
    """
    # Flip image
    new_image = FT.hflip(image)

    # Flip boxes
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image.width - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes


def photometric_distort(image):
    """
    Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.

    :param image: image, a PIL Image
    :return: distorted image
    """
    new_image = image

    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ is 'adjust_hue':
                # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                adjust_factor = random.uniform(0.5, 1.5)

            # Apply this distortion
            new_image = d(new_image, adjust_factor)

    return new_image


def get_map(det_boxes, det_classes, det_scores, gt_boxes, gt_classes):
    '''
        please go through this wonderful explanation of map in object detection.
        https://github.com/rafaelpadilla/Object-Detection-Metrics#average-precision
    :param det_boxes:
    :param det_classes:
    :param det_scores:
    :param gt_boxes:
    :param gt_classes:
    :return:
    '''
    TP = {}
    FP = {}

    for cls in range(20):
        TP[cls] = []
        FP[cls] = []

    Objs = [0] * 20

    x = len(det_boxes)

    det_images = [[i] * len(j) for i, j in zip(range(len(det_boxes[:x])), det_boxes[:x])]
    gt_images = [[i] * len(j) for i, j in zip(range(len(gt_boxes[:x])), gt_boxes[:x])]

    det_images = torch.tensor([item for sublist in det_images for item in sublist])
    gt_images = torch.tensor([item for sublist in gt_images for item in sublist])

    det_boxes_, det_classes_, det_scores_, gt_boxes_, gt_classes_ = torch.cat(det_boxes[:x]), torch.cat(det_classes[:x]),\
                                                                    torch.cat(det_scores[:x]), torch.cat(gt_boxes[:x]), torch.cat(gt_classes[:x])

    is_available = torch.ones(gt_boxes_.shape[0])
    cum_sum = np.array([i.shape[0] for i in gt_classes[:x]]).cumsum()
    counts_objects = np.array([i.shape[0] for i in gt_classes[:x]])

    for cls in range(20):

        gt_mask, det_mask = gt_classes_ == cls, det_classes_ == cls

        gt_images_cls, det_images_cls = gt_images[gt_mask], det_images[det_mask]
        n_objs_gt_class = gt_mask.long().sum()
        Objs[cls] += n_objs_gt_class

        det_bb_class, det_score_class = det_boxes_[det_mask], det_scores_[det_mask]

        det_score_class, sort_idx = det_score_class.sort(dim=0, descending=True)
        det_bb_class = det_bb_class[sort_idx]
        det_images_cls = det_images_cls[sort_idx]

        if det_bb_class.shape[0] == 0:
            continue

        for i, det_bb in enumerate(det_bb_class):

            image = det_images_cls[i]

            gt_all_cls_this_image = gt_classes_[gt_images == image]
            is_available_this_image = is_available[gt_images == image]

            cls_mask = gt_all_cls_this_image == cls
            gt_boxes_this = gt_boxes_[gt_images == image][cls_mask]

            is_available_this_image_cls = is_available_this_image[cls_mask]

            n_objs_gt_class = gt_boxes_this.shape[0]

            if n_objs_gt_class == 0:
                FP[cls].append(1.)
                TP[cls].append(0.)
                continue

            det_bb = det_bb.unsqueeze(0)
            iou = get_jaccard_tensor1(det_bb, gt_boxes_this)

            overlap, idx = iou.squeeze(0).max(dim=0)
            if overlap > 0.5:
                if is_available_this_image_cls[idx] == 1:
                    TP[cls].append(1.)
                    FP[cls].append(0.)
                    is_available[cum_sum[image] - counts_objects[image] + torch.arange(0, cls_mask.shape[0])[cls_mask][idx]] = 0
                else:
                    FP[cls].append(1.)
                    TP[cls].append(0.)
            else:
                FP[cls].append(1.)
                TP[cls].append(0.)

    all_precisions = {}
    for cls in range(20):
        cum_TP, cum_FP = np.array(TP[cls]).cumsum(), np.array(FP[cls]).cumsum()
        cum_precision = cum_TP / (cum_TP + cum_FP)
        cum_recall = cum_TP / Objs[cls]
        precisions = []
        for rec in torch.arange(0, 1.1, 0.1):
            recalls_mask = cum_recall >= rec
            if recalls_mask.sum() != 0:
                precisions.append(cum_precision[recalls_mask].max())
            else:
                precisions.append(0.)
        all_precisions[cls] = np.array(precisions).mean()
        print(cls, np.array(precisions).mean())

    return all_precisions
