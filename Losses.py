import torch
import torch.nn as nn
from Util import *
import torch.nn.functional as F
import numpy as np
from DataLists import all_multi_labels



def get_predictions_inf(outputs, index, device, k=9, threshold=0.75):
    ancs_xywh = create_ancs_xywh_zoom_ratio()

    p_ancs_xywh_, p_labels_ = get_p_bbox_labels_za(outputs, ancs_xywh, device, apply_softmax=True)

    p_ancs_xywh, p_labels = p_ancs_xywh_[0], p_labels_[0]
    # soft = nn.Softmax(dim=1)
    p_labels = torch.sigmoid(p_labels)

    pred_probs, pred_classes = p_labels.max(dim=1)
    # print(pred_probs)
    # print(pred_classes)

    pos_ancs = pred_probs >= pred_probs.max().item() * threshold

    return p_ancs_xywh[pos_ancs], p_labels[pos_ancs]


def get_predictions_noinf(outputs, index, device, k=9):
    ancs_xywh = create_ancs_xywh_zoom_ratio()

    p_ancs_xywh_, p_labels_ = get_p_bbox_labels_za(outputs, ancs_xywh, device, apply_softmax=True)

    p_ancs_xywh, p_labels = p_ancs_xywh_[0], p_labels_[0]
    jacc = get_jaccard(index, get_diags_yxyx_from_xywh(ancs_xywh))
    map_anc_to_labels, map_anc_to_idx = map_anc_to_bb(jacc, all_multi_labels[index])
    # print(map_anc_to_idx)
    pos_ancs = map_anc_to_idx >= 0
    pos_ancs = pos_ancs.to(device)

    return p_ancs_xywh[pos_ancs], p_labels[pos_ancs]


def ssd(outputs, tr_labels, tr_bboxs, tr_indxs, print_it=False, k=9):
    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

    lbb = 0.0
    lc = 0.0
    ancs_xywh = create_ancs_xywh_zoom_ratio()

    p_ancs_xywh_s, p_labels = get_p_bbox_labels_za(outputs, ancs_xywh, device, apply_softmax=True)
    # print(p_ancs_xywh.shape)

    count = 0
    for p_ancs_xywh, p_label, tr_bbox, tr_label, tr_idx in zip(p_ancs_xywh_s, p_labels, tr_bboxs, tr_labels, tr_indxs):
        lc_, lbb_ = ssd1(p_ancs_xywh, p_label, remove_padding(tr_bbox), tr_label, tr_idx, device,
                         printit=(count == 10) and (print_it))
        # print(lc_, lbb_)
        count += 1
        lbb += lbb_
        lc += lc_
    if print_it == True:
        print(lbb, lc)

    return lbb, lc


def ssd1(p_ancs_xywh, p_label, tr_bbox, tr_label, index, device, k=9, printit=False):
    p_ancs_xywh = p_ancs_xywh.to(device)
    jacc = get_jaccard(index, get_diags_yxyx_from_xywh(create_ancs_xywh_zoom_ratio()))
    map_anc_to_labels, map_anc_to_idx = map_anc_to_bb(jacc, all_multi_labels[index])
    # print(map_anc_to_idx)
    pos_ancs = map_anc_to_idx >= 0
    pos_ancs = pos_ancs.to(device)

    map_anc_to_idx_pos = map_anc_to_idx[pos_ancs]
    map_anc_to_idx_pos = map_anc_to_idx_pos.to(device)
    # print(index)

    map_anc_to_class = torch.Tensor(np.array([label_to_class[v] if v != 'bg' else -1 for v in map_anc_to_labels]))

    map_anc_to_class[map_anc_to_class < 0] = 20

    if printit == True:
        print([class_to_label[int(i)] for i in map_anc_to_class])
    # gets example_idx of each of the positive anc, it will be used to get gt_bb
    targets = torch.nn.functional.one_hot(map_anc_to_class.to(torch.int64), 21).float()
    # print(p_label.shape)
    predss = p_label.to(device)

    if printit == True:
        print([class_to_label[i] for i in predss.max(axis=1)[1]])
    # print("preds",predss.argmax(axis=0))
    # print("targets", targets.argmax(axis=0))

    p, t = predss[:, :-1], targets[:, :-1].to(device)
    loss_f = BCE_Loss()
    c_loss = loss_f(p, t)
    # print(map_anc_to_idx_pos)
    # print(tr_bbox)
    tr_bbox = tr_bbox.to(device)
    gg = tr_bbox[map_anc_to_idx_pos]
    # print(pos_ancs.shape, p_ancs_xywh.shape)
    pp = p_ancs_xywh[pos_ancs]
    # print("gg",gg)
    # print("pp",pp)
    diff = gg.to(device) - pp.to(device)
    # print(diff)
    l1_loss = diff.abs().mean()
    # print(c_loss, l1_loss)

    return c_loss, l1_loss


class BCE_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 20

    def forward(self, p, t):
        # w = self.get_weight(p,t)
        return F.binary_cross_entropy_with_logits(p, t)
        # return F.binary_cross_entropy_with_logits(p, t, w.detach(),
        #                     size_average=False)/self.num_classes

    def get_weight(self, x, t):
        alpha, gamma = 0.25, 2.
        p = torch.sigmoid(x)
        pt = p * t + (1 - p) * (1 - t)
        w = alpha * t + (1 - alpha) * (1 - t)
        return w * (1 - pt).pow(gamma)


