import torch
import torch.nn as nn
from Util import *
import torch.nn.functional as F

ancs_xywh = create_priors_ssd300()
ancs_xyxy = xywh_to_xyxy(ancs_xywh)
use_cuda = torch.cuda.is_available()  # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

def inference(l_, c_, index, top_k = 200, phase = 'train'):
    """
    :param l_: predicts offsets (gcxgcy) (8732,4)
    :param c_: (8732,21)
    :param index:
    :return:
    """

    all_pred_bboxes = []
    all_pred_prob = []
    all_pred_class = []

    all_bboxs_xywh = gcxgcy_to_cxcy(l_, ancs_xywh).to(device)

    all_probs = F.softmax(c_, dim=1)

    for c in range(20):
        min_score = 0.2

        all_bbox_class = all_bboxs_xywh.to(device)
        all_prob_class = all_probs[:, c].to(device)

        pos_ancs = all_prob_class >= min_score
        if pos_ancs.sum().item() == 0: continue

        all_bbox_class = all_bbox_class[pos_ancs].to(device)
        all_prob_class = all_prob_class[pos_ancs].to(device)

        all_prob_class, sort_ind = all_prob_class.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
        all_bbox_class = all_bbox_class[sort_ind].to(device)  # (n_min_score, 4)

        jacc = get_jaccard_tensor1(xywh_to_xyxy(all_bbox_class), xywh_to_xyxy(all_bbox_class)).to(device)
        suppress = torch.zeros((pos_ancs.sum()), dtype=torch.bool).to(device)  # (n_qualified)

        for box in range(all_bbox_class.size(0)):
            # If this box is already marked for suppression
            if suppress[box] == 1:
                continue

            # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
            # Find such boxes and update suppress indices
            suppress = torch.max(suppress, jacc[box,:] >= 0.45)
            # The max operation retains previously suppressed boxes, like an 'OR' operation

            # Don't suppress this box, even though it has an overlap of 1 with itself
            suppress[box] = 0


        all_pred_bboxes.append(all_bbox_class[~suppress].cpu().detach())
        all_pred_prob.append(all_prob_class[~suppress].cpu().detach())
        all_pred_class.append(torch.Tensor([c] * (~suppress).sum().item()))

    if len(all_pred_bboxes)==0:
        return
    sum=0
    for i in all_pred_bboxes:
        sum+=len(i)
    print(sum)

    print(len(all_pred_bboxes), len(all_pred_prob), len(all_pred_class))

    all_pred_bboxes_xyxy = xywh_to_xyxy(torch.cat(all_pred_bboxes)).to(device)
    all_pred_prob= torch.cat(all_pred_prob).to(device)
    all_pred_class = torch.cat(all_pred_class).long().to(device)

    print(all_pred_bboxes_xyxy.shape, all_pred_prob.shape, all_pred_class.shape)

    if all_pred_bboxes_xyxy.shape[0] > top_k:
        all_pred_prob, sort_ind = all_pred_prob.sort(dim=0, descending=True)
        all_pred_prob = all_pred_prob[:top_k]  # (top_k)
        all_pred_bboxes_xyxy = all_pred_bboxes_xyxy[sort_ind][:top_k]  # (top_k, 4)
        all_pred_class = all_pred_class[sort_ind][:top_k]  # (top_k)

    lbs_set = set([class_to_label[i] for i in all_pred_class.tolist()])
    lbs = list([class_to_label[i] for i in all_pred_class.tolist()])
    for i in lbs_set: print(i, lbs.count(i), end =" ")

    w,h = get_img_sz(all_images[phase][index])

    draw_image_with_ancs_xyxy(all_images[phase][index],
                              all_pred_bboxes_xyxy * torch.FloatTensor([w, h, w, h]).unsqueeze(0).to(device),
                              [class_to_label[all_pred_class[i]] for i in range(all_pred_class.shape[0])],
                              all_pred_prob
                              )

def ssd(outputs, tr_classes, tr_bboxs):
    """
    :param outputs: (bs, 8732, 4), (bs, 8732, 21)
    :param tr_classes:
    :param tr_bboxs:
    :param tr_indxs:
    :return:
    """
    bs = len(tr_bboxs)
    lbb = 0.0
    lc = 0.0
    pred_bb_offsets, pred_class_scores = outputs

    for pred_bb_offset, pred_class_score, tr_bbox, tr_class in zip(pred_bb_offsets, pred_class_scores, tr_bboxs, tr_classes):
        lc_, lbb_ = ssd1(pred_bb_offset, pred_class_score, tr_bbox, tr_class)
        lbb += lbb_
        lc += lc_
    return lbb/bs, lc/bs


def ssd1(pred_bb_offset, pred_class_score, tr_bbox, tr_class):
    """
    :param pred_bb_offset: (8732,4)
    :param pred_class_score: (8732,21)
    :param tr_bbox:
    :return:
    """
    smooth_l1 = nn.L1Loss()

    jacc = get_jaccard_tensor1(tr_bbox, ancs_xyxy) # (#bboxes,8732)
    map_prior_to_class, map_prior_to_obj = map_prior_to_bb(jacc, tr_class)
    gt_locations = xyxy_to_xywh(tr_bbox[map_prior_to_obj])

    true_classes = map_prior_to_class.to(torch.int64).to(device)    #(8732),
    pos_ancs = (true_classes != 20).to(device)                      #(8732)

    cce = F.cross_entropy(pred_class_score.to(device), true_classes, reduce=False)
    cce1 = cce.clone()                                              #(8732)
    cce1[pos_ancs] = 0.0
    cce1, _ = cce1.sort(descending=True)
    c_loss = (cce[pos_ancs].sum() + cce1[:3*pos_ancs.sum()].sum()) / pos_ancs.sum().float()

    loc_loss = smooth_l1(pred_bb_offset[pos_ancs, :], get_offsets_coords(gt_locations[pos_ancs, :], ancs_xywh[pos_ancs, :]).to(device))

    return c_loss, loc_loss


class Focal_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 20

    def forward(self, p, t):
        w = self.get_weight(p,t)
        return F.binary_cross_entropy_with_logits(p, t, w.detach())

    def get_weight(self, x, t):
        alpha, gamma = 0.25, 2.
        p = torch.sigmoid(x)
        pt = p * t + (1 - p) * (1 - t)
        w = alpha * t + (1 - alpha) * (1 - t)
        return w * (1 - pt).pow(gamma)
