import itertools
import numpy as np
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
from shapely.geometry import Polygon
from matplotlib import patches
import time
import gc

from Losses import ssd, get_predictions_inf, get_predictions_noinf
from DataLists import all_images, all_multi_bboxes, all_multi_labels

grid_sizes_list = list(itertools.chain([0.25] * 144, [0.5] * 36, [1.] * 9))

transform = transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])

def denormalize(input):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return np.multiply(std,input) + mean
def get_diags_xyxy_from_diags_yxyx(anchors):
    x1 = anchors[:, 0]
    y1 = anchors[:, 1]
    x2 = anchors[:, 2]
    y2 = anchors[:, 3]
    # x1,y1,x2,y2 = x1.cpu().detach().numpy(), y1.cpu().detach().numpy(), x2.cpu().detach().numpy(),y2.cpu().detach().numpy()
    yx_diags = np.stack((y1, x1, y2, x2), axis=1)
    # xy_diags = np.stack((x-w/2, y-h/2, x+w/2, y+h/2), axis=1)
    return yx_diags


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


def get_diags_yxyx_from_xywh(anchors):
    x = anchors[:, 0]
    y = anchors[:, 1]
    w = anchors[:, 2]
    h = anchors[:, 3]
    x, y, w, h = x.cpu().detach().numpy(), y.cpu().detach().numpy(), w.cpu().detach().numpy(), h.cpu().detach().numpy()
    yx_diags = np.stack((y - h / 2, x - w / 2, y + h / 2, x + w / 2), axis=1)
    # xy_diags = np.stack((x-w/2, y-h/2, x+w/2, y+h/2), axis=1)
    return yx_diags


def draw_with_ancs(bboxes=None, scale=224, size=(4, 4)):
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


def get_jaccard(index, diags_ancs, scale=224, transform_img_sz=(224, 224)):
    diags_ancs = diags_ancs * scale
    # expects diags_ancs in yxyx format

    overlaps = []
    img_sz = get_img_sz(all_images[index])
    bboxes = scale_bboxes(standardize_bboxes(all_multi_bboxes[index], img_sz), transform_img_sz)

    for bbox in bboxes:
        overlaps.append([calculate_iou(anc, bbox) for anc in diags_ancs])
    return torch.Tensor(overlaps)


def get_jaccard_old(index, ancs_sz=(4, 4), scale=224, transform_img_sz=(224, 224)):
    diags_ancs = get_ancs_diag_coords(create_ancs_centre_coords(ancs_sz) * scale)

    overlaps = []
    img_sz = get_img_sz(all_images[index])
    bboxes = scale_bboxes(standardize_bboxes(all_multi_bboxes[index], img_sz), transform_img_sz)

    for bbox in bboxes:
        overlaps.append([calculate_iou(anc, bbox) for anc in diags_ancs])
    return torch.Tensor(overlaps)


class_to_label = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                  'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', "bg"]

label_to_class = {}
for idx, i in enumerate(class_to_label):
    label_to_class[i] = idx


def map_anc_to_bb(jacc, labels, threshold=0.4):
    axis1argmax = jacc.max(axis=1)[1]  # for row
    axis1max = jacc.max(axis=1)[0]
    axis0argmax = jacc.max(axis=0)[1]
    axis0max = jacc.max(axis=0)[0]

    # print(axis1argmax)
    for anc_idx in axis1argmax:
        axis0max[anc_idx] = 2

    for anc_idx, iou in enumerate(axis0max):
        if iou < threshold:
            axis0argmax[anc_idx] = -1
    return np.array(['bg' if val < 0 else labels[val] for val in axis0argmax]), axis0argmax
    # return axis0argmax, axis0argmax


def get_center_coords_from_diags(diags):
    cent_coords = []
    for diag in diags:
        d1y, d1x, d2y, d2x = diag

        w = d2x - d1x
        h = d2y - d1y
        cy = d1y + h / 2
        cx = d1x + w / 2
        cent_coords.append([cx, cy, w, h])
    return cent_coords


def get_center_coords_from_diags_tensor(diags):
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


def standardize_bboxes(bboxes, img_sz):
    w, h = img_sz
    if torch.is_tensor(bboxes):
        bboxes = bboxes.tolist()
    n_bboxes = []
    for bbox in bboxes:
        n_bboxes.append([bbox[0] / h, bbox[1] / w, bbox[2] / h, bbox[3] / w])
    return n_bboxes


def scale_bboxes(bboxes, scale_sz):
    w, h = scale_sz
    n_bboxes = []
    for bbox in bboxes:
        n_bboxes.append([bbox[0] * h, bbox[1] * w, bbox[2] * h, bbox[3] * w])
    return n_bboxes


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
    output = output.to(device)  # (bs,16*k,25)
    ancs_xywh = ancs_xywh.to(device)

    p_bbox = output[:, :, :4].to(device)
    p_labels = output[:, :, 4:].to(device)
    bs = p_bbox.shape[0]

    grid_sizes_list_t = torch.Tensor(grid_sizes_list).to(device)
    grid_sizes_list_st = torch.stack((grid_sizes_list_t, grid_sizes_list_t), dim=1)
    new_xy = torch.tanh(p_bbox)[:, :, :2] / 2 * grid_sizes_list_st + ancs_xywh[:, :2]
    # new_xy = torch.tanh(p_bbox)[:,:,:2]/2 * 0.25 + ancs_xywh[:,:2]
    new_wh = (torch.tanh(p_bbox)[:, :, 2:] / 2 + 1) * (ancs_xywh[:, 2:])
    # new_xy = ancs_xywh[:,:2]
    # new_wh = ancs_xywh[:,2:]
    # print(output.shape, ancs_xywh.shape, p_label.shape)

    # if apply_softmax == True:
    #     p_labels = soft(p_labels)
    # return torch.cat((new_xy, new_wh), dim=1), p_labels
    return torch.cat((new_xy, new_wh), dim=2), p_labels
    # (torch.Size([bs, 16*k, 4]), torch.Size([bs, 21*k, 4, 4]))


def c_to_diag(p):
    x_y = p[:, :2]
    w_h = p[:, 2:]
    return torch.cat((x_y - w_h / 2, x_y + w_h / 2), dim=1)


def draw_image_with_ancs(image_path, bboxes=None, labels=None, isTransform=True, scale=224, size=(4, 4)):
    img = Image.open(image_path)
    _, ax = plt.subplots(figsize=(5, 5))
    if isTransform == True:
        ax.imshow(denormalize(transform(img).permute(1, 2, 0)))
    else:
        ax.imshow(img)

    if bboxes != None:
        for idx, bbox in enumerate(bboxes):
            rect = patches.Rectangle((bbox[1], bbox[0]), bbox[3] - bbox[1], bbox[2] - bbox[0], linewidth=1,
                                     edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            if labels != None:
                ax.text(bbox[1], bbox[0], str(idx) + "_" + labels[idx], verticalalignment='top', color="r", fontsize=10,
                        weight='bold')
    elif labels != None:
        for idx, label in enumerate(labels):
            print(labels, len(labels), idx)
            ax.text(0, idx * 30, str(idx) + "_" + label, verticalalignment='top', color="r", fontsize=10, weight='bold')

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

    plt.show()


def draw_image_with_labelled_ancs(image_path, labels=None, isTransform=True, scale=224, size=(4, 4)):
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


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_model(unet, optimizer, scheduler, dataloader, dataset_sizes, device, loadModel=False, num_epochs=200):
    since = time.time()
    lbb = []
    lc = []
    epoch_losses = {}
    epoch_accuracies = {}
    cnn = unet.to(device)
    for k in ['train', 'val']:
        epoch_losses[k] = []
        epoch_accuracies[k] = []

    best_acc = 0.0
    epoch = 0
    # OLD_PATH = '/content/drive/MyDrive/ssd-v2-temp_val_corr_nF'
    # PATH = '/content/drive/MyDrive/ssd-v2-temp_val_corr_nF'

    OLD_PATH = '/content/drive/MyDrive/ssd-v3'
    PATH = '/content/drive/MyDrive/ssd-v3'
    if loadModel == True:
        checkpoint = torch.load(OLD_PATH)
        cnn.load_state_dict(checkpoint['cnn_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        cnn = cnn.to(device)
        epoch_losses = checkpoint['epoch_losses']

    for epoch in range(epoch, num_epochs):
        epoch_b = time.time()

        print(device)
        # print(torch.cuda.memory_summary(device=device, abbreviated=False)
        torch.cuda.empty_cache()
        gc.collect()

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            unet = unet.to(device)
            if phase == 'train':
                unet.train()  # Set model to training mode
            else:
                unet.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            count = 0
            it_begin = time.time()
            for inputs, labels, boxes, idx in dataloader[phase]:
                # print(labels.shape)
                # labels = labels.squeeze(1)
                inputs, labels, boxes, idx = inputs.to(device), labels.to(device), boxes.to(device), idx.to(device)

                if count % 40 == 0:
                    print(phase)
                    random_index = torch.randint(0, 64, (1,))[0]
                    index = idx[random_index]
                    print(index)
                    draw_image_with_ancs(all_images[index], scale_bboxes(
                        standardize_bboxes(all_multi_bboxes[index], get_img_sz(all_images[index])), (224, 224)),
                                         all_multi_labels[index], isTransform=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = cnn(inputs)
                    soft = nn.Softmax(dim=1)

                    loss1, loss2 = ssd(outputs, labels, boxes, idx, count % 100 == 0)
                    loss = loss1 + loss2
                    lbb.append(loss1.item())
                    lc.append(loss2.item())
                    if count % 40 == 0:
                        with torch.no_grad():
                            print(phase)
                            print("predictions")

                            print(random_index)
                            index = idx[random_index]
                            p, q = get_predictions_inf(outputs[random_index].unsqueeze(0), index, 'cuda')

                            pred_bboxes = p
                            print(q)
                            print(lbb)
                            print(lc)
                            pred_labels = [class_to_label[q_.argmax().item()] + str(round(q_.max().item(), 2)) for q_ in
                                           q]
                            print(pred_labels)
                            xyxy = c_to_diag(p * 224)

                            draw_image_with_ancs(all_images[index],
                                                 torch.stack((xyxy[:, 1], xyxy[:, 0], xyxy[:, 3], xyxy[:, 2]), dim=1),
                                                 pred_labels,
                                                 isTransform=True)

                            p, q = get_predictions_noinf(outputs[random_index].unsqueeze(0), index, 'cuda')

                            pred_bboxes = p
                            # print(q)
                            # print(lbb)
                            # print(lc)
                            pred_labels = [class_to_label[q_.argmax().item()] + str(round(q_.max().item(), 2)) for q_ in
                                           q]
                            # print(pred_labels)
                            xyxy = c_to_diag(p * 224)

                            draw_image_with_ancs(all_images[index],
                                                 torch.stack((xyxy[:, 1], xyxy[:, 0], xyxy[:, 3], xyxy[:, 2]), dim=1),
                                                 pred_labels,
                                                 isTransform=True)
                            # ll = [class_to_label[i] for i in r.argmax(dim=1)]
                            # print(r, r.shape)
                            # print(r.argmax(dim=1))
                            # im = all_images[index]
                            # draw_image_with_labelled_ancs(im, ll)

                    # print(loss)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += 10
                if count % 20 == 0:
                    time_elapsed = time.time() - it_begin
                    print("IIterated over ", count, "LR=", get_lr(optimizer),
                          'Iteration Completed in {:.0f}m {:.0f}s'.format(
                              time_elapsed // 60, time_elapsed % 60), "l1", loss1.item(), ", l2", loss2.item())

                count += 1

            print(count)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            epoch_losses[phase].append(epoch_loss)
            epoch_accuracies[phase].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

        torch.save({
            'epoch': epoch,
            'cnn_state_dict': cnn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'epoch_losses': epoch_losses
        }, PATH)

        time_elapsed = time.time() - epoch_b
        print('epoch completed in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        print()
        print(epoch_losses)
        print(epoch_accuracies)
        print('-' * 30)
        # plot_stats(epoch + 1, epoch_losses, epoch_accuracies)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return unet
