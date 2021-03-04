from torch.utils.data import DataLoader
from Dataset import MultiImageMultiBBoxDataset
from DataLists import all_images, all_multi_labels, all_multi_bboxes
from Model import  SSD_resnet34
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import time
import gc


multiLabelImageDataset = MultiImageMultiBBoxDataset(all_images, all_multi_labels, all_multi_bboxes)

train_size = int(0.9 * len(multiLabelImageDataset))
val_size = len(multiLabelImageDataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(multiLabelImageDataset, [train_size, val_size])
#
multiLabelImageDatasetDict = {"train": train_dataset, "val":val_dataset}

dataset_sizes = {x: len(multiLabelImageDatasetDict[x]) for x in ['train', 'val']}

dataloader = {x: DataLoader(multiLabelImageDatasetDict[x], batch_size=64,
                        shuffle=True, num_workers=2) for x in ['train', 'val']}


use_cuda = torch.cuda.is_available()  # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
print(device)

cnn = SSD_resnet34(20, k=9).to(device)


optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
model = train_model(cnn, optimizer, exp_lr_scheduler, dataloader, dataset_sizes, device, loadModel=False, num_epochs=30)


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
