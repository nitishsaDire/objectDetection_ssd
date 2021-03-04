import time
import torch
from Losses import *
from Util import *
import gc

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_model(cnn, optimizer, scheduler, dataloader, dataset_sizes, device, loadModel=False, num_epochs=200):
    since = time.time()
    lbb = []
    lc = []
    epoch_losses = {}
    cnn = cnn.to(device)
    for k in ['train', 'test']:
        epoch_losses[k] = []

    epoch = 0

    OLD_PATH = 'ssd_1'
    PATH = 'ssd_2'
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
        torch.cuda.empty_cache()
        gc.collect()

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            cnn = cnn.to(device)
            if phase == 'train':
                cnn.train()  # Set model to training mode
            else:
                cnn.eval()  # Set model to evaluate mode

            running_loss = 0.0
            # Iterate over data.
            count = 0
            it_begin = time.time()

            for inputs, classes, bboxes, indices in dataloader[phase]:

                inputs= inputs.to(device)
                classes = [c.to(device) for c in classes]
                bboxes = [b.to(device) for b in bboxes]
                bs = inputs.shape[0]

                if count % 400 == 0 and epoch%4==0:
                    print(phase)
                    random_index = torch.randint(0, bs, (1,))[0]
                    index = indices[random_index]
                    print(index)
                    draw_image_with_ancs_xyxy(all_images[phase][index], all_multi_bboxes[phase][index],
                                              all_multi_labels[phase][index])

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = cnn(inputs)

                    loss1, loss2 = ssd(outputs, classes, bboxes)

                    loss = loss1 + loss2
                    lbb.append(loss1.item())
                    lc.append(loss2.item())
                    if count % 400 == 0 and epoch%4==0:
                            cnn.eval()
                            l,c = cnn(inputs)
                            inference(l[random_index], c[random_index], index, phase=phase)
                            if phase=='train':cnn.train()
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                if count % 20 == 0:
                    te = time.time() - it_begin
                    print("IIterated over ", count, "LR=", get_lr(optimizer),
                          'Iteration Completed in {:.0f}m {:.0f}s'.format(
                              te // 60, te % 60), "l1", loss1.item(), ", l2", loss2.item())

                count += 1

            print(count)
            epoch_loss = running_loss / dataset_sizes[phase]

            epoch_losses[phase].append(epoch_loss)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

        torch.save({
            'epoch': epoch,
            'cnn_state_dict': cnn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'epoch_losses': epoch_losses
        }, PATH)

        te = time.time() - epoch_b
        print('epoch completed in {:.0f}m {:.0f}s'.format(
            te // 60, te % 60))

        print()
        print(epoch_losses)
        print('-' * 30)
        # plot_stats(epoch + 1, epoch_losses, epoch_accuracies)

    te = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(te // 60, te % 60))

    return cnn
