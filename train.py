from torch.utils.data import DataLoader
from Dataset import MultiImageMultiBBoxDataset, collate_fn
from Model import  SSD_300
import torch
from train_function import train_model
from Util import all_images, all_multi_bboxes, all_multi_labels, device

multiImageMultiBBoxDataset = {}
dataset_sizes = {}
dataloader = {}

for phase in ['train']:
    print(phase, len(all_images[phase]))
    multiImageMultiBBoxDataset[phase] = MultiImageMultiBBoxDataset(all_images[phase][:2500], all_multi_bboxes[phase][:2500], all_multi_labels[phase][:2500])
    dataset_sizes[phase] = len(multiImageMultiBBoxDataset[phase])
    dataloader[phase] = DataLoader(multiImageMultiBBoxDataset[phase], batch_size=10, shuffle=True, num_workers=2, collate_fn = collate_fn)

for phase in ['test']:
    print(phase, len(all_images[phase][:500]))
    multiImageMultiBBoxDataset[phase] = MultiImageMultiBBoxDataset(all_images[phase][:500], all_multi_bboxes[phase][:500], all_multi_labels[phase][:500])
    dataset_sizes[phase] = len(multiImageMultiBBoxDataset[phase])
    dataloader[phase] = DataLoader(multiImageMultiBBoxDataset[phase], batch_size=10, shuffle=True, num_workers=2, collate_fn = collate_fn)


cnn = SSD_300().to(device)
biases = list()
not_biases = list()
for param_name, param in cnn.named_parameters():
    if param.requires_grad:
        if param_name.endswith('.bias'):
            biases.append(param)
        else:
            not_biases.append(param)

lr = 1e-5
optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2*lr}, {'params': not_biases}],
                            lr=lr, momentum=0.9, weight_decay=5e-4)

exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

train_model(cnn, optimizer, exp_lr_scheduler, dataloader, dataset_sizes, device, loadModel=True, num_epochs=1000)
