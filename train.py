from torch.utils.data import DataLoader
from Dataset import MultiImageMultiBBoxDataset, collate_fn
from Model import  SSD_300
import torch
from train_function import train_model
from Util import all_images, all_multi_bboxes, all_multi_labels, all_difficulties, device

multiImageMultiBBoxDataset = {}
dataset_sizes = {}
dataloader = {}

n = len(all_images['train'])
torch.random.manual_seed(10)
test_indices = torch.randint(0, n, (int(n*0.1),)).tolist()
tr_indices = []
for i in range(n):
    if i not in test_indices:
        tr_indices.append(i)

print(n, len(tr_indices), len(test_indices))
for phase in ['train']:
    print(phase, len([all_images[phase][i] for i in tr_indices]))
    multiImageMultiBBoxDataset[phase] = MultiImageMultiBBoxDataset([all_images[phase][i] for i in tr_indices],
                                                                   [all_multi_bboxes[phase][i] for i in tr_indices],
                                                                   [all_multi_labels[phase][i] for i in tr_indices],
                                                                   [all_difficulties[phase][i] for i in tr_indices],
                                                                   tr_indices)
    dataset_sizes[phase] = len(multiImageMultiBBoxDataset[phase])
    dataloader[phase] = DataLoader(multiImageMultiBBoxDataset[phase], batch_size=20, shuffle=True, num_workers=2, collate_fn = collate_fn)

for phase in ['test']:
    print(phase, len([all_images['train'][i] for i in test_indices]))
    multiImageMultiBBoxDataset[phase] = MultiImageMultiBBoxDataset([all_images['train'][i] for i in test_indices],
                                                                   [all_multi_bboxes['train'][i] for i in test_indices],
                                                                   [all_multi_labels['train'][i] for i in test_indices],
                                                                   [all_difficulties['train'][i] for i in test_indices],
                                                                   test_indices,
                                                                   isTest=True)
    dataset_sizes[phase] = len(multiImageMultiBBoxDataset[phase])
    dataloader[phase] = DataLoader(multiImageMultiBBoxDataset[phase], batch_size=20, shuffle=True, num_workers=2, collate_fn = collate_fn)


cnn = SSD_300().to(device)
biases = list()
not_biases = list()
for param_name, param in cnn.named_parameters():
    if param.requires_grad:
        if param_name.endswith('.bias'):
            biases.append(param)
        else:
            not_biases.append(param)

lr = 1e-4
optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2*lr}, {'params': not_biases}],
                            lr=lr, momentum=0.9, weight_decay=5e-4)

exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

train_model(cnn, optimizer, exp_lr_scheduler, dataloader, dataset_sizes, device, lr, loadModel=True, num_epochs=1000)
