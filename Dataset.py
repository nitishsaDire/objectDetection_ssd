from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch
from Util import label_to_class, transform
from PIL import Image

class MultiImageMultiBBoxDataset(Dataset):
    def __init__(self, all_images, all_multi_bboxes, all_multi_labels, all_difficulties, all_indices, isTest=False, keep_difficult = False):
        self.images_list = all_images
        self.transform = transforms.Compose([   transforms.Resize((300, 300)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ])
        self.multi_labels_list = all_multi_labels
        self.isTest = isTest
        self.multi_bbox_list = all_multi_bboxes
        self.all_difficulties = [torch.tensor(i) for i in all_difficulties]
        self.keep_difficult = keep_difficult
        self.all_indices = all_indices

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        image = Image.open(self.images_list[index])
        c = torch.Tensor([label_to_class[i] for i in self.multi_labels_list[index]])
        bboxes = torch.Tensor(self.multi_bbox_list[index])

        if self.keep_difficult == False:
            bboxes = bboxes[self.all_difficulties[index]==0]
            c = c[self.all_difficulties[index]==0]

        if self.isTest==False:
            image, bboxes, c = transform(image, bboxes, c)
        w, h = image.size
        standardized_bbox = bboxes/torch.FloatTensor([w, h, w, h]).unsqueeze(0)
        image = self.transform(image)

        return image, c, standardized_bbox, self.all_indices[index]

def collate_fn(batch):
    images = []
    classes = []
    standardized_bboxes = []
    indices = []

    for b in batch:
        images.append(b[0])
        classes.append(b[1])
        standardized_bboxes.append(b[2])
        indices.append(b[3])

    return torch.stack(images, dim=0), classes, standardized_bboxes, indices
