from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch
from Util import  get_img_sz, label_to_class
from PIL import Image

class MultiImageMultiBBoxDataset(Dataset):
    def __init__(self, all_images, all_multi_bboxes, all_multi_labels):
        self.images_list = all_images
        self.transform = transforms.Compose([   transforms.Resize((300, 300)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ])
        self.multi_labels_list = all_multi_labels
        self.multi_bbox_list = all_multi_bboxes

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        image = self.transform(Image.open(self.images_list[index]))
        c = torch.Tensor([label_to_class[i] for i in self.multi_labels_list[index]])
        w, h = get_img_sz(self.images_list[index])
        standardized_bbox = torch.Tensor(self.multi_bbox_list[index]) / torch.FloatTensor([w, h, w, h]).unsqueeze(0)
        return image, c, standardized_bbox, index

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
