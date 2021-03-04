from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import torch
from sklearn import preprocessing


class ImageBBoxDataset(Dataset):
    def __init__(self, images_list, bbox, labels_list):
        self.images_list = images_list
        self.bbox = bbox
        self.labels_list = labels_list
        self.imageDataset = ImageDataset(self.images_list, self.labels_list)
        # self.size = (224,224)
        # self.centre_crop = transforms.CenterCrop(self.size)

    def get_bbox(self, index):
        '''Assuming video has label name in its name'''
        return torch.Tensor(self.bbox[index])

    def __getitem__(self, index):
        image, label = self.imageDataset[index]
        bbox = self.get_bbox(index)
        return image, label, bbox

    def __len__(self):
        return len(self.images_list)


class BBoxDataset(Dataset):
    def __init__(self, images_list, bbox):
        self.images_list = images_list
        self.bbox = bbox

    def get_bbox(self, index):
        return torch.Tensor(self.bbox[index])

    def get_image(self, index):
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        image = Image.open(self.images_list[index])
        image = transform(image)

        return image

    def __getitem__(self, index):
        image, bbox = self.get_image(index), self.get_bbox(index)
        return image, bbox

    def __len__(self):
        return len(self.images_list)


class ImageDataset(Dataset):
    def __init__(self, images_list, labels_list):
        self.images_list = images_list
        self.labels_list = labels_list
        self.le = preprocessing.LabelEncoder()
        self.cats = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                     'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                     'tvmonitor']
        # pascal dataset
        self.le.fit(self.cats)

    @classmethod
    def getLabelFromIndex(cls, c):
        cls.le.transform(c)

    def get_label(self, index):
        '''Assuming video has label name in its name'''
        return self.le.transform(list(self.labels_list[index].split(" ")))[0]

    def get_image(self, index):
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])

        image = Image.open(self.images_list[index])
        image = transform(image)

        return image

    def __getitem__(self, index):
        image, label = self.get_image(index), self.get_label(index)
        return image, label

    def __len__(self):
        return len(self.images_list)


class MultiImageDataset(ImageDataset):
    def __init__(self, images_list, multi_labels_list):
        super().__init__(images_list, multi_labels_list)
        self.multi_labels_list = multi_labels_list

    # @override
    def get_label(self, index):
        # print(self.le.transform(list(self.multi_labels_list[index])))
        return one_zero(self.le.transform(list(self.multi_labels_list[index])), 20)


def get_max_bbox_per_image(l):
    lengths = np.array([len(i) for i in l])
    return lengths.max(), lengths.argmax()


class MultiImageMultiBBoxDataset():
    def __init__(self, images_list, multi_labels_list, multi_bbox_list):
        self.images_list = images_list
        self.multi_labels_list = multi_labels_list
        self.multi_bbox_list = multi_bbox_list
        self.max_card_bbox = get_max_bbox_per_image(self.multi_bbox_list)[0]
        self.multiImageDataset = MultiImageDataset(self.images_list, self.multi_labels_list)

    def get_bbox(self, index):
        sl = [[0.0, 0.0, 0.0, 0.0]] * self.max_card_bbox
        for idx, bbox in enumerate(self.multi_bbox_list[index]):
            sl[idx] = bbox
        return torch.Tensor(sl)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        image, label = self.multiImageDataset[index]
        bbox = self.get_bbox(index)
        return image, label, get_center_coords_from_diags_tensor(
            torch.Tensor(standardize_bboxes(bbox, get_img_sz(self.images_list[index])))), index

