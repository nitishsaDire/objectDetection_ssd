import torch
import torch.nn as nn
from torchvision import models
import torchvision
from Util import subsampling
import torch.nn.functional as F


# created by Nitish Sandhu
# date 05/feb/2021

class SSD_resnet34(nn.Module):
    def __init__(self, n_classes, dropout_p=0.4, k=3):
        super().__init__()
        use_cuda = torch.cuda.is_available()  # check if GPU exists
        self.device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
        # self.bias = bias
        self.k = k
        self.n_classes = n_classes
        self.dropout_p = dropout_p
        self.resnet = models.resnet34(pretrained=True)
        self.resnet_layers = list(self.resnet.children())
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.4)

        self.seq1 = nn.Sequential(*self.resnet_layers[0:3])
        self.seq2 = nn.Sequential(*self.resnet_layers[3:5])
        self.seq3 = nn.Sequential(*self.resnet_layers[5])
        self.seq4 = nn.Sequential(*self.resnet_layers[6])
        self.seq5 = nn.Sequential(*self.resnet_layers[7])

        self.conv2d_0 = self.conv2d(512, 256, kernel=3, stride=1, padding=1)
        self.conv2d_01 = self.conv2d(256, 256, kernel=3, stride=2, padding=1)
        self.conv2d_02 = self.conv2d(256, 256, kernel=3, stride=2, padding=1)
        self.conv2d_03 = self.conv2d(256, 256, kernel=3, stride=2, padding=1)

        self.conv2d_02_bb4 = self.conv2d_final(256, 4 * k, kernel=3, stride=1, padding=1)
        self.conv2d_02_c4 = nn.Conv2d(256, (self.n_classes + 1) * k, kernel_size=3, stride=1, padding=1)
        self.conv2d_02_c4.bias.data.zero_().add_(-2)

        self.conv2d_02_bb2 = self.conv2d_final(256, 4 * k, kernel=3, stride=1, padding=1)
        self.conv2d_02_c2 = nn.Conv2d(256, (self.n_classes + 1) * k, kernel_size=3, stride=1, padding=1)
        self.conv2d_02_c2.bias.data.zero_().add_(-2)

        self.conv2d_02_bb1 = self.conv2d_final(256, 4 * k, kernel=3, stride=1, padding=1)
        self.conv2d_02_c1 = nn.Conv2d(256, (self.n_classes + 1) * k, kernel_size=3, stride=1, padding=1)
        self.conv2d_02_c1.bias.data.zero_().add_(-2)

        # self.adaptivePooling2 = nn.AdaptiveAvgPool2d((2,2))
        # self.adaptivePooling1 = nn.AdaptiveAvgPool2d((1,1))

        self.bn4 = nn.BatchNorm2d((self.n_classes + 1) * k)
        self.bn2 = nn.BatchNorm2d((self.n_classes + 1) * k)
        self.bn1 = nn.BatchNorm2d((self.n_classes + 1) * k)

    def conv2d(self, in_channels, out_channels, kernel=1, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=0.4),
        ).to(self.device)

    def conv2d_final(self, in_channels, out_channels, kernel=1, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=0.4),
            # nn.ReLU()
        ).to(self.device)

    def forward(self, x):
        """

        :param x: shape (bs, 3, 224, 224)
        :return:
        """
        # for p in self.seq1.parameters():
        #     print(p.norm().item())

        with torch.no_grad():
            x1e = self.seq1(x)  # 112x112, 64
            x2e = self.seq2(x1e)  # 56x56, 64
            x3e = self.seq3(x2e)  # 28x28, 128
            x4e = self.seq4(x3e)  # 14x14, 256
            x5e = self.seq5(x4e)  # 7x7, 512

        x5e = self.drop(self.relu(x5e))
        x6e = self.conv2d_0(x5e)  # 7x7, 256

        x7e = self.conv2d_01(x6e)  # 4x4, 256

        f4b = self.conv2d_02_bb4(x7e)  # 4x4, (4)*k
        f4c = self.conv2d_02_c4(x7e)  # 4x4, (c+1)*k
        # print(x7e.shape)
        x8e = self.conv2d_01(x7e)

        f2b = self.conv2d_02_bb2(x8e)  # 2x2, (4)*k
        f2c = self.conv2d_02_c2(x8e)  # 2x2, (c+1)*k

        x9e = self.conv2d_02(x8e)

        f1b = self.conv2d_02_bb1(x9e)  # 1x1, (4)*k
        f1c = self.conv2d_02_c1(x9e)  # 1x1, (c+1)*k

        # print(x5e.shape, x6e.shape, x7e.shape)

        # f2b = self.conv2d_02_bb2(self.adaptivePooling2(x7e))                                      # 2x2, (4)*k
        # f2c = self.bn2(self.conv2d_02_c2(self.adaptivePooling2(x7e)))                             # 2x2, (c+1)*k

        # f1b = self.conv2d_02_bb1(self.adaptivePooling1(x7e))                                      # 1x1, (4)*k
        # f1c = self.bn1(self.conv2d_02_c1(self.adaptivePooling1(x7e)))                             # 1x1, (c+1)*k
        # print(f4b.shape, f4c.shape,  torch.cat((f4b,f4c), dim=1).shape)
        f4b = f4b.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 4)
        # bs, 16*9, 4

        f4c = f4c.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 21)
        # bs, 16*9, 21

        f2b = f2b.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 4)
        f2c = f2c.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 21)

        f1b = f1b.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 4)
        f1c = f1c.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 21)

        return torch.cat((f4b, f2b, f1b), dim=1), torch.cat((f4c, f2c, f1c), dim=1)

class SSD_300(nn.Module):
    def __init__(self):
        super(SSD_300, self).__init__()
        self.model = models.vgg16(pretrained=True)
        self.rescaling_conv_4_3 = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # there are 512 channels in conv4_3_feats
        nn.init.constant_(self.rescaling_conv_4_3, 20.)

        self.conv_4_3 = nn.Sequential(
                                *self.model.features[0:16],
                                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True),
                                *self.model.features[17:23]
        )
        self.seq5 = nn.Sequential(
                            *self.model.features[23:30],
                            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        )

        self.fc6 = subsampling(list(self.model.classifier[0].parameters())[0].view(4096, 512, 7, 7), [4, None, 3, 3])
        self.fc6_b = subsampling(list(self.model.classifier[0].parameters())[1], [4])

        # self.conv_fc6 = nn.Conv2d(512, 1024, 3, padding=1)
        self.conv_fc6 = nn.Conv2d(512, 1024, 3, padding=4, dilation=4)  # gives more field of view aka receptive field, thats why atrous
                                                                        # converges faster. For k=3, p=d should be satisfied.

        self.conv_fc6.weight = torch.nn.Parameter(self.fc6)
        self.conv_fc6.bias = torch.nn.Parameter(self.fc6_b)

        # self.fc7 = subsampling(list(self.model.classifier[3].parameters())[0].view(4096, 1024, 2, 2), [4, None, 2, 2])
        self.fc7 = subsampling(list(self.model.classifier[3].parameters())[0].view(4096, 4096, 1, 1), [4, 4, None, None])
        self.fc7_b = subsampling(list(self.model.classifier[3].parameters())[1], [4])

        self.conv_fc7 = nn.Conv2d(1024, 1024, 1)
        self.conv_fc7.weight = torch.nn.Parameter(self.fc7)
        self.conv_fc7.bias = torch.nn.Parameter(self.fc7_b)
        self.seq7 = nn.Sequential(self.conv_fc6, nn.ReLU(), self.conv_fc7, nn.ReLU())
        self.seq8 = nn.Sequential(nn.Conv2d(1024, 256, 1), nn.ReLU(), nn.Conv2d(256, 512, 3, 2, padding=1), nn.ReLU())
        self.seq9 = nn.Sequential(nn.Conv2d(512, 128, 1), nn.ReLU(), nn.Conv2d(128, 256, 3, 2, padding=1), nn.ReLU())
        self.seq10 = nn.Sequential(nn.Conv2d(256, 128, 1), nn.ReLU(), nn.Conv2d(128, 256, 3, 1), nn.ReLU())
        self.seq11 = nn.Sequential(nn.Conv2d(256, 128, 1), nn.ReLU(), nn.Conv2d(128, 256, 3, 1), nn.ReLU())

        self.c_4_bb = nn.Conv2d(512, 16, 3, padding=1)
        self.c_4_cl = nn.Conv2d(512, 21 * 4, 3, padding=1)

        self.c_7_bb = nn.Conv2d(1024, 24, 3, padding=1)
        self.c_7_cl = nn.Conv2d(1024, 21 * 6, 3, padding=1)

        self.c_8_bb = nn.Conv2d(512, 24, 3, padding=1)
        self.c_8_cl = nn.Conv2d(512, 21 * 6, 3, padding=1)

        self.c_9_bb = nn.Conv2d(256, 24, 3, padding=1)
        self.c_9_cl = nn.Conv2d(256, 21 * 6, 3, padding=1)

        self.c_10_bb = nn.Conv2d(256, 16, 3, padding=1)
        self.c_10_cl = nn.Conv2d(256, 21 * 4, 3, padding=1)

        self.c_11_bb = nn.Conv2d(256, 16, 3, padding=1)
        self.c_11_cl = nn.Conv2d(256, 21 * 4, 3, padding=1)
        self.initialization()

    def get_norm(self):
        return torch.norm(self.fc6) + torch.norm(self.fc6_b) + torch.norm(self.fc7) + torch.norm(self.fc7_b)

    def initialization(self):
        for l in list(self.children())[-16:]:
            if isinstance(l, nn.Conv2d):
                self.initialize(l)
            if isinstance(l, nn.Sequential):
                self.initialize(l[0])
                self.initialize(l[2])

    def initialize(self, c):
        nn.init.xavier_uniform_(c.weight)
        nn.init.constant_(c.bias, 0.)


    def forward(self, x):
        bs = x.shape[0]
        x = self.conv_4_3(x)
        conv4_3_Features = x.clone()
        norm = conv4_3_Features.pow(2).sum(dim=1, keepdim=True).sqrt()  # (bs, 1, 38, 38), so each feature is normalized to [0,1]
        conv4_3_Features = conv4_3_Features / norm  # (N, 512, 38, 38)
        conv4_3_Features = conv4_3_Features * self.rescaling_conv_4_3   # (N, 512, 38, 38), it's need is same as in BN. by setting it
                                                                        # to l-2 norm we can recover the original value.

        bb_4 = self.c_4_bb(conv4_3_Features).permute(0, 2, 3, 1).contiguous().view(bs, -1, 4)
        cl_4 = self.c_4_cl(conv4_3_Features).permute(0, 2, 3, 1).contiguous().view(bs, -1, 21)

        x = self.seq7(self.seq5(x))
        bb_7 = self.c_7_bb(x).permute(0, 2, 3, 1).contiguous().view(bs, -1, 4)
        cl_7 = self.c_7_cl(x).permute(0, 2, 3, 1).contiguous().view(bs, -1, 21)

        x = self.seq8(x)
        bb_8 = self.c_8_bb(x).permute(0, 2, 3, 1).contiguous().view(bs, -1, 4)
        cl_8 = self.c_8_cl(x).permute(0, 2, 3, 1).contiguous().view(bs, -1, 21)

        x = self.seq9(x)
        bb_9 = self.c_9_bb(x).permute(0, 2, 3, 1).contiguous().view(bs, -1, 4)
        cl_9 = self.c_9_cl(x).permute(0, 2, 3, 1).contiguous().view(bs, -1, 21)

        x = self.seq10(x)
        bb_10 = self.c_10_bb(x).permute(0, 2, 3, 1).contiguous().view(bs, -1, 4)
        cl_10 = self.c_10_cl(x).permute(0, 2, 3, 1).contiguous().view(bs, -1, 21)

        x = self.seq11(x)
        bb_11 = self.c_11_bb(x).permute(0, 2, 3, 1).contiguous().view(bs, -1, 4)
        cl_11 = self.c_11_cl(x).permute(0, 2, 3, 1).contiguous().view(bs, -1, 21)

        return torch.cat([bb_4, bb_7, bb_8, bb_9, bb_10, bb_11], dim=1), torch.cat([cl_4, cl_7, cl_8, cl_9, cl_10, cl_11], dim=1)
