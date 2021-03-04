import torch
import torch.nn as nn
from torchvision import models


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

        self.conv2d_02_bb4 = nn.Conv2d(256, 4 * k, kernel_size=3, stride=1, padding=1)
        self.conv2d_02_c4 = nn.Conv2d(256, (self.n_classes + 1) * k, kernel_size=3, stride=1, padding=1)
        self.conv2d_02_c4.bias.data.zero_().add_(-4)

        self.conv2d_02_bb2 = nn.Conv2d(256, 4 * k, kernel_size=3, stride=1, padding=1)
        self.conv2d_02_c2 = nn.Conv2d(256, (self.n_classes + 1) * k, kernel_size=3, stride=1, padding=1)
        self.conv2d_02_c2.bias.data.zero_().add_(-4)

        self.conv2d_02_bb1 = nn.Conv2d(256, 4 * k, kernel_size=3, stride=1, padding=1)
        self.conv2d_02_c1 = nn.Conv2d(256, (self.n_classes + 1) * k, kernel_size=3, stride=1, padding=1)
        self.conv2d_02_c1.bias.data.zero_().add_(-4)

        # self.adaptivePooling2 = nn.AdaptiveAvgPool2d((2,2))
        # self.adaptivePooling1 = nn.AdaptiveAvgPool2d((1,1))

        self.bn4 = nn.BatchNorm2d((self.n_classes + 1) * k)
        # self.bn2 = nn.BatchNorm2d((self.n_classes + 1)*k)
        # self.bn1 = nn.BatchNorm2d((self.n_classes + 1)*k)

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
            # nn.Dropout2d(p=self.dropout_p),
            nn.ReLU()
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
        f4b = f4b.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 4 * self.k).view(x.shape[0], -1, self.k, 4).contiguous()

        f4c = f4c.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 21 * self.k).view(x.shape[0], -1, self.k, 21).contiguous()

        f2b = f2b.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 4 * self.k).view(x.shape[0], -1, self.k, 4).contiguous()
        f2c = f2c.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 21 * self.k).view(x.shape[0], -1, self.k, 21).contiguous()

        f1b = f1b.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 4 * self.k).view(x.shape[0], -1, self.k, 4).contiguous()
        f1c = f1c.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 21 * self.k).view(x.shape[0], -1, self.k, 21).contiguous()

        # output.permute(0,2,3,1).view(2,-1,63).view(2, -1, 3, 21).permute(0,2,1,3).shape

        # print(f4b.shape, f4c.shape)

        output4 = torch.cat((f4b, f4c), dim=3)
        output2 = torch.cat((f2b, f2c), dim=3)
        output1 = torch.cat((f1b, f1c), dim=3)

        # print(output4.shape)
        # .permute(0,2,3,1).contiguous().view(x.shape[0],-1,4 + self.n_classes + 1)
        # output2 = torch.cat((f2b,f2c), dim=1).permute(0,2,3,1).contiguous().view(x.shape[0],-1,4 + self.n_classes + 1)
        # output1 = torch.cat((f1b,f1c), dim=1).permute(0,2,3,1).contiguous().view(x.shape[0],-1,4 + self.n_classes + 1)

        # print(output4.shape)
        # print(output2.shape)
        # print(output1.shape)

        output4 = output4.view(x.shape[0], -1, 25)
        output2 = output2.view(x.shape[0], -1, 25)
        output1 = output1.view(x.shape[0], -1, 25)
        # print(output4.shape)
        # print(output2.shape)
        # print(output1.shape)

        # torch.Size([2, 144, 25])
        # torch.Size([2, 36, 25])
        # torch.Size([2, 9, 25])

        return torch.cat((output4, output2, output1), dim=1)
        # return output4


