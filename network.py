import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


# base model
class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 5 * 10)
        x = self.classifier(x)
        return x


class MultiImageAlexNet(AlexNet):
    def __init__(self, num_classes=1000):
        super(MultiImageAlexNet, self).__init__(num_classes=num_classes)
        self.global_features = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 5 * 10, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            # nn.Linear(4096, 1024),
            # nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )
        self.gru = nn.GRU(4096, 1024, batch_first=True, num_layers=2)
        self.is_cuda = False

    def cuda(self, device_id=None):
        super(MultiImageAlexNet, self).cuda(device_id)
        self.is_cuda = True

    def forward(self, imgs):
        feature_outputs = []
        for x in imgs:
            if self.is_cuda:
                x = Variable(x.cuda(async=True))
            else:
                x = Variable(x)
            if x.size(1) == 1:  # single channel image?
                x = x.expand(x.size(0), 3, x.size(2), x.size(3))
            elif x.size(1) == 4:  # RGBA?
                x = x[:, :3, :, :]
            try:
                x = self.features(x)
                x = x.view(x.size(0), 256 * 5 * 10)
                x = self.global_features(x)
            except RuntimeError:
                raise ValueError("Dimensionality Error!")
            feature_outputs.append(x.clone())
        y = torch.stack(feature_outputs, 2).transpose(1, 2)
        _, y = self.gru(y)
        # y, _ = torch.max(y, 1)
        y = y[-1, :, :].squeeze(1)
        y = self.classifier(y)
        return y

    def encode(self, imgs):
        feature_outputs = []
        for x in imgs:
            if self.is_cuda:
                x = Variable(x.cuda(async=True))
            else:
                x = Variable(x)
            if x.size(1) == 1:  # single channel image?
                x = x.expand(x.size(0), 3, x.size(2), x.size(3))
            elif x.size(1) == 4:  # RGBA?
                x = x[:, :3, :, :]
            try:
                x = self.features(x)
                x = x.view(x.size(0), 256 * 5 * 10)
                x = self.global_features(x)
            except RuntimeError:
                raise ValueError("Dimensionality Error!")
            feature_outputs.append(x.clone())
        y = torch.stack(feature_outputs, 2).transpose(1, 2)
        _, y = self.gru(y)
        # y, _ = torch.max(y, 1)
        y = y[-1, :, :].squeeze(1)
        return y


def softmax(input, axis=1):
    input_size = input.size()

    trans_input = input.transpose(axis, len(input_size) - 1)
    trans_size = trans_input.size()

    input_2d = trans_input.contiguous().view(-1, trans_size[-1])

    soft_max_2d = F.softmax(input_2d)

    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(axis, len(input_size) - 1)

