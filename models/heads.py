import math
import torch
import torch.nn as nn


class CLSHead(nn.Module):
    def __init__(self,
                 in_channels,
                 feat_channels,
                 num_stacked,
                 num_anchors,
                 num_classes):
        super(CLSHead, self).__init__()
        assert num_stacked >= 1, ''
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.convs = nn.ModuleList()
        for i in range(num_stacked):
            chns = in_channels if i == 0 else feat_channels
            self.convs.append(nn.Conv2d(chns, feat_channels, 3, 1, 1))
            self.convs.append(nn.ReLU(inplace=True))
        self.head = nn.Conv2d(feat_channels, num_anchors*num_classes, 3, 1, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        prior = 0.01
        self.head.weight.data.fill_(0)
        self.head.bias.data.fill_(-math.log((1.0 - prior) / prior))

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = torch.sigmoid(self.head(x))
        x = x.permute(0, 2, 3, 1)
        n, w, h, c = x.shape
        x = x.reshape(n, w, h, self.num_anchors, self.num_classes)
        return x.reshape(x.shape[0], -1, self.num_classes)


class REGHead(nn.Module):
    def __init__(self,
                 in_channels,
                 feat_channels,
                 num_stacked,
                 num_anchors,
                 num_regress):
        super(REGHead, self).__init__()
        assert num_stacked >= 1, ''
        self.num_anchors = num_anchors
        self.num_regress = num_regress
        self.convs = nn.ModuleList()
        for i in range(num_stacked):
            chns = in_channels if i == 0 else feat_channels
            self.convs.append(nn.Conv2d(chns, feat_channels, 3, 1, 1))
            self.convs.append(nn.ReLU(inplace=True))
        self.head = nn.Conv2d(feat_channels, num_anchors*num_regress, 3, 1, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.head.weight.data.fill_(0)
        self.head.bias.data.fill_(0)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = self.head(x)
        x = x.permute(0, 2, 3, 1)
        return x.reshape(x.shape[0], -1, self.num_regress)