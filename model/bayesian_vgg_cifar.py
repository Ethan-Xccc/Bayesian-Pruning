import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class GaussianScale(nn.Module):
    def __init__(self, theta):
        super(GaussianScale, self).__init__()
        self.mean = nn.Parameter(theta)
        self.var = nn.Parameter(theta*(1-theta))

    def forward(self, x):
        # 生成每个特征通道的高斯分布θ值
        theta = torch.normal(self.mean, torch.sqrt(torch.exp(self.var))).to(x.device)
        # θ值的形状是(num_features,)，需要调整为与x兼容的形状
        theta = theta.view(1, -1, 1, 1)
        return x * theta

class VGG(nn.Module):
    def __init__(self, vgg_name, priors=None, num_classes=10):
        super(VGG, self).__init__()
        if priors is None:
            priors = {}
        self.features = self._make_layers(cfg[vgg_name], priors)
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, priors):
        layers = []
        in_channels = 3
        conv_index = 0  # Track the index of convolutional layers
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # For convolutional layers, apply GaussianScale with specified prior
                prior = priors.get(conv_index, 0.5)  # Default prior is 0.5 if not specified
                conv_layer = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv_layer,
                           nn.BatchNorm2d(v),
                           GaussianScale(v, initial_prob=prior),
                           nn.ReLU(inplace=True)]
                in_channels = v
                conv_index += 1  # Update the convolutional layer index only for conv layers
        return nn.Sequential(*layers)
