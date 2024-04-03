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
        initial_mean = theta
        initial_var = theta * (1 - theta)
        self.initial_mean = initial_mean
        self.initial_var = initial_var

        self.mean = nn.Parameter(theta)
        self.log_var = nn.Parameter(torch.log(initial_var))

    def forward(self, x):
        std = torch.exp(0.5 * self.log_var)
        eps = torch.randn_like(std).to(x.device)
        theta = self.mean + 0.01 * eps
        theta = torch.clamp(theta, 0, 1)
        theta = theta.view(1,-1,1,1)
        return x * theta
    def kl_divergence(self):
         # 后验分布的参数
        prior_mean = self.initial_mean
        prior_var = self.initial_var
        post_mean = self.mean
        post_var = torch.exp(self.log_var)

        # 计算KL散度
        kl_div = torch.log(prior_var / post_var) + \
                (post_var + (post_mean - prior_mean) ** 2) / (2 * prior_var) - 0.5
        return kl_div.sum()
    
class Bayes_VGG(nn.Module):
    def __init__(self, vgg_name, layer_cfg=None, priors=None,  num_classes=10):
        super(Bayes_VGG, self).__init__()
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

    def _make_layers(self, layer_cfg, priors):
        layers = []
        in_channels = 3
        conv_index = 0  # Track the index of convolutional layers
        for v in layer_cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # For convolutional layers, apply GaussianScale with specified prior
                prior = priors.get(conv_index, 0.5)  # Default prior is 0.5 if not specified
                conv_layer = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv_layer,
                           nn.BatchNorm2d(v),
                           GaussianScale(theta=prior),
                           nn.ReLU(inplace=True)]
                in_channels = v
                conv_index += 1  # Update the convolutional layer index only for conv layers
        return nn.Sequential(*layers)
        
    def kl_divergence_loss(self):
        kl = 0
        for layer in self.features:
            if isinstance(layer, GaussianScale):
                kl += layer.kl_divergence()
        return kl 
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
