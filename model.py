import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, classes=12):
        super(Network, self).__init__()
        self.feature = resnet50(pretrained=True)

        self.att = nn.Sequential(
            nn.Conv2d(2048, 1, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, classes)

    def forward(self, x):
        x = self.feature(x)
        att = self.att(x)
        x = x * att
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x