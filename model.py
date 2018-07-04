import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from basemodel import *

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}

class ResNet18(nn.Module):
    def __init__(self, classes=12):
        super(ResNet18, self).__init__()
        self.base = resnet18(pretrained=False)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
        self.base.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)

    def forward(self, x):
        f1,f2,f3,f4 = self.base(x)
        x = f4
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResNet50(nn.Module):
    def __init__(self, classes=12):
        super(ResNet50, self).__init__()
        self.base = resnet50(pretrained=False)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
        self.base.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)

    def forward(self, x):
        f1,f2,f3,f4 = self.base(x)
        x = f4
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResNet50_Pyramid(nn.Module):
    def __init__(self, classes=12):
        super(ResNet50_Pyramid, self).__init__()
        self.base = resnet50(pretrained=False)

        self.pool2 = nn.AdaptiveAvgPool2d(4)
        self.pool3 = nn.AdaptiveAvgPool2d(2)
        self.pool4 = nn.AdaptiveAvgPool2d(1)
        self.fc2 = nn.Sequential(
            nn.Linear(512*16, 2048),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(1024*4, 2048),
            nn.ReLU()
        )

        self.classifier = nn.Linear(2048*3, classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
        self.base.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)

    def forward(self, x):
        f1,f2,f3,f4 = self.base(x)
        f2 = self.pool2(f2)
        f3 = self.pool3(f3)
        f4 = self.pool4(f4)

        f2 = f2.view(f2.size(0), -1)
        f2 = self.fc2(f2)

        f3 = f3.view(f3.size(0), -1)
        f3 = self.fc3(f3)

        f4 = f4.view(f4.size(0), -1)
        x = torch.cat([f2,f3,f4], dim=1)
        x = self.classifier(x)
        return x

class ResNet50_Att(nn.Module):
    def __init__(self, classes=12):
        super(ResNet50_Att, self).__init__()
        self.base = resnet50(pretrained=False)

        self.att = nn.Sequential(
            nn.Conv2d(2048, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
        self.base.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)

    def forward(self, x):
        f1,f2,f3,f4 = self.base(x)
        x = f4
        att = self.att(x) + 1
        x = x * att
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResNet101(nn.Module):
    def __init__(self, classes=12):
        super(ResNet101, self).__init__()
        self.base = resnet101(pretrained=False)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
        self.base.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)

    def forward(self, x):
        f1,f2,f3,f4 = self.base(x)
        x = f4
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x