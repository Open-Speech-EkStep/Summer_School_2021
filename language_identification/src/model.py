from torchvision.models import resnet34
import torch
import torch.nn as nn

def get_model(device, num_classes, pretrained=False):
    model = resnet34(pretrained=pretrained)
    model.fc = nn.Linear(512, num_classes)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.to(device, dtype=torch.float)
    return model