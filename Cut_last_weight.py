import os
import torch
from torchsummary import summary

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model = torch.load("/root/Yolact/yolact/weights/resnet101_reducedfc.pth")
for name, param in model.named_parameters():
    print(name, param.data)
print()