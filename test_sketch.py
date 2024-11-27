import os
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from torchvision.models import resnet34
from torch.utils.data import DataLoader
from ImageNet_load import load_ImageNet
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets

def load_ImageNet(batch_size=64, workers=8, pin_memory=True): 
    
    traindir = os.getcwd() + r'/imagenet/train'
    valdir   = os.getcwd() + r'/imagenet-sketch/frist/sketch'
    
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalizer
        ])
    )

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalizer
        ])
    )
    print('train_dataset = ',len(train_dataset))
    print('val_dataset   = ',len(val_dataset))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    return val_loader




def test_model(model, val_loader):
    model.eval()  # 设置模型为评估模式
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():  # 在测试时不需要计算梯度
        for inputs, labels in tqdm(val_loader, desc='Testing', unit='batch'):
            inputs, labels = inputs.to(device), labels.to(device)  # 移动数据到 GPU

            outputs = model(inputs)  # 前向传播
            _, predicted = torch.max(outputs.data, 1)  # Top-1
            total += labels.size(0)

            correct_top1 += (predicted == labels).sum().item()  # 计算 Top-1 正确数
            
            # 计算 Top-5 正确率
            _, predicted_top5 = torch.topk(outputs, 5, dim=1)
            correct_top5 += (predicted_top5 == labels.view(-1, 1)).sum().item()  # 计算 Top-5 正确数

    top1_error_rate = 1 - (correct_top1 / total)
    top5_error_rate = 1 - (correct_top5 / total)

    print(f'Top-1 Error Rate: {top1_error_rate:.4f}')
    print(f'Top-5 Error Rate: {top5_error_rate:.4f}')
    
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载预训练权重
model = resnet34(pretrained=False).to(device)  # 使用预训练模型并移动到 GPU

# checkpoint = torch.load('imagenet_new)_origin.pth')
# checkpoint = torch.load('imagenet_new.pth')
checkpoint = torch.load('imagenet_8_2_res34.pth')
model.load_state_dict(checkpoint['model_state_dict'])

val_loader = load_ImageNet()
test_model(model, val_loader)
