import os
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
# from torchvision.models import resnet50
# from torchvision.models import resnet34
from torchvision.models import resnext50_32x4d
from torch.utils.data import DataLoader
from ImageNet_load import load_ImageNet
from tqdm import tqdm
import torch

# 设置超参数
learning_rate = 0.001
num_epochs = 10
model_save_path = './imagenet_resnext_origin.pth'

# 加载 ImageNet 数据集
train_loader, val_loader, train_dataset, val_dataset = load_ImageNet()

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnext50_32x4d(pretrained=False).to(device)  # 使用预训练模型并移动到 GPU
model.load_state_dict(torch.load(os.getcwd()+'/resnext50_32x4d-7cdf4587.pth'))
# 设置损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 检查是否有已保存的模型
start_epoch = 0
if os.path.exists(model_save_path):
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Resuming training from epoch {start_epoch}...")

    

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
    print(f'Top-1 error: {top1_error_rate:.4f}, Top-5 error: {top5_error_rate:.4f}')

    return top1_error_rate,top5_error_rate
    


# 训练模型
model.train()
for epoch in range(start_epoch, num_epochs):
    running_loss = 0.0
    correct_top1 = 0  # 在每个 epoch 开始时重置
    total = 0  # 记录总样本数
    with tqdm(total=len(train_loader), desc=f'Epoch [{epoch + 1}/{num_epochs}]', unit='batch') as pbar:
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # 移动数据到 GPU
            
            optimizer.zero_grad()  # 清零梯度

            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
            correct_top1 += (predicted == labels).sum().item()  # 累加正确预测数
            total += labels.size(0)  # 累加总样本数
            
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (i + 1),correct=correct_top1 ,acc=correct_top1 / total)
            pbar.update(1)  # 更新进度条

        # 保存模型
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_save_path)
        # 测试模型并记录准确率
        with torch.no_grad():
            top1_accuracy, top5_accuracy = test_model(model, val_loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}] - Top-1 error: {top1_accuracy:.4f}, Top-5 error: {top5_accuracy:.4f}')


print('Training finished.')
test_model(model, val_loader)
