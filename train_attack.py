import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from utils import get_network
from myDataLoader import *
from tqdm import tqdm
import os
        
import torchvision.models as models
import torch.optim as optim
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import *

# FGSM attack code
import torch
import torch.nn.functional as F

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def pgd_attack(model, images, labels, eps, alpha, iters, device):
    original_images = images.clone().detach()
    images = images.clone().detach().to(device)
    labels = labels.to(device)

    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)
        model.zero_grad()
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - original_images, min=-eps, max=eps)
        images = torch.clamp(original_images + eta, min=0, max=1).detach_()
    return images



def evaluate(model, device, dataloader, epsilon, attack_type='fgsm'):
    model.eval()
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True

        # Initial forward pass
        output = model(images)
        loss = F.cross_entropy(output, labels)
        model.zero_grad()
        loss.backward()

        # Generate adversarial examples
        if attack_type == 'fgsm':
            perturbed_images = fgsm_attack(images, epsilon, images.grad)
        elif attack_type == 'pgd':
            perturbed_images = pgd_attack(model, images, labels, eps=epsilon, alpha=2/255, iters=7, device=device)
        else:
            raise ValueError("Unsupported attack type")

        # Re-classify the perturbed images
        output = model(perturbed_images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    error_rate = 1 - correct / total
    print(f'Top-1 Error Rate: {error_rate:.4f}')
    return error_rate



args = 'preactresnet18'
net = get_network(args, num_classes=100)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained_model = os.getcwd() + r"/cifar100_model/preActRes18_edge1_best_model.pth"
# Initialize the network
model = net.to(device)
# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location=device))

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.0004, momentum=0.9, weight_decay=5e-4)
# Train the model with adversarial training
epsilon = 8/255  # Perturbation size
train_loader = valid_dataloader
evaluate(model, device, train_loader, epsilon, attack_type='pgd')

