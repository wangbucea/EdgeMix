import random
from abc import ABC, abstractmethod

import skimage
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets import CIFAR10
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.transforms import transforms
from tqdm import tqdm
import cv2 as cv
import numpy as np
from PIL import Image

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = True
set_random_seed(42)

# model = vit_b_16(ViT_B_16_Weights.IMAGENET1K_V1)
# model.heads = nn.Sequential(
#     nn.Linear(model.heads.head.in_features, 10)
# )
# model = vit_b_16(ViT_B_16_Weights.IMAGENET1K_V1)

weights = Swin_T_Weights.IMAGENET1K_V1
model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
num_features = model.head.in_features
model.head = nn.Linear(num_features, 10)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
# Freeze all layers
# for param in model.parameters():
#     param.requires_grad = False

# Unfreeze the last encoder layer and the head
# for param in model.encoder.layers[-1].parameters():
#     param.requires_grad = True
# for param in model.heads.parameters():
#     param.requires_grad = True


class AugConfig:
    """配置参数容器"""

    def __init__(self):
        self.prob = 0.5  # 基础触发概率
        self.grid_size = 8  # 基础网格尺寸
        self.min_grid_units = 1  # 最小选择网格数
        self.max_grid_units = 50  # 最大选择网格数
        self.entropy_range = (0.1, 0.9)  # 动态权重范围


class EdgeDetector(ABC):
    """边缘检测策略抽象基类"""

    @abstractmethod
    def detect(self, img: np.ndarray) -> np.ndarray:
        pass


class LaplacianDetector(EdgeDetector):
    def __init__(self, ksize_range=(1, 17)):
        self.ksizes = [k for k in range(*ksize_range) if k % 2 == 1]

    def detect(self, img):
        ksize = np.random.choice(self.ksizes)
        scale = np.random.uniform(1, 50)
        return cv.Laplacian(img, -1, ksize=ksize, scale=scale)


class SobelDetector(EdgeDetector):
    def detect(self, img):
        ksize = np.random.choice([1, 3, 5])
        x = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=ksize)
        y = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=ksize)
        Sobel = cv.addWeighted(x, 0.5, y, 0.5, 1)
        return cv.convertScaleAbs(Sobel)


class CannyDetector(EdgeDetector):
    def detect(self, img):
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        low = np.random.randint(50, 150)
        high = np.random.randint(low + 50, 300)
        edges = cv.Canny(gray, low, high)
        return cv.cvtColor(edges, cv.COLOR_GRAY2RGB)


class DynamicGridGenerator:
    """修改后的网格生成器（均匀划分）"""

    def __init__(self, config: AugConfig):
        self.config = config

    def __call__(self, img):
        h, w = img.shape[:2]
        grid_size = self.config.grid_size
        grid_w = w // grid_size
        grid_h = h // grid_size

        grids = []
        for i in range(grid_size):
            for j in range(grid_size):
                x1 = i * grid_w
                y1 = j * grid_h
                x2 = (i + 1) * grid_w
                y2 = (j + 1) * grid_h
                grids.append((x1, y1, x2, y2))
        return grids


class MyComposeV2:
    """改进版数据增强处理器"""

    def __init__(self, config: AugConfig = None):
        self.config = config or AugConfig()
        self.edge_detectors = {
            'laplacian': LaplacianDetector(),
            'sobel': SobelDetector(),
            'canny': CannyDetector()
        }
        self.grid_gen = DynamicGridGenerator(self.config)

    def __call__(self, img):
        if np.random.rand() > self.config.prob:
            return img

        img_np = np.array(img).astype('uint8')
        edge_type = np.random.choice(list(self.edge_detectors.keys()))

        # 边缘检测
        detector = self.edge_detectors[edge_type]
        edge_map = detector.detect(img_np)

        # 动态网格生成
        grids = self.grid_gen(img_np)

        # 随机选择网格
        num_grids = np.random.randint(
            self.config.min_grid_units,
            self.config.max_grid_units + 1
        )
        selected = np.random.choice(len(grids), num_grids, replace=False)

        # 创建内存视图避免复制
        result = np.ndarray(
            shape=img_np.shape,
            dtype=img_np.dtype,
            buffer=img_np.data
        )

        # 处理每个网格
        for idx in selected:
            x1, y1, x2, y2 = grids[idx]
            self._process_grid(
                result[y1:y2, x1:x2],
                edge_map[y1:y2, x1:x2]
            )

        return Image.fromarray(result).convert('RGB')

    def _process_grid(self, orig_patch, edge_patch):
        """处理单个网格块"""
        # 独立变换参数
        transform_params = {
            'angle': np.random.uniform(0, 360) if np.random.rand() < 0.1 else 0,
            'flip': np.random.choice([-1, 0, 1]) if np.random.rand() < 0.1 else 0
        }

        # 执行变换
        transformed_orig = self._transform(orig_patch, **transform_params)
        transformed_edge = self._transform(edge_patch, **transform_params)

        # 动态混合权重
        weight = self._calc_content_weight(transformed_orig)
        blended = cv.addWeighted(
            transformed_edge, weight,
            transformed_orig, 1 - weight,
            0
        )

        # 原位更新
        np.copyto(orig_patch, blended)

    def _transform(self, patch, angle=0, flip=0):
        """执行几何变换"""
        if angle != 0:
            center = (patch.shape[1] / 2, patch.shape[0] / 2)
            M = cv.getRotationMatrix2D(center, angle, 1.0)
            patch = cv.warpAffine(patch, M, patch.shape[1::-1])
        if flip != 0:
            patch = cv.flip(patch, flip)
        return patch

    def _calc_content_weight(self, patch):
        """基于信息熵计算混合权重"""
        gray = cv.cvtColor(patch, cv.COLOR_RGB2GRAY)
        entropy = skimage.measure.shannon_entropy(gray)
        min_e, max_e = self.config.entropy_range
        return np.clip(entropy / 7, min_e, max_e)

#  95.58    96.34
#  SWIN Test accuracy: 95.77%      94.79
config = AugConfig()
config.prob = 0.7
aug = MyComposeV2(config)
transform_train = transforms.Compose([
    transforms.Resize(224),
    # aug,  # 做变换
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_set = CIFAR10(root='/home/wang/project/pythonProject/data', train=True, download=False, transform=transform_train)
test_set = CIFAR10(root='/home/wang/project/pythonProject/data', train=False, download=False, transform=transform_test)

train_loader = DataLoader(train_set, shuffle=True, batch_size=64)
test_loader = DataLoader(test_set, shuffle=False, batch_size=64)

n_epochs = 10
lr = 0.0001

optimizer = Adam(model.parameters(), lr=lr)
criterion = CrossEntropyLoss()

for epoch in range(n_epochs):
    train_loss = 0.0
    for i,batch in enumerate(train_loader):
        x, y = batch
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y)

        batch_loss = loss.detach().cpu().item()
        train_loss += batch_loss / len(train_loader)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%100==0:
          print(f"Batch {i}/{len(train_loader)} loss: {batch_loss:.03f}")

    print(f"Epoch {epoch + 1}/{n_epochs} loss: {train_loss:.03f}")

    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")
