from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import os
import torch
import numpy as np
import cv2 as cv
from PIL import Image
import torchvision
import matplotlib.pyplot as plt

class MyCompose(object):
    def __init__(self, pro, N, grids):
        """
        :param pro: 变换概率
        :param N: 分为N*N的栅格
        :param grids: 选择多少个栅格
        """
        self.pro = pro
        self.N = N
        self.grids = grids
        self.Flag = False
        pass

    def changeImage(self, oring_img, edge_img):
        random_img_p = np.random.rand()
        if random_img_p > 0.5:
            oring_img = oring_img
            edge_img = edge_img
        else:
            oring_img, edge_img = edge_img, oring_img
            self.Flag = True

        # 计算网格尺寸
        N = self.N  # 栅格的行列数
        height, width = oring_img.shape[:2]
        grid_width = width // N
        grid_height = height // N

        # 创建网格列表
        grids = [(i * grid_width, j * grid_height, (i + 1) * grid_width, (j + 1) * grid_height) for i in range(N) for j in range(N)]

        # 预先决定变换参数
        should_transform = np.random.rand() < 0.1
        if should_transform:
            angle = np.random.uniform(0, 360)
            flip_code = np.random.choice([-1, 0, 1])
        else:
            angle = None
            flip_code = None

        def random_transform(image, target_height, target_width, angle, flip_code):
            """对图像进行指定的旋转、翻折，并调整尺寸以匹配目标尺寸"""
            if angle is not None:
                image_center = (image.shape[1] / 2, image.shape[0] / 2)
                rotation_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
                abs_cos = abs(rotation_mat[0, 0])
                abs_sin = abs(rotation_mat[0, 1])
                bound_w = int(image.shape[0] * abs_sin + image.shape[1] * abs_cos)
                bound_h = int(image.shape[0] * abs_cos + image.shape[1] * abs_sin)
                rotation_mat[0, 2] += bound_w / 2 - image_center[0]
                rotation_mat[1, 2] += bound_h / 2 - image_center[1]
                image = cv.warpAffine(image, rotation_mat, (bound_w, bound_h))
                image = cv.flip(image, flip_code)
            resized_image = cv.resize(image, (target_width, target_height))
            return resized_image

        result_img = oring_img.copy()
        num_selected_grids = self.grids  # 可根据需要调整选取的栅格数量
        selected_indices = np.random.choice(len(grids), num_selected_grids, replace=False)

        for index in selected_indices:
            x1, y1, x2, y2 = grids[index]
            original_rect = result_img[y1:y2, x1:x2]
            edge_rect = edge_img[y1:y2, x1:x2]

            transformed_original_rect = random_transform(original_rect, x2 - x1, y2 - y1, angle, flip_code)
            transformed_edge_rect = random_transform(edge_rect, x2 - x1, y2 - y1, angle, flip_code)

            rand_weight = np.random.uniform(0.9, 1)
            if self.Flag:
                blended_rect = cv.addWeighted(transformed_edge_rect, 1 - rand_weight, transformed_original_rect, rand_weight, 1, dtype=cv.CV_32F)
            else:
                blended_rect = cv.addWeighted(transformed_edge_rect, rand_weight, transformed_original_rect, 1 -  rand_weight, 1, dtype=cv.CV_32F)
            result_img[y1:y2, x1:x2] = blended_rect

        return result_img

    def __call__(self, img):
        self.p = np.random.rand()
        img_ = np.array(img).astype('uint8')  # 转imge到ndrrary形式
        if self.p > self.pro:
            # laplac
            arr_ksize = [1, 3, 5, 7, 9, 11, 13, 15, 17]
            # arr_ksize = [11]
            random_scale = np.random.uniform(1, 50)
            random_weight = np.random.rand()

            # 生成 Beta 分布的随机数
            weights = np.random.beta(0.4, 4, len(arr_ksize))
            weights /= weights.sum()
            rand_ksize = np.random.choice(arr_ksize, size=1, p=weights)

            arr_func = ['Laplac_img', 'Laplac_img2', 'Sobel', 'Sobel2', 'Roberts', 'segmented_image_color', 'canny']
            # arr_func = ['Laplac_img', 'Sobel', 'Roberts', 'canny', 'segmented_image_color', 'Laplac_img2', 'Sobel2']
            alpha_img = np.random.choice(arr_func, 1, p=[0.3, 0.01, 0.15, 0.01, 0.19, 0.04, 0.3])
            # alpha_img = np.random.choice(arr_func, 1, p=[0, 0, 0, 0, 0, 0, 1])
            # weights2 = np.random.beta(0.4, 4, len(arr_func))
            # weights2 /= weights2.sum()
            # alpha_img = np.random.choice(arr_func, size=1, p=weights2)

            if alpha_img == 'Laplac_img':
                Laplac_img = cv.Laplacian(img_, -1, ksize=rand_ksize[0], scale=random_scale)
                alpha_img = Laplac_img.astype('uint8')
                mix_img = self.changeImage(img_, alpha_img)
                pil_img = Image.fromarray(mix_img.astype('uint8')).convert('RGB')  # 转为pil
                return pil_img
            elif alpha_img == 'Laplac_img2':
                Laplac_img = cv.Laplacian(img_, -1, ksize=rand_ksize[0], scale=random_scale)
                Laplac_img2 = cv.Laplacian(Laplac_img, -1, ksize=rand_ksize[0], scale=random_scale)
                alpha_img = Laplac_img2.astype('uint8')
                mix_img = self.changeImage(img_, alpha_img)
                pil_img = Image.fromarray(mix_img.astype('uint8')).convert('RGB')  # 转为pil
                return pil_img
            elif alpha_img == 'Sobel':
                # sobel
                x = cv.Sobel(img_, cv.CV_32F, 1, 0, ksize=arr_ksize[0])
                y = cv.Sobel(img_, cv.CV_32F, 0, 1, ksize=arr_ksize[0])
                absX = cv.convertScaleAbs(x)
                absY = cv.convertScaleAbs(y)
                # 转 uint8 ,图像融合
                Sobel = cv.addWeighted(x, 1 - random_weight, y, random_weight, 1)
                alpha_img = Sobel.astype('uint8')
                mix_img = self.changeImage(img_, alpha_img)
                pil_img = Image.fromarray(mix_img.astype('uint8')).convert('RGB')  # 转为pil
                return pil_img
            elif alpha_img == 'Sobel2':
                # sobel
                x = cv.Sobel(img_, cv.CV_32F, 1, 0, ksize=arr_ksize[0])
                y = cv.Sobel(img_, cv.CV_32F, 0, 1, ksize=arr_ksize[0])
                # 转 uint8 ,图像融合
                Sobel = cv.addWeighted(x, 1 - random_weight, y, random_weight, 1)
                x2 = cv.Sobel(Sobel, cv.CV_32F, 1, 0, ksize=arr_ksize[0])
                y2 = cv.Sobel(Sobel, cv.CV_32F, 0, 1, ksize=arr_ksize[0])
                absX = cv.convertScaleAbs(x2)
                absY = cv.convertScaleAbs(y2)
                # 转 uint8 ,图像融合
                Sobel2 = cv.addWeighted(x2, random_weight, y2, 1 - random_weight, 1)
                alpha_img = Sobel2.astype('uint8')
                mix_img = self.changeImage(img_, alpha_img)
                pil_img = Image.fromarray(mix_img.astype('uint8')).convert('RGB')  # 转为pil
                return pil_img
            elif alpha_img == 'Roberts':
                # Roberts算子
                kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
                kernely = np.array([[0, -1], [1, 0]], dtype=int)
                x = cv.filter2D(img_, cv.CV_32F, kernelx)
                y = cv.filter2D(img_, cv.CV_32F, kernely)
                absX = cv.convertScaleAbs(x)
                absY = cv.convertScaleAbs(y)
                # 转uint8
                Roberts = cv.addWeighted(x, random_weight, y, 1 - random_weight, 10)
                alpha_img = Roberts.astype('uint8')
                mix_img = self.changeImage(img_, alpha_img)
                pil_img = Image.fromarray(mix_img.astype('uint8')).convert('RGB')  # 转为pil
                return pil_img
            elif alpha_img == 'segmented_image_color':
                gray_image = cv.cvtColor(img_, cv.COLOR_BGR2GRAY)
                gray_image = gray_image.astype(np.uint8)
                _, binary_image = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                # 将二值化图像转换为三通道的灰度图像
                segmented_image_gray = cv.cvtColor(binary_image, cv.COLOR_GRAY2BGR)
                # 将灰度图像转换为彩色图像
                segmented_image_color = cv.cvtColor(segmented_image_gray, cv.COLOR_BGR2RGB)
                alpha_img = segmented_image_color.astype('uint8')
                mix_img = self.changeImage(img_, alpha_img)
                pil_img = Image.fromarray(mix_img.astype('uint8')).convert('RGB')  # 转为pil
                return pil_img
            elif alpha_img == 'canny':
                gray_image = cv.cvtColor(img_, cv.COLOR_BGR2GRAY)
                gray_image = gray_image.astype(np.uint8)
                # 边缘检测分割
                edges = cv.Canny(gray_image, 50, 100)
                # 根据边缘信息进行图像分割
                segmented_image = np.zeros_like(gray_image)
                segmented_image[edges > 0] = 255
                # 将分割后的灰度图像转换为三通道彩色图像
                color_segmented_image = cv.cvtColor(segmented_image, cv.COLOR_GRAY2BGR)
                alpha_img = color_segmented_image.astype('uint8')
                mix_img = self.changeImage(img_, alpha_img)
                pil_img = Image.fromarray(mix_img.astype('uint8')).convert('RGB')  # 转为pil
                return pil_img
        else:
            return img

def load_ImageNet(batch_size=64, workers=8, pin_memory=True): 
    
    traindir = os.getcwd() + r'/imagenet/train'
    valdir   = os.getcwd() + r'/imagenet/val'
    
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # MyCompose(0.85, N=8, grids=2),
            transforms.RandomHorizontalFlip(),
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
    return train_loader, val_loader, train_dataset, val_dataset


# def imshow(img):
#     # 反归一化
#     img = img / 2 + 0.5     # 逆归一化
#     npimg = img.numpy()     # 转换为 NumPy 数组
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 转置为 (H, W, C) 格式
#     plt.axis('off')         # 关闭坐标轴

# def visualize_and_save(train_loader):
#     # 获取一批图像
#     dataiter = iter(train_loader)
#     images, labels = next(dataiter)

#     # 可视化图像
#     imshow(torchvision.utils.make_grid(images))
#     plt.title('Sample Images from ImageNet')
#     plt.savefig('imagenet_samples.png')  # 保存图像
#     plt.show()  # 显示图像

# if __name__ == "__main__":
#     train_loader, val_loader, train_dataset, val_dataset = load_ImageNet()
#     visualize_and_save(train_loader)
