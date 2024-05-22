import cv2 as cv
import numpy as np
from scipy import signal
from torchvision import transforms
from PIL import Image
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class MyCompose(object):
    def __init__(self, pro, N, grids):
        self.pro = pro
        self.N = N
        self.grids = grids
        self.flag = False

    def changeImage(self, oring_img, edge_img):

        random_img_p = np.random.rand()
        if random_img_p > 0.5:
            oring_img = oring_img
            edge_img = edge_img
        else:
            exchange = oring_img
            oring_img = edge_img
            edge_img = exchange
            self.flag = True

        def select_random_grids(grids, num_selected):
            """Select a specified number of grids at random from all possible grids"""
            return np.random.choice(len(grids), num_selected, replace=False)

        def random_transform(image, target_height, target_width):
            # Randomly decide whether to perform the transformation
            if np.random.rand() < 0.5:
                angle = np.random.uniform(0, 360)
                image_center = (image.shape[1] / 2, image.shape[0] / 2)
                rotation_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
                abs_cos = abs(rotation_mat[0, 0])
                abs_sin = abs(rotation_mat[0, 1])
                bound_w = int(image.shape[0] * abs_sin + image.shape[1] * abs_cos)
                bound_h = int(image.shape[0] * abs_cos + image.shape[1] * abs_sin)
                rotation_mat[0, 2] += bound_w / 2 - image_center[0]
                rotation_mat[1, 2] += bound_h / 2 - image_center[1]
                rotated_image = cv.warpAffine(image, rotation_mat, (bound_w, bound_h))

                # Random folding
                flip_code = np.random.choice([-1, 0, 1])
                transformed_image = cv.flip(rotated_image, flip_code)

                #
                # hsv_image = cv.cvtColor(transformed_image, cv.COLOR_BGR2HSV)
                #
                #
                # exponential_factor = np.random.uniform(0.25, 4)
                # hsv_image[:, :, 1:] = np.power(hsv_image[:, :, 1:], exponential_factor)
                #
                #
                # final_image = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)

                transformed_image = transformed_image
            else:

                transformed_image = image

            resized_image = cv.resize(transformed_image, (target_width, target_height))

            return resized_image

        result_img = oring_img.copy()

        # The image is divided into N*N grids
        N = self.N
        height, width = oring_img.shape[:2]
        grid_width = width // N
        grid_height = height // N
        grids = [(i * grid_width, j * grid_height, (i + 1) * grid_width, (j + 1) * grid_height) for i in range(N) for j
                 in range(N)]

        # Select a random number grid
        num_selected_grids = self.grids
        selected_indices = select_random_grids(grids, num_selected_grids)

        for index in selected_indices:
            x1, y1, x2, y2 = grids[index]
            original_rect = result_img[y1:y2, x1:x2]
            edge_rect = edge_img[y1:y2, x1:x2]

            # The selected region of the edge detection image is rotated and folded randomly
            transformed_edge_rect = random_transform(edge_rect, x2 - x1, y2 - y1)

            rand_weight = np.random.uniform(0.9, 1)
            if self.flag:
                blended_rect = cv.addWeighted(transformed_edge_rect, rand_weight, original_rect, 1 - rand_weight, 1,
                                              dtype=cv.CV_32F)
            else:
                blended_rect = cv.addWeighted(transformed_edge_rect, 1 - rand_weight, original_rect, rand_weight, 1,
                                              dtype=cv.CV_32F)
            result_img[y1:y2, x1:x2] = blended_rect

        return result_img

    def __call__(self, img):
        self.p = np.random.rand()
        img_ = np.array(img).astype('uint8')
        if self.p > self.pro:
            # laplac
            arr_ksize = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
            # arr_ksize = [1, 3, 5, 7, 9, 11]
            random_scale = np.random.uniform(1, 50)
            random_weight = np.random.rand()

            # beta distributes random numbers
            weights = np.random.beta(0.4, 4, len(arr_ksize))
            weights /= weights.sum()
            rand_ksize = np.random.choice(arr_ksize, size=1, p=weights)

            arr_func = ['Laplac_img', 'Laplac_img2', 'Sobel', 'Sobel2', 'Roberts', 'segmented_image_color', 'canny']
            alpha_img = np.random.choice(arr_func, 1, p=[0.3, 0.01, 0.3, 0.01, 0.3, 0.04, 0.04])

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

                Sobel = cv.addWeighted(x, 1 - random_weight, y, random_weight, 1)
                alpha_img = Sobel.astype('uint8')
                mix_img = self.changeImage(img_, alpha_img)
                pil_img = Image.fromarray(mix_img.astype('uint8')).convert('RGB')  # 转为pil
                return pil_img
            elif alpha_img == 'Sobel2':
                # sobel
                x = cv.Sobel(img_, cv.CV_32F, 1, 0, ksize=arr_ksize[0])
                y = cv.Sobel(img_, cv.CV_32F, 0, 1, ksize=arr_ksize[0])

                Sobel = cv.addWeighted(x, 1 - random_weight, y, random_weight, 1)
                x2 = cv.Sobel(Sobel, cv.CV_32F, 1, 0, ksize=arr_ksize[0])
                y2 = cv.Sobel(Sobel, cv.CV_32F, 0, 1, ksize=arr_ksize[0])
                absX = cv.convertScaleAbs(x2)
                absY = cv.convertScaleAbs(y2)

                Sobel2 = cv.addWeighted(x2, random_weight, y2, 1 - random_weight, 1)
                alpha_img = Sobel2.astype('uint8')
                mix_img = self.changeImage(img_, alpha_img)
                pil_img = Image.fromarray(mix_img.astype('uint8')).convert('RGB')  # 转为pil
                return pil_img
            elif alpha_img == 'Roberts':
                # Roberts
                kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
                kernely = np.array([[0, -1], [1, 0]], dtype=int)
                x = cv.filter2D(img_, cv.CV_32F, kernelx)
                y = cv.filter2D(img_, cv.CV_32F, kernely)
                absX = cv.convertScaleAbs(x)
                absY = cv.convertScaleAbs(y)
                
                Roberts = cv.addWeighted(x, random_weight, y, 1 - random_weight, 10)
                alpha_img = Roberts.astype('uint8')
                mix_img = self.changeImage(img_, alpha_img)
                pil_img = Image.fromarray(mix_img.astype('uint8')).convert('RGB')  # 转为pil
                return pil_img
            elif alpha_img == 'segmented_image_color':
                gray_image = cv.cvtColor(img_, cv.COLOR_BGR2GRAY)
                gray_image = gray_image.astype(np.uint8)
                _, binary_image = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                
                segmented_image_gray = cv.cvtColor(binary_image, cv.COLOR_GRAY2BGR)
                
                segmented_image_color = cv.cvtColor(segmented_image_gray, cv.COLOR_BGR2RGB)
                alpha_img = segmented_image_color.astype('uint8')
                mix_img = self.changeImage(img_, alpha_img)
                pil_img = Image.fromarray(mix_img.astype('uint8')).convert('RGB')  # 转为pil
                return pil_img
            elif alpha_img == 'canny':
                gray_image = cv.cvtColor(img_, cv.COLOR_BGR2GRAY)
                gray_image = gray_image.astype(np.uint8)
                
                edges = cv.Canny(gray_image, 50, 100)
                
                segmented_image = np.zeros_like(gray_image)
                segmented_image[edges > 0] = 255
                
                color_segmented_image = cv.cvtColor(segmented_image, cv.COLOR_GRAY2BGR)
                alpha_img = color_segmented_image.astype('uint8')
                mix_img = self.changeImage(img_, alpha_img)
                pil_img = Image.fromarray(mix_img.astype('uint8')).convert('RGB')  # 转为pil
                return pil_img
        else:
            return img


transform_edge = transforms.Compose([
    # transforms.Resize((128, 128)),
    transforms.RandomCrop(padding=4, size=32),
    MyCompose(0.5, N=8, grids=50),
    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),
    transforms.Normalize([0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                         [0.2673342858792401, 0.2564384629170883, 0.27615047132568404])

])

transform_val = transforms.Compose([
    # transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                         [0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    # transforms.Normalize([0.48836562, 0.48134598, 0.4451678],
    #                      [0.24833508, 0.24547848, 0.26617324])
])



train_root = r"your_root"
test_root = r"your_root"

train_dataset = datasets.ImageFolder(train_root, transform=transform_edge)
valid_dataset = datasets.ImageFolder(test_root, transform=transform_val)

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8, drop_last=True, )
valid_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=8,
                              drop_last=True, )
