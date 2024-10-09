import os
import re
import os.path as osp
from glob import glob

import torch
from PIL import Image
from torch.utils.data import Dataset
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
'''
    Specific dataset classes for person re-identification dataset. 
'''

def cutmix(batch, alpha):
    data = batch
    targets = batch
    N, H, W = data.shape
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    lam = np.random.beta(alpha, alpha)
    image_h = H
    image_w = W
    cx = np.random.uniform(0, W)
    cy = np.random.uniform(0, H)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data[:, y0:y1, x0:x1] = shuffled_data[:, y0:y1, x0:x1]
    return data

def divide_method2(img, m, n):
    h, w = img.shape[0], img.shape[1]
    grid_h = int(h * 1.0 / (m - 1) + 0.5)
    grid_w = int(w * 1.0 / (n - 1) + 0.5)

    h = grid_h * (m - 1)
    w = grid_w * (n - 1)

    img_re = cv2.resize(img, (w, h),cv2.INTER_LINEAR)
    gx, gy = np.meshgrid(np.linspace(0, w, n), np.linspace(0, h, m))
    gx = gx.astype(np.int_)
    gy = gy.astype(np.int_)

    divide_image = np.zeros([m - 1, n - 1, grid_h, grid_w, 3],np.uint8)

    for i in range(m - 1):
        for j in range(n - 1):
            divide_image[i, j, ...] = img_re[
                                      gy[i][j]:gy[i + 1][j + 1], gx[i][j]:gx[i + 1][j + 1], :]
    return divide_image

class SYSUDataset(Dataset):
    def __init__(self, root, mode='train', transform=None):
        assert os.path.isdir(root)
        assert mode in ['train', 'gallery', 'query']

        if mode == 'train':
            train_ids = open(os.path.join(root, 'exp', 'train_id.txt')).readline()
            val_ids = open(os.path.join(root, 'exp', 'val_id.txt')).readline()

            train_ids = train_ids.strip('\n').split(',')
            val_ids = val_ids.strip('\n').split(',')
            selected_ids = train_ids + val_ids
        else:
            test_ids = open(os.path.join(root, 'exp', 'test_id.txt')).readline()
            selected_ids = test_ids.strip('\n').split(',')

        selected_ids = [int(i) for i in selected_ids]
        num_ids = len(selected_ids)

        img_paths = glob(os.path.join(root, '**/*.jpg'), recursive=True)
        img_paths = [path for path in img_paths if int(path.split('/')[-2]) in selected_ids]

        if mode == 'gallery':
            img_paths = [path for path in img_paths if int(path.split('/')[-3][-1]) in (1, 2, 4, 5)]
        elif mode == 'query':
            img_paths = [path for path in img_paths if int(path.split('/')[-3][-1]) in (3, 6)]

        img_paths = sorted(img_paths)
        self.img_paths = img_paths
        self.cam_ids = [int(path.split('/')[-3][-1]) for path in img_paths]
        self.num_ids = num_ids
        self.transform = transform

        if mode == 'train':
            id_map = dict(zip(selected_ids, range(num_ids)))
            self.ids = [id_map[int(path.split('/')[-2])] for path in img_paths]
        else:
            self.ids = [int(path.split('/')[-2]) for path in img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        path = self.img_paths[item]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, path, item

class RegDBDataset(Dataset):
    def __init__(self, root, mode='train', transform=None):
        assert os.path.isdir(root)
        assert mode in ['train', 'gallery', 'query']

        def loadIdx(index):
            Lines = index.readlines()
            idx = []
            for line in Lines:
                tmp = line.strip('\n')
                tmp = tmp.split(' ')
                idx.append(tmp)
            return idx

        num = '1'
        if mode == 'train':
            index_RGB = loadIdx(open(root + '/idx/train_visible_'+num+'.txt','r'))
            index_IR  = loadIdx(open(root + '/idx/train_thermal_'+num+'.txt','r'))
        else:
            index_RGB = loadIdx(open(root + '/idx/test_visible_'+num+'.txt','r'))
            index_IR  = loadIdx(open(root + '/idx/test_thermal_'+num+'.txt','r'))

        if mode == 'gallery':
            img_paths = [root + '/' + path for path, _ in index_RGB]
        elif mode == 'query':
            img_paths = [root + '/' + path for path, _ in index_IR]
        else:
            img_paths = [root + '/' + path for path, _ in index_RGB] + [root + '/' + path for path, _ in index_IR]

        selected_ids = [int(path.split('/')[-2]) for path in img_paths]
        selected_ids = list(set(selected_ids))
        num_ids = len(selected_ids)

        img_paths = sorted(img_paths)
        self.img_paths = img_paths
        self.cam_ids = [int(path.split('/')[-3] == 'Thermal') + 2 for path in img_paths]
        self.num_ids = num_ids
        self.transform = transform

        if mode == 'train':
            id_map = dict(zip(selected_ids, range(num_ids)))
            self.ids = [id_map[int(path.split('/')[-2])] for path in img_paths]
        else:
            self.ids = [int(path.split('/')[-2]) for path in img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        path = self.img_paths[item]
        img = Image.open(path)
        filename = path.split('.')[0][23:66]
        m = 6
        n = 2

        data_path = "/home/gpu/Desktop/RegDB-C"
        ave_img = cv2.imread('/home/gpu/Desktop/MyReidProject-main/avreage_reg_0.jpg')
        ave_image = divide_method2(ave_img, m + 1, n + 1)

        true_path = data_path + filename + '.png'
        imgg = cv2.imread(true_path)
        imgg = cv2.resize(imgg, (144, 288))
        divide_image = divide_method2(imgg, m + 1, n + 1)
        m, n = divide_image.shape[0], divide_image.shape[1]
        ssim = 0
        for i in range(m):
            for j in range(n):
                ssim += compare_ssim(ave_image[i, j, :], divide_image[i, j, :], win_size=11, data_range=255,
                                     channel_axis=2)
        if (ssim < 1.465):
            if self.transform is not None:
                img = self.transform(img)
                label = torch.tensor(self.ids[item], dtype=torch.long)
                cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
                item = torch.tensor(item, dtype=torch.long)
        else:
            if self.transform is not None:
                img = self.transform(img)
                img = cutmix(img, 0.3)
                label = torch.tensor(self.ids[item], dtype=torch.long)
                cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
                item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, path, item


class MarketDataset(Dataset):
    def __init__(self, root, mode='train', transform=None):
        assert os.path.isdir(root)
        assert mode in ['train', 'gallery', 'query']

        self.transform = transform

        if mode == 'train':
            img_paths = glob(os.path.join(root, 'bounding_box_train/*.jpg'), recursive=True)
        elif mode == 'gallery':
            img_paths = glob(os.path.join(root, 'bounding_box_test/*.jpg'), recursive=True)
        elif mode == 'query':
            img_paths = glob(os.path.join(root, 'query/*.jpg'), recursive=True)
        
        pattern = re.compile(r'([-\d]+)_c(\d)')
        all_pids = {}
        relabel = mode == 'train'
        self.img_paths = []
        self.cam_ids = []
        self.ids = []
        for fpath in img_paths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            self.img_paths.append(fpath)
            self.ids.append(all_pids[pid])
            self.cam_ids.append(cam - 1)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        path = self.img_paths[item]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, path, item
