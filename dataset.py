from os import listdir
from os.path import join
import os
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
# from scipy.misc import imread, imresize
import torch
from PIL import Image


def make_dataset(root, train=True):
    dataset = []

    if train:
        dir_img = os.path.join(root)
    # # 自然图像
    # for index in range(8001):
    #     imgA = str(index + 1) + '.bmp'
    #     dataset.append(os.path.join(dir_img, imgA))
    # return dataset


#医学图像
    for index in range(2, 219):
        imgA = str(index) + '_1.bmp'
        imgB = str(index) + '_2.bmp'
        detected_img1 = str(index) + '_3.bmp'
        detected_img2 = str(index) + '_4.bmp'
        PC_img1 = str(index) + '_5.bmp'
        PC_img2 = str(index) + '_6.bmp'
        g_img1 = str(index) + '_7.bmp'
        g_img2 = str(index) + '_8.bmp'
        # dataset.append([os.path.join(dir_img, imgA), os.path.join(dir_img, imgB)])
        dataset.append([os.path.join(dir_img, imgA), os.path.join(dir_img, imgB), os.path.join(dir_img, detected_img1), os.path.join(dir_img, detected_img2), os.path.join(dir_img, PC_img1), os.path.join(dir_img, PC_img2), os.path.join(dir_img, g_img1), os.path.join(dir_img, g_img2)])
    return dataset


class fusiondata(data.Dataset):

    def __init__(self, root, transform=None, train=True):
        self.train = train
        if self.train:
            self.train_set_path = make_dataset(root, train)

    def __getitem__(self, idx):
        if self.train:
            # imgA_path, imgB_path  = self.train_set_path[idx]
            imgA_path, imgB_path, detected_img1_path, detected_img2_path, PC_img1_path, PC_img2_path, g_img1_path, g_img2_path = self.train_set_path[idx]
            self.train_set_path[idx]
            imgA = Image.open(imgA_path)
            imgA = imgA.convert('L')
            imgA = np.asarray(imgA)
            imgA = np.atleast_3d(imgA).transpose(2, 0, 1).astype(float)
            imgA = imgA / float(255)
            imgA = torch.from_numpy(imgA).float()

            imgB = Image.open(imgB_path)
            imgB = imgB.convert('L')
            imgB = np.asarray(imgB)
            imgB = np.atleast_3d(imgB).transpose(2, 0, 1).astype(float)
            imgB = imgB / float(255)
            imgB = torch.from_numpy(imgB).float()

            detected_img1 = Image.open(detected_img1_path)
            detected_img1 = detected_img1.convert('L')
            detected_img1 = np.asarray(detected_img1)
            detected_img1 = np.atleast_3d(detected_img1).transpose(2, 0, 1).astype(np.float)
            detected_img1 = detected_img1 / float(255)
            detected_img1 = torch.from_numpy(detected_img1).float()

            detected_img2 = Image.open(detected_img2_path)
            detected_img2 = detected_img2.convert('L')
            detected_img2 = np.asarray(detected_img2)
            detected_img2 = np.atleast_3d(detected_img2).transpose(2, 0, 1).astype(np.float)
            detected_img2 = detected_img2 / float(255)
            detected_img2 = torch.from_numpy(detected_img2).float()

            PC_img1 = Image.open(PC_img1_path)
            PC_img1 = PC_img1.convert('L')
            PC_img1 = np.asarray(PC_img1)
            PC_img1 = np.atleast_3d(PC_img1).transpose(2, 0, 1).astype(np.float)
            PC_img1 = PC_img1 / float(255)
            PC_img1 = torch.from_numpy(PC_img1).float()

            PC_img2 = Image.open(PC_img2_path)
            PC_img2 = PC_img2.convert('L')
            PC_img2 = np.asarray(PC_img2)
            PC_img2 = np.atleast_3d(PC_img2).transpose(2, 0, 1).astype(np.float)
            PC_img2 = PC_img2 / float(255)
            PC_img2 = torch.from_numpy(PC_img2).float()

            g_img1 = Image.open(g_img1_path)
            g_img1 = g_img1.convert('L')
            g_img1 = np.asarray(g_img1)
            g_img1 = np.atleast_3d(g_img1).transpose(2, 0, 1).astype(np.float)
            g_img1 = g_img1 / float(255)
            g_img1 = torch.from_numpy(g_img1).float()

            g_img2 = Image.open(g_img2_path)
            g_img2 = g_img2.convert('L')
            g_img2 = np.asarray(g_img2)
            g_img2 = np.atleast_3d(g_img2).transpose(2, 0, 1).astype(np.float)
            g_img2 = g_img2 / float(255)
            g_img2 = torch.from_numpy(g_img2).float()

            return imgA, imgB, detected_img1, detected_img2, PC_img1, PC_img2, g_img1, g_img2
            # return imgA, imgB

    def __len__(self):
        if self.train:
            return 217

class Singledata(data.Dataset):

    def __init__(self, root, transform=None, train=True):
        self.train = train
        if self.train:
            self.train_set_path = make_dataset(root, train)

    def __getitem__(self, idx):
        if self.train:
            # imgA_path, imgB_path  = self.train_set_path[idx]
            imgA_path, imgB_path, detected_img1_path, detected_img2_path, PC_img1_path, PC_img2_path, g_img1_path, g_img2_path = self.train_set_path[idx]
            self.train_set_path[idx]
            imgA = Image.open(imgA_path)
            imgA = imgA.convert('L')
            imgA = np.asarray(imgA)
            imgA = np.atleast_3d(imgA).transpose(2, 0, 1).astype(float)
            imgA = imgA / float(255)
            imgA = torch.from_numpy(imgA).float()

            imgB = Image.open(imgB_path)
            imgB = imgB.convert('L')
            imgB = np.asarray(imgB)
            imgB = np.atleast_3d(imgB).transpose(2, 0, 1).astype(float)
            imgB = imgB / float(255)
            imgB = torch.from_numpy(imgB).float()

            detected_img1 = Image.open(detected_img1_path)
            detected_img1 = detected_img1.convert('L')
            detected_img1 = np.asarray(detected_img1)
            detected_img1 = np.atleast_3d(detected_img1).transpose(2, 0, 1).astype(np.float)
            detected_img1 = detected_img1 / float(255)
            detected_img1 = torch.from_numpy(detected_img1).float()

            detected_img2 = Image.open(detected_img2_path)
            detected_img2 = detected_img2.convert('L')
            detected_img2 = np.asarray(detected_img2)
            detected_img2 = np.atleast_3d(detected_img2).transpose(2, 0, 1).astype(np.float)
            detected_img2 = detected_img2 / float(255)
            detected_img2 = torch.from_numpy(detected_img2).float()

            PC_img1 = Image.open(PC_img1_path)
            PC_img1 = PC_img1.convert('L')
            PC_img1 = np.asarray(PC_img1)
            PC_img1 = np.atleast_3d(PC_img1).transpose(2, 0, 1).astype(np.float)
            PC_img1 = PC_img1 / float(255)
            PC_img1 = torch.from_numpy(PC_img1).float()

            PC_img2 = Image.open(PC_img2_path)
            PC_img2 = PC_img2.convert('L')
            PC_img2 = np.asarray(PC_img2)
            PC_img2 = np.atleast_3d(PC_img2).transpose(2, 0, 1).astype(np.float)
            PC_img2 = PC_img2 / float(255)
            PC_img2 = torch.from_numpy(PC_img2).float()

            g_img1 = Image.open(g_img1_path)
            g_img1 = g_img1.convert('L')
            g_img1 = np.asarray(g_img1)
            g_img1 = np.atleast_3d(g_img1).transpose(2, 0, 1).astype(np.float)
            g_img1 = g_img1 / float(255)
            g_img1 = torch.from_numpy(g_img1).float()

            g_img2 = Image.open(g_img2_path)
            g_img2 = g_img2.convert('L')
            g_img2 = np.asarray(g_img2)
            g_img2 = np.atleast_3d(g_img2).transpose(2, 0, 1).astype(np.float)
            g_img2 = g_img2 / float(255)
            g_img2 = torch.from_numpy(g_img2).float()

            return imgA, imgB, detected_img1, detected_img2, PC_img1, PC_img2, g_img1, g_img2
            # return imgA, imgB

    def __len__(self):
        if self.train:
            return 1