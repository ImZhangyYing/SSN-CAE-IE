from __future__ import print_function
import os
import numpy as np
import torch
from PIL import Image
import time
import matplotlib.pyplot as plt

start = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = './test_datasetes/'
files = os.listdir(path)
eps = 1e-8

ae_model_path = './model/ae/net_1000.pth'
fuse_model_path = './model/fuse/f_net_1000.pth'
save_path = './results/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

net = torch.load(ae_model_path)
f_net = torch.load(fuse_model_path)

for i, f in enumerate(files):
    i += 1
    print(i, f)
    route = f + '/'

    imgA1_path = path + route + '1.bmp'
    imgB1_path = path + route + '2.bmp'


    imgA1 = Image.open(imgA1_path)
    imgA1 = imgA1.convert('L')
    imgA1 = np.asarray(imgA1)
    imgA1 = np.atleast_3d(imgA1).transpose(2, 0, 1).astype(float)

    imgB1 = Image.open(imgB1_path)
    imgB1 = imgB1.convert('L')
    imgB1 = np.asarray(imgB1)
    imgB1 = np.atleast_3d(imgB1).transpose(2, 0, 1).astype(float)

    c, Row, Col = imgA1.shape
    imgA1 = imgA1 / float(255)

    c, Row, Col = imgB1.shape
    imgB1 = imgB1 / float(255)

    imgA1 = torch.from_numpy(imgA1).float()
    imgA1 = imgA1.view(1, 1, Row, Col)
    imgB1 = torch.from_numpy(imgB1).float()
    imgB1 = imgB1.view(1, 1, Row, Col)

    f_net = f_net.to(device)
    img_1, img_2 = imgA1.to(device), imgB1.to(device)

    outc1, outc2, out1, out2 = net(img_1, img_2)
    w1_1, w1_2, w1_3, w2_1, w2_2, w2_3, p1, p2 = f_net(outc1, outc2)
    out_image = p1*img_1 + p2*img_2

    ##############################################

    out1 = out_image.cpu()
    out_img = out1.data[0]
    out_img = out_img.squeeze()
    out_img = out_img.numpy()

    plt.imsave(save_path + f + '.bmp', out_img, cmap='gray')



