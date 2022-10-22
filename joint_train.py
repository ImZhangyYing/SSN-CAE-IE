from dataset import fusiondata
from dataset import Singledata
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from fuse_network import fuse_net



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root_path = './train_datasets/'
ae_model_path = './model/pre-ae/net_1000.pth'
net = torch.load(ae_model_path)
dataset = fusiondata(os.path.join(root_path))
training_data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

f_net = fuse_net().to(device)
optimizer1 = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
optimizer2 = optim.Adam(f_net.parameters(), lr=0.001, betas=(0.9, 0.999))
mse_loss = nn.MSELoss(reduction='mean')


#训练
def train(epoch):
    for iteration, batch in enumerate(training_data_loader, 1):

        imgA1, imgA2, detected_img1, detected_img2, pc_img1, pc_img2, g_img1, g_img2 = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7]

        img_A = imgA1.to(device)
        img_B = imgA2.to(device)
        detected_img1 = detected_img1.to(device)
        detected_img2 = detected_img2.to(device)
        pc_img1 = pc_img1.to(device)
        pc_img2 = pc_img2.to(device)
        g_img1 = g_img1.to(device)
        g_img2 = g_img2.to(device)
        l_img1 = img_A
        l_img2 = img_B

        outc1, outc2, out1, out2 = net(img_A, img_B)
        w1_1, w1_2, w1_3, w2_1, w2_2, w2_3, p1, p2 = f_net(outc1, outc2)
        out_image = p1*img_A + p2*img_B

        a = 1
        b = 40
        eps = 1e-8
        #################################
        u1 = detected_img1 * l_img1
        u2 = detected_img2 * l_img2
        v1 = pc_img1 * g_img1
        v2 = pc_img2 * g_img2
        w_1 = (u1 ** a + v1 ** b + eps) / ((u1 ** a + v1 ** b) + (u2 ** a + v2 ** b) + eps)
        w_2 = 1 - w_1



        loss_i1 = ((out_image - img_A)*w_1).norm(2)
        loss_i2 = ((out_image - img_B)*w_2).norm(2)


        loss_e = ((w1_1 - w2_1).norm(2)+(w1_2 - w2_2).norm(2)+(w1_3 - w2_3).norm(2))*(1/(256*256))
        loss_sum = (loss_i1 + loss_i2) - 0.7 * loss_e

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss_sum.backward()
        optimizer1.step()
        optimizer2.step()
        print("===> Epoch[{}]/({}/{}): loss_mse: {:.4f}".format(epoch, iteration, len(training_data_loader), loss_sum.item()))
        print("===> Epoch[{}]/({}/{}): loss_e: {:.4f}".format(epoch, iteration, len(training_data_loader),
                                                                loss_e.item()))

def checkpoint(epoch):

    net_ae_model_out_path = './model/ae/net_{}.pth'.format(epoch)
    torch.save(net, net_ae_model_out_path)
    print('checkpoint_AE', str(epoch), 'has saved!')

    f_net_model_out_path = './model/fuse/f_net_{}.pth'.format(epoch)
    torch.save(f_net, f_net_model_out_path)
    print('checkpoint', str(epoch), 'has saved!')

if __name__ == '__main__':
    for epoch in range(1001):
        train(epoch)

        if epoch % 100 == 0:
            checkpoint(epoch)