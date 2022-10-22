import os
import math
import argparse
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from vit_cnn import vit_base_patch16_224_in21k
from dataset import fusiondata
import numpy as np
import torch.nn as nn

plot_loss = []
N = 1001

def main(args):
    device = torch.device('cuda')
    model = vit_base_patch16_224_in21k().to(device)

    batch_size = 1
    root_path = './train_datasets/'
    dataset = fusiondata(os.path.join(root_path))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    training_data_loader = DataLoader(dataset=dataset, num_workers=nw, batch_size=batch_size, shuffle=True)


    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))



    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)  # 每step_size个epoch乘Gamma


    for epoch in range(N):
        # train
        for iteration, batch in enumerate(training_data_loader, 1):
            imgA1, imgA2, detected_img1, detected_img2, pc_img1, pc_img2, g_img1, g_img2 = batch[0], batch[1], batch[2],batch[3], batch[4], batch[5], batch[6], batch[7]
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
            ##############################################
            a = 1
            b = 10
            eps = 1e-8
            ##############################################
            u1 = detected_img1 * l_img1
            u2 = detected_img2 * l_img2
            v1 = pc_img1 * g_img1
            v2 = pc_img2 * g_img2
            w_1 = (u1 ** a + v1 ** b + eps) / ((u1 ** a + v1 ** b) + (u2 ** a + v2 ** b) + eps)
            w_2 = 1 - w_1

            ##############################################
            outc1, outc2, out1, out2 = model(img_A, img_B)


            loss1 = ((out1 - img_A)).norm(2) + ((out2 - img_B)).norm(2)
            loss_c = ((out1 - img_B)).norm(2) + ((out2 - img_A)).norm(2) + 2*(((out1 - out2)).norm(2))#contrastive loss
            loss = loss1 - 0.1*loss_c

            print("===> Epoch[{}]({}/{}): Loss: {:.4f} ".format(epoch, iteration, len(training_data_loader), loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 500 == 0 and iteration % len(training_data_loader) == 0:
                torch.save(model, "./model/pre-ae/net_{}.pth".format(epoch))
                plot_loss.append(loss.item())
                train_loss_out_path = 'model/pre-ae/loss_{}.npy'.format(epoch)
                m = np.array(plot_loss)
                np.save(train_loss_out_path, m)
    # plot_curve(plot_loss)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0001) # 0.001
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--data-path', type=str,
                        default="/data/flower_photos")
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    main(opt)
