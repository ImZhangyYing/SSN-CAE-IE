import torch.nn as nn
import torch
# from Attention import ChannelAttention
# from Attention import SpatialAttention
# from Attention import CA
# from Supervise_Attention import SAM
# from Subspace_Attention import SubspaceAttention as SSA

class fuse_net(nn.Module):
    def __init__(self):
        super(fuse_net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.GroupNorm(2, 64),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(64),
            nn.GroupNorm(2, 64),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=7, stride=1, padding=3),
            # nn.BatchNorm2d(64),
            nn.GroupNorm(2, 64),
            nn.ReLU(inplace=True),
        )

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3),
            # nn.BatchNorm2d(128),
            nn.GroupNorm(2, 128),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            nn.GroupNorm(2, 64),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            nn.GroupNorm(2, 64),
            nn.ReLU(inplace=True),
        )


        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            nn.GroupNorm(2, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )


    def forward(self, outc1, outc2):
        y1_1 = self.conv1(outc1)
        y1_2 = self.conv2(outc1)
        y1_3 = self.conv3(outc1)
        y2_1 = self.conv1(outc2)
        y2_2 = self.conv2(outc2)
        y2_3 = self.conv3(outc2)

        u1 = torch.cat((y1_1, y2_1), 1)
        U1 = self.conv1_1(u1)
        w1_1, w2_1 = torch.split(U1, 64, 1)
        u2 = torch.cat((y1_2, y2_2), 1)
        U2 = self.conv1_1(u2)
        w1_2, w2_2 = torch.split(U2, 64, 1)
        u3 = torch.cat((y1_3, y2_3), 1)
        U3 = self.conv1_1(u3)
        w1_3, w2_3 = torch.split(U3, 64, 1)

        y1 = torch.cat((w1_1, w1_2), 1)
        y2 = torch.cat((w2_1, w2_2), 1)
        out1_1 = self.conv4(y1)
        out2_1 = self.conv4(y2)
        out1_2 = torch.cat((out1_1, w1_3), 1)
        out2_2 = torch.cat((out2_1, w2_3), 1)
        out1_3 = self.conv5(out1_2)
        out2_3 = self.conv5(out2_2)

        p1 = self.decoder(out1_3)
        p2 = self.decoder(out2_3)

        return w1_1, w1_2, w1_3, w2_1, w2_2, w2_3, p1, p2


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

