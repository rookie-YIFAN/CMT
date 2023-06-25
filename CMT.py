import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
from torchsummary import summary

class TransEncoder(nn.Module):
    def __init__(self, emb_size=60, proj_size=32, nhead=4):
        super(TransEncoder, self).__init__()

        self.emb_size = emb_size

        self.mulitheadAtt1 = nn.MultiheadAttention(emb_size, nhead, dropout=0.25)

        self.layernorm = nn.LayerNorm(emb_size)

        self.ff1_1 = nn.Linear(emb_size, proj_size)
        self.ff1_2 = nn.Linear(proj_size, emb_size)

        self.dropout = nn.Dropout(0.25)

        self.FF1 = nn.Sequential(
            self.ff1_1,
            nn.ReLU(inplace=True),
            self.ff1_2,
            self.dropout
        )

    def forward(self, x):
        src1, _ = self.mulitheadAtt1(x, x, x)
        src1_n1 = self.layernorm(src1 + x)
        src1_n2 = self.layernorm(self.FF1(src1_n1) + src1_n1)

        return src1_n2


class DEPTHWISECONV(nn.Module):
    def __init__(self, in_ch, out_ch, group):
        super(DEPTHWISECONV, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch ,
                                    kernel_size=3,
                                    stride=1,
                                    padding=0,
                                    groups=group)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=group)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)

        return out[ :, :,1:-1, 1:-1]


class TDFE_G(nn.Module):
    def __init__(self, Patchsz, emb_dim, band, group, group_out):
        '''
        :param Patchsz:  patchSize of input sample
        :param emb_dim:  embedding dimension of single pixel
        :param band: band of input sample
        :param group:  grouping num of band
        :param group_out: embedding dimension for every group feature
        '''

        super(TDFE_G, self).__init__()

        print("TDFE branch running")
        self.name = 'TDFE'

        self.G = group
        self.G_dim = int(band / group)

        self.group_conv_num = 4

        self.group_emb_in = self.G_dim - 2 - 2
        self.group_emb_out = group_out

        self.batchNum1 = Patchsz * Patchsz
        self.batchNum2 = (Patchsz - 4) * (Patchsz - 4)

        self.depthSeperate = nn.Sequential(
            Rearrange("B t w h C -> B (t C) w h"),
            DEPTHWISECONV(band, emb_dim, group),
            Rearrange("B t w h  -> B (w h) t"),
        )

        self.conv1 = nn.Sequential(
            Rearrange("B t w h (C D) -> B (t C) w h D", C=self.G, D=self.G_dim),
            nn.Conv3d(
                in_channels=self.G,
                out_channels=self.G * self.group_conv_num,
                kernel_size=(3, 3, 3),
                groups=self.G),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(self.G * self.group_conv_num, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
        )

        self.conv_m = nn.Sequential(
            Rearrange("B t w h (C D) -> B (t C) w h D", C=self.G, D=self.G_dim),
            nn.Conv3d(
                in_channels=self.G,
                out_channels=self.G * self.group_conv_num,
                kernel_size=(3, 3, 3),
                groups=self.G),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(self.G * self.group_conv_num, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=self.G * self.group_conv_num,
                out_channels=self.G * self.group_conv_num,
                kernel_size=(3, 3, 3),
                groups=self.G),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(self.G * self.group_conv_num, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
        )

        self.conv3 = nn.Sequential(
            Rearrange("B t w h b -> B (t b) w h "),
            nn.Conv2d(in_channels=self.G * self.group_conv_num * self.group_emb_in, out_channels=group_out * self.G,
                      kernel_size=1, padding=0, stride=1, groups=self.G),
            nn.ReLU(inplace=True),
            Rearrange("B a b c -> B  (b c) a"),
        )

        self.conv4 = nn.Sequential(
            nn.Linear(group_out * group, emb_dim),
            nn.ReLU(inplace=True),
        )

        self.conv5 = nn.Sequential(
            Rearrange("B t w h b -> B (w h) (t b) "),
            nn.BatchNorm1d(self.batchNum2),
            nn.Linear(band, emb_dim),
            nn.ReLU(inplace=True)
        )

        self.Linear = nn.Sequential(
            Rearrange("B t w h b -> B (t b) w h"),
            nn.BatchNorm2d(band),
            nn.Conv2d(band, emb_dim, groups=8, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            Rearrange("B o w h  -> B (w h) o")

        )

        self.convMedia = nn.Sequential(
            Rearrange("B (a b) c d e -> B b a (c d) e ", a=8, b=4),
            Rearrange("B a b c d -> B (c a) (b d)"),
            nn.Conv1d(in_channels=81 * 4, out_channels=81 * 2, kernel_size=3, groups=81, padding=1),
            Rearrange("B (a b) c -> B a (b c)", a=81, b=2)
        )


    def forward(self, X):
        middleScale = self.depthSeperate(X);
        x = self.conv1(X)

        x = self.conv2(x)

        x = self.conv3(x)

        x = self.conv4(x)

        y = X[:, :, 2:-2, 2:-2, :]
        sep_residual = self.Linear(y)

        return x + middleScale + sep_residual



class SP_T(nn.Module):
    def __init__(self, TDFE_hyper_params, class_num, patchSZ, emb_size=60, proj_size=32, nhead=4):
        super(SP_T, self).__init__()
        print("MergeT  running")
        self.name = 'MergeT'
        self.NornNum = (patchSZ - 4) * (patchSZ - 4)
        self.localFE = TDFE_G(*TDFE_hyper_params)

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, emb_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))  # 用于分类的token - 创新点

        self.spectral = nn.Sequential(
            # Rearrange("B a b  -> B b a "),
            TransEncoder(emb_size, proj_size, nhead),
            TransEncoder(emb_size, proj_size, nhead)
        )

        self.spectral_cls_layer = nn.Sequential(
            nn.AvgPool2d((self.NornNum, 1), stride=1),
            Rearrange("B a b -> B (a b)"),
            nn.Dropout(0.2),
            nn.Linear(emb_size, class_num)
        )

        self.spectral_cls_layer1 = nn.Sequential(
            nn.AvgPool2d((self.NornNum, 1), stride=1),
            Rearrange("B a b -> B (a b)"),
            nn.Dropout(0.3),
            nn.Linear(emb_size, 64),
            nn.Dropout(0.3),
            nn.Linear(64, class_num),
        )

        self.spectral_cls_layer2 = nn.Sequential(
            nn.AvgPool2d((self.NornNum, 1), stride=1),
            Rearrange("B a b -> B (a b)"),
            nn.Dropout(0.3),
            nn.Linear(emb_size, emb_size),
            nn.Dropout(0.3),
            nn.Linear(emb_size, 64),
            nn.Dropout(0.3),
            nn.Linear(64, class_num),
        )

    def forward(self, x):
        local_res = self.localFE(x)
        local_out = self.spectral(local_res)
        cls_res = self.spectral_cls_layer(local_out)

        return cls_res

    def encoder(self, x):
        local_res = self.localFE(x)
        local_out = self.spectral(local_res)
        return local_out


if __name__ == '__main__':
    DEVICE = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    TDFE_hyper_params = (13, 8 * 16, 80, 8, 16)
    model = SP_T(TDFE_hyper_params, 20, 13, 8 * 16, 4 * 16, 4).cuda()
    summary(model, (1, 13, 13, 80))


