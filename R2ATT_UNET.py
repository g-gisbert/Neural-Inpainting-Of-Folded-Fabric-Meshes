import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import time

import helpers


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, t, v, tb):
        self.end = time.time()
        print(f"{self.name}: {self.end - self.start}s")

class Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU())

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, is_res=True):
        super(ConvBlock, self).__init__()
        self.conv1 = Conv(in_ch, out_ch)
        self.conv2 = Conv(out_ch, out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.is_res = is_res

    def forward(self, x):
        x = self.conv1(x)
        y = self.conv2(x)
        y = self.conv3(y)
        y = self.bn(y)
        if self.is_res:
            y += x
        return self.relu(y)


class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, attn=None):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.conv = ConvBlock(in_ch, out_ch)
        self.attn = attn

    def forward(self, x, bridge):
        x = self.deconv(x)
        if self.attn:
            bridge = self.attn(bridge, x)
        x = torch.cat([x, bridge], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, depth=5):
        super(Encoder, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.convs = nn.ModuleList()
        for _ in range(depth):
            self.convs.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
            out_ch *= 2

    def forward(self, x):
        res = []
        for i, m in enumerate(self.convs):
            if i > 0:
                x = self.pool(x)
            x = m(x)
            res.append(x)
        return res


class Attn(nn.Module):
    '''
    Attention U-Net: Learning Where to Look for the Pancreas
    https://arxiv.org/pdf/1804.03999.pdf
    '''

    def __init__(self, ch):
        super(Attn, self).__init__()
        self.wx = nn.Conv2d(ch, ch, 1)
        self.wg = nn.Conv2d(ch, ch, 1)
        self.psi = nn.Conv2d(ch, ch, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        identity = x
        x = self.wx(x)
        g = self.wg(g)
        x = self.relu(x + g)
        x = self.psi(x)
        x = self.sigmoid(x)
        return identity * (x + 1)


class Decoder(nn.Module):
    def __init__(self, in_ch=1024, depth=4, attn=True):
        super(Decoder, self).__init__()
        self.depth = depth
        self.deconvs = nn.ModuleList()
        for _ in range(depth):
            self.deconvs.append(DeconvBlock(in_ch, in_ch // 2, Attn(in_ch // 2) if attn else None))
            in_ch //= 2

    def forward(self, x_list):
        for i in range(self.depth):
            if i == 0:
                x = x_list.pop()
            bridge = x_list.pop()
            x = self.deconvs[i](x, bridge)
        return x


class ScalarEmbeddingModel(nn.Module):
    def __init__(self, input_size=1, embedding_size=16, output_size=1):
        super(ScalarEmbeddingModel, self).__init__()
        self.embedding_layer = nn.Linear(input_size, embedding_size)
        self.output_layer = nn.Linear(embedding_size, output_size)

    def forward(self, x):
        x = x.view(-1, 1)
        x = 2.0*x - 1.0
        embedded_scalar = torch.relu(self.embedding_layer(x))
        output_scalar = self.output_layer(embedded_scalar)

        return output_scalar

class AUNet(nn.Module):

    def __init__(self, in_ch=4, out_ch=3, encoder_depth=5, regressive=True, attn=True):#5
        super(AUNet, self).__init__()
        self.encoder = Encoder(in_ch, 64, encoder_depth) 
        self.decoder = Decoder(1024, encoder_depth - 1, attn)
        self.conv = nn.Conv2d(64, out_ch, 1) 
        self.sigmoid = nn.Sigmoid()
        self.regressive = regressive


        self._init_weights()
        self.im_pred = None

    def forward(self, input):# curv

        x = self.encoder(input)
        x = self.decoder(x)
        x = self.conv(x)

        self.im_pred = x


    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def loss(self, im_gt, conf):
        N = 64
        batch_size = im_gt.shape[0]
        device = self.im_pred.device
        alpha = conf['alpha']
        beta = conf['beta']
        gamma = conf['gamma']

        losses = {}
        loss_mse = ((im_gt - self.im_pred) ** 2).mean()
        losses['loss_mse'] = gamma * loss_mse

        diff_x = torch.cat((torch.ones(batch_size, 3, N, 1, device=device), self.im_pred), 3) - \
                 torch.cat((self.im_pred, torch.ones(batch_size, 3, N, 1, device=device)), 3)
        diff_x_loss = diff_x[:, :, :, 1:-1]

        diff_y = torch.cat((torch.ones(batch_size, 3, 1, N, device=device), self.im_pred), 2) - \
                 torch.cat((self.im_pred, torch.ones(batch_size, 3, 1, N, device=device)), 2)
        diff_y_loss = diff_y[:, :, 1:-1, :]

        diff_x_gt = torch.cat((torch.ones(batch_size, 3, N, 1, device=device), im_gt), 3) - \
                    torch.cat((im_gt, torch.ones(batch_size, 3, N, 1, device=device)), 3)
        diff_x_loss_gt = diff_x_gt[:, :, :, 1:-1]

        diff_y_gt = torch.cat((torch.ones(batch_size, 3, 1, N, device=device), im_gt), 2) - \
                    torch.cat((im_gt, torch.ones(batch_size, 3, 1, N, device=device)), 2)
        diff_y_loss_gt = diff_y_gt[:, :, 1:-1, :]

        loss_reg = (diff_x_loss - diff_x_loss_gt).norm(p=2, dim=1).square().mean() + (diff_y_loss - diff_y_loss_gt).norm(p=2, dim=1).square().mean()
        losses['loss_reg'] = alpha * loss_reg

        mean_diff = 0.5 * diff_x_loss_gt.norm(p=2, dim=1).mean() + 0.5 * diff_y_loss_gt.norm(p=2, dim=1).mean()
        distance_matrix = torch.cdist(self.im_pred.view(-1, 3, N * N).transpose(1, 2), self.im_pred.view(-1, 3, N * N).transpose(1, 2), p=2)
        diagonal_mask = torch.eye(N*N, N*N, device='cuda') * mean_diff/1.5
        distance_matrix = distance_matrix + diagonal_mask.expand(batch_size, -1, -1)
        loss_interpen = torch.nn.functional.relu(mean_diff/1.5 - distance_matrix).square().sum()
        losses['loss_interpen'] = beta * loss_interpen
        losses['loss_interpen'] = torch.tensor([0.0])


        losses['loss_tot'] = gamma * loss_mse + alpha * loss_reg + beta * loss_interpen
        return losses
