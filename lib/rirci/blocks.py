from torch import nn


class ECABlock(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


import torch
import torch.nn.functional as F
from torch import nn


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


def conv3x3(in_channels, out_channels, stride=1, padding=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=False,
        groups=groups,
        dilation=dilation)


def up_conv3x3(in_channels, out_channels):
    return nn.Sequential(
        nn.Upsample(mode='bilinear', scale_factor=2),
        conv3x3(in_channels, out_channels))


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, dilations=None):
        super(UpConv, self).__init__()
        self.up_conv = up_conv3x3(in_channels, out_channels)
        self.norm0 = nn.BatchNorm2d(out_channels)

        if not dilations:
            dilations = [1] * blocks

        self.conv1 = conv3x3(2 * out_channels, out_channels)
        self.norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.ModuleList([
            conv3x3(out_channels, out_channels, dilation=d, padding=d)
            for d in dilations
        ])

        self.bn = nn.ModuleList([
            nn.BatchNorm2d(out_channels)
            for _ in range(blocks)
        ])

    def forward(self, from_up, from_down):
        from_up = F.relu(self.norm0(self.up_conv(from_up)))
        x1 = torch.cat((from_up, from_down), 1)
        xfuse = x1 = F.relu(self.norm1(self.conv1(x1)))

        x2 = None
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1)
            x2 = self.bn[idx](x2)
            x2 = x2 + x1  # residual connection
            x2 = F.relu(x2)
            x1 = x2

        return x2


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, pooling=True, dilations=None):
        super(DownConv, self).__init__()
        self.pooling = pooling
        self.conv1 = conv3x3(in_channels, out_channels)
        self.norm1 = nn.BatchNorm2d(out_channels)

        if not dilations:
            dilations = [1] * blocks

        self.conv2 = nn.ModuleList([
            conv3x3(out_channels, out_channels, dilation=d, padding=d)
            for d in dilations
        ])

        self.bn = nn.ModuleList([
            nn.BatchNorm2d(out_channels)
            for _ in range(blocks)
        ])

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = F.relu(self.norm1(self.conv1(x)))
        x2 = None

        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1)
            x2 = self.bn[idx](x2)
            x2 = x2 + x1  # residual connection
            x2 = F.relu(x2)
            x1 = x2

        before_pool = x2
        if self.pooling:
            x2 = self.pool(x2)
        return x2, before_pool


class MBEBlock(nn.Module):
    def __init__(self, in_channels=512, out_channels=3, blocks=1):
        super(MBEBlock, self).__init__()
        self.up_conv = up_conv3x3(in_channels, out_channels)
        self.norm0 = nn.BatchNorm2d(out_channels)
        self.conv1 = conv3x3(2 * out_channels, out_channels)
        self.norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.ModuleList()
        self.conv3 = nn.ModuleList()
        for _ in range(blocks):
            self.conv2.append(
                nn.Sequential(
                    nn.Conv2d(out_channels // 2 + 1, out_channels // 4, 5, 1, 2),
                    nn.ReLU(True),
                    nn.Conv2d(out_channels // 4, 1, 5, 1, 2),
                    nn.Sigmoid()
                )
            )
            self.conv3.append(conv3x3(out_channels // 2, out_channels))

        self.bn = nn.ModuleList([
            nn.BatchNorm2d(out_channels)
            for _ in range(blocks)
        ])

    def forward(self, from_up, from_down, mask):
        from_up = F.relu(self.norm0(self.up_conv(from_up)))
        x1 = torch.cat((from_up, from_down), 1)
        x1 = F.relu(self.norm1(self.conv1(x1)))

        _, C, H, W = x1.shape
        for idx, (mask_conv, feat_conv) in enumerate(zip(self.conv2, self.conv3)):
            mask = mask_conv(torch.cat([x1[:, :C // 2], mask], dim=1))
            x2_actv = x1[:, C // 2:] * mask
            x2 = feat_conv(x1[:, C // 2:] + x2_actv)
            x2 = self.bn[idx](x2)
            x2 = x2 + x1  # residual connection
            x2 = F.relu(x2)
            x1 = x2

        return x2


class SMRBlock(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=1):
        super(SMRBlock, self).__init__()
        self.threshold = 0.5
        self.upconv = UpConv(in_channels, out_channels, blocks, out_fuse=True)

        self.primary_mask = nn.Sequential(
            nn.Conv2d(out_channels, 1, 1, 1, 0),
            nn.Sigmoid()
        )

        self.refine_branch = nn.Sequential(
            nn.Conv2d(out_channels, 1, 1, 1, 0),
            nn.Sigmoid()
        )

        self.self_calibrated = SelfAttentionSimple(out_channels)

    def forward(self, input, encoder_outs=None):
        mask_x, fuse_x = self.upconv(input, encoder_outs)
        primary_mask = self.primary_mask(mask_x)
        mask_x, self_calibrated_mask = self.self_calibrated(mask_x, mask_x, primary_mask)
        return {"feats": [mask_x], "attn_maps": [primary_mask, self_calibrated_mask]}


class SelfAttentionSimple(nn.Module):
    def __init__(self, in_channel, k_center=1):
        super(SelfAttentionSimple, self).__init__()
        self.k_center = k_center
        self.reduction = 1
        self.project_mode = 'linear'

        self.q_conv = nn.Conv2d(in_channel, in_channel, 1, 1, 0)
        self.k_conv = nn.Conv2d(in_channel, in_channel * k_center, 1, 1, 0)
        self.v_conv = nn.Conv2d(in_channel, in_channel * k_center, 1, 1, 0)

        self.min_area = 100
        self.threshold = 0.5

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 8, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(in_channel // 8, 1, 3, 1, 1)
        )

        self.sim_func = nn.Conv2d(in_channel + in_channel, 1, 1, 1, 0)
        self.k_weight = nn.Parameter(torch.full((1, k_center, 1, 1), fill_value=1, dtype=torch.float32),
                                     requires_grad=True)

    def compute_attention(self, query, key, mask, eps=1):
        b, c, h, w = query.shape
        query_org = query
        query = self.q_conv(query)
        key_in = key
        key = self.k_conv(key_in)
        keys = list(key.split(c, dim=1))

        importance_map = torch.where(mask >= self.threshold, torch.ones_like(mask), torch.zeros_like(mask))
        s_area = torch.clamp_min(torch.sum(importance_map, dim=[2, 3]), self.min_area)[:, 0:1]

        if self.k_center != 2:
            keys = [torch.sum(k * importance_map, dim=[2, 3]) / s_area for k in keys]
        else:
            keys = [
                torch.sum(keys[0] * importance_map, dim=[2, 3]) / s_area,
                torch.sum(keys[1] * (1 - importance_map), dim=[2, 3]) /
                (keys[1].shape[2] * keys[1].shape[3] - s_area + eps)
            ]

        f_query = query
        f_key = [k.reshape(b, c, 1, 1).repeat(1, 1, f_query.size(2), f_query.size(3)) for k in keys]

        attention_scores = []
        for k in f_key:
            combine_qk = torch.cat([f_query, k], dim=1).tanh()
            sk = self.sim_func(combine_qk)
            attention_scores.append(sk)

        s = torch.cat(attention_scores, dim=1)
        s = s.permute(0, 2, 3, 1)

        v = self.v_conv(key_in)
        if self.k_center == 2:
            v_fg = torch.sum(v[:, :c] * importance_map, dim=[2, 3]) / s_area
            v_bg = torch.sum(v[:, c:] * (1 - importance_map), dim=[2, 3]) / (v.shape[2] * v.shape[3] - s_area + eps)
            v = torch.cat([v_fg, v_bg], dim=1)
        else:
            v = torch.sum(v * importance_map, dim=[2, 3]) / s_area

        v = v.reshape(b, self.k_center, c)
        attn = torch.bmm(s.reshape(b, h * w, self.k_center), v).reshape(b, h, w, c).permute(0, 3, 1, 2)
        s = self.out_conv(attn + query)
        return s

    def forward(self, xin, xout, xmask):
        b_num, c, h, w = xin.shape
        attention_score = self.compute_attention(xin, xout, xmask)
        attention_score = attention_score.reshape(b_num, 1, h, w)
        return xout, attention_score.sigmoid()
