import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

__all__ = ['CAFA','RepVGG']

# -----------------------------
# 基础工具函数
# -----------------------------
def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor


# -----------------------------
# RepVGG 依赖
# -----------------------------
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels)
    )


class SEBlock(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        internal = input_channels // 8
        self.down = nn.Conv2d(input_channels, internal, 1)
        self.up = nn.Conv2d(internal, input_channels, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.down(w))
        w = torch.sigmoid(self.up(w))
        return x * w


class RepVGG(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1):
        super().__init__()
        self.rbr_dense = conv_bn(c1, c2, k, s, p)
        self.rbr_1x1 = conv_bn(c1, c2, 1, s, p - k // 2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.rbr_dense(x) + self.rbr_1x1(x))


# -----------------------------
# 交错合并
# -----------------------------
def interleave_merge(x1, x2):
    b, c, h, w = x1.shape
    out = torch.stack((x1, x2), dim=2)
    return out.view(b, c * 2, h, w)


# -----------------------------
# OESRInterleave
# -----------------------------
class OESRInterleave(nn.Module):
    def __init__(self, c1, c2, mode='odd_keep'):
        super().__init__()
        assert c1 % 2 == 0 and c2 % 2 == 0
        self.mode = mode
        self.repconv = RepVGG(c1 // 2, c2 // 2)

    def forward(self, x):
        x_odd = x[:, 0::2, :, :]
        x_even = x[:, 1::2, :, :]

        if self.mode == 'odd_keep':
            y_keep = x_odd
            y_trans = self.repconv(x_even)
        else:
            y_keep = x_even
            y_trans = self.repconv(x_odd)

        return interleave_merge(y_keep, y_trans)


# -----------------------------
# ⭐ 目标类：CAFA
# -----------------------------
class CAFA(nn.Module):
    def __init__(self, c1, c2, n=1, se=False, e=0.5):
        super().__init__()
        n_ = max(n // 2, 1)
        c_ = make_divisible(int(c1 * e), 8)

        assert c_ % 2 == 0

        self.conv1 = RepVGG(c1, c_)

        self.oesr1 = nn.Sequential(*[
            OESRInterleave(c_, c_, 'odd_keep') for _ in range(n_)
        ])

        self.oesr2 = nn.Sequential(*[
            OESRInterleave(c_, c_, 'even_keep') for _ in range(n_)
        ])

        self.conv3 = RepVGG(c_ * 3, c2)
        self.se = SEBlock(c2) if se else nn.Identity()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.oesr1(x1)
        x3 = self.oesr2(x2)

        out = torch.cat((x1, x2, x3), dim=1)
        out = self.conv3(out)
        return self.se(out)


# -----------------------------
# 测试
# -----------------------------
if __name__ == "__main__":
    x = torch.randn(1, 64, 80, 80)
    model = CAFA(64, 64)
    y = model(x)
    print(y.shape)