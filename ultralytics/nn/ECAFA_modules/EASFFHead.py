import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.tal import dist2bbox, make_anchors

__all__ = ['EASFFHead']

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class FASFF(nn.Module):
    """
    支持 3 路(P2,P3,P4) 或 4 路(P2,P3,P4,P5) 的多尺度融合模块。
    - 4 路时，level=0..3，与你原实现一致；
    - 3 路时，level=0..2，分别输出增强后的 P4、P3、P2。
    """
    def __init__(self, level, ch, multiplier=1, rfb=False, vis=False):
        super(FASFF, self).__init__()
        self.level = level
        self.vis = vis
        self.n_scales = len(ch)  # 3 或 4

        if self.n_scales == 4:
            # 原实现：按 [P5,P4,P3,P2] 的通道列表（从大到小）
            self.dim = [int(ch[3] * multiplier), int(ch[2] * multiplier),
                        int(ch[1] * multiplier), int(ch[0] * multiplier)]
            self.inter_dim = self.dim[self.level]

            # ---- 原 4 路构造，保持不变 ----
            if level == 0:  # 输出增强后的 P5（融合 P5,P4,P3）
                self.stride_level_1 = Conv(int(ch[2] * multiplier), self.inter_dim, 3, 2)  # P4 -> P5 对齐
                self.stride_level_2 = Conv(int(ch[1] * multiplier), self.inter_dim, 3, 2)  # 下采样后的 P3 -> P5 对齐
                self.expand = Conv(self.inter_dim, int(ch[3] * multiplier), 3, 1)
            elif level == 1:  # 输出增强后的 P4（融合 P5,P4,P3）
                self.compress_level_0 = Conv(int(ch[3] * multiplier), self.inter_dim, 1, 1)
                self.stride_level_2 = Conv(int(ch[1] * multiplier), self.inter_dim, 3, 2)
                self.expand = Conv(self.inter_dim, int(ch[2] * multiplier), 3, 1)
            elif level == 2:  # 输出增强后的 P3（融合 P4,P3,P2）
                self.compress_level_0 = Conv(int(ch[2] * multiplier), self.inter_dim, 1, 1)
                self.stride_level_2 = Conv(int(ch[0] * multiplier), self.inter_dim, 3, 2)
                self.expand = Conv(self.inter_dim, int(ch[1] * multiplier), 3, 1)
            elif level == 3:  # 输出增强后的 P2（融合 P4,P3,P2）
                self.compress_level_0 = Conv(int(ch[2] * multiplier), self.inter_dim, 1, 1)
                self.compress_level_1 = Conv(int(ch[1] * multiplier), self.inter_dim, 1, 1)
                self.expand = Conv(self.inter_dim, int(ch[0] * multiplier), 3, 1)

        elif self.n_scales == 3:
            # 仅 P2/P3/P4：传入 ch 顺序为 [C2, C3, C4]
            # 为了让 level 与输出尺度对齐，用 [C4, C3, C2]
            self.dim = [int(ch[2] * multiplier), int(ch[1] * multiplier), int(ch[0] * multiplier)]  # [C4,C3,C2]
            assert 0 <= self.level <= 2, "For 3-scale FASFF, level must be 0..2"
            self.inter_dim = self.dim[self.level]

            # ---- 3 路构造（轻改，不影响 4 路）----
            if level == 0:  # 输出 P4 (40x40)
                self.stride_p3 = Conv(self.dim[1], self.inter_dim, 3, 2)      # P3: 80->40
                self.pool_p2 = nn.MaxPool2d(3, 2, 1)                           # P2: 160->80
                self.stride_p2 = Conv(self.dim[2], self.inter_dim, 3, 2)      # 80->40
                self.expand = Conv(self.inter_dim, self.dim[0], 3, 1)         # C4
            elif level == 1:  # 输出 P3 (80x80)
                self.compress_p4 = Conv(self.dim[0], self.inter_dim, 1, 1)     # C4->inter
                self.stride_p2 = Conv(self.dim[2], self.inter_dim, 3, 2)       # P2:160->80
                self.expand = Conv(self.inter_dim, self.dim[1], 3, 1)          # C3
            else:  # level == 2，输出 P2 (160x160)
                self.compress_p4 = Conv(self.dim[0], self.inter_dim, 1, 1)
                self.up_p4 = nn.Upsample(scale_factor=4, mode='nearest')       # 40->160
                self.compress_p3 = Conv(self.dim[1], self.inter_dim, 1, 1)
                self.up_p3 = nn.Upsample(scale_factor=2, mode='nearest')       # 80->160
                self.expand = Conv(self.inter_dim, self.dim[2], 3, 1)          # C2

        else:
            raise ValueError(f"FASFF expects 3 or 4 scales, got {self.n_scales}")

        # 注意力权重分支（与原实现一致）
        compress_c = 8 if rfb else 16
        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_levels = Conv(compress_c * 3, 3, 1, 1)

    def forward(self, x):
        # === 三路(P2,P3,P4) ===
        if self.n_scales == 3:
            x_p2, x_p3, x_p4 = x[0], x[1], x[2]

            if self.level == 0:  # 输出 P4
                l0 = x_p4
                l1 = self.stride_p3(x_p3)
                l2 = self.stride_p2(self.pool_p2(x_p2))
            elif self.level == 1:  # 输出 P3
                l0 = F.interpolate(self.compress_p4(x_p4), scale_factor=2, mode='nearest')
                l1 = x_p3
                l2 = self.stride_p2(x_p2)
            else:  # 输出 P2
                l0 = self.up_p4(self.compress_p4(x_p4))
                l1 = self.up_p3(self.compress_p3(x_p3))
                l2 = x_p2

            w0 = self.weight_level_0(l0)
            w1 = self.weight_level_1(l1)
            w2 = self.weight_level_2(l2)
            weights = F.softmax(self.weight_levels(torch.cat((w0, w1, w2), 1)), dim=1)
            fused = l0 * weights[:, 0:1, :, :] + l1 * weights[:, 1:2, :, :] + l2 * weights[:, 2:3, :, :]
            out = self.expand(fused)
            if self.vis:
                return out, weights, fused.sum(dim=1)
            return out

        # === 四路(P2,P3,P4,P5)：原逻辑保持不变 ===
        x_level_add = x[2]   # p4
        x_level_0 = x[3]     # p5
        x_level_1 = x[1]     # p3
        x_level_2 = x[0]     # p2

        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_add)
            level_2_downsampled_inter = F.max_pool2d(x_level_1, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_add
            level_2_resized = self.stride_level_2(x_level_1)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_add)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        else:  # level == 3
            level_0_compressed = self.compress_level_0(x_level_add)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            x_level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(x_level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = F.softmax(self.weight_levels(levels_weight_v), dim=1)

        fused_out_reduced = (
            level_0_resized * levels_weight[:, 0:1, :, :] +
            level_1_resized * levels_weight[:, 1:2, :, :] +
            level_2_resized * levels_weight[:, 2:3, :, :]
        )
        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


class EASFFHead(nn.Module):
    """
    兼容 3 头 或 4 头 的检测头：
      - 传入 ch 为 [C2,C3,C4] → 只用 P2/P3/P4
      - 传入 ch 为 [C2,C3,C4,C5] → 用 P2/P3/P4/P5（原逻辑）
    """
    dynamic = False
    export = False
    end2end = False
    max_det = 300
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, ch=(), multiplier=1, rfb=False):
        super().__init__()
        print("[DEBUG] Entered FASFFHead.__init__")
        self.nc = nc
        self.nl = len(ch)  # 3 或 4
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        # 中间通道：保证 ≥ reg_max*4 (=64)
        def _c2(cin): return max(16, cin // 4, self.reg_max * 4)
        def _c3(cin): return max(cin, min(self.nc, 100))

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(c, _c2(c), 3), Conv(_c2(c), _c2(c), 3), nn.Conv2d(_c2(c), 4 * self.reg_max, 1))
            for c in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(c, c, 3), Conv(c, _c3(c), 1)),
                nn.Sequential(DWConv(_c3(c), _c3(c), 3), Conv(_c3(c), _c3(c), 1)),
                nn.Conv2d(_c3(c), self.nc, 1),
            )
            for c in ch
        )

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        # 仅按需要创建融合器
        self.l0_fusion = FASFF(level=0, ch=ch, multiplier=multiplier, rfb=rfb)
        self.l1_fusion = FASFF(level=1, ch=ch, multiplier=multiplier, rfb=rfb)
        self.l2_fusion = FASFF(level=2, ch=ch, multiplier=multiplier, rfb=rfb)
        self.l3_fusion = FASFF(level=3, ch=ch, multiplier=multiplier, rfb=rfb) if self.nl == 4 else None

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        # x 为特征列表，长度 3 或 4
        if self.nl == 4:
            x1 = self.l0_fusion(x)   # P5'
            x2 = self.l1_fusion(x)   # P4'
            x3 = self.l2_fusion(x)   # P3'
            x4 = self.l3_fusion(x)   # P2'
            x = [x4, x3, x2, x1]     # [P2,P3,P4,P5]
        else:
            # x: [P2,P3,P4]
            x4 = self.l0_fusion(x)   # P4'
            x3 = self.l1_fusion(x)   # P3'
            x2 = self.l2_fusion(x)   # P2'
            x = [x2, x3, x4]         # [P2,P3,P4]

        # 头部
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:
            return x

        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and getattr(self, "format", None) in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and getattr(self, "format", None) in {"tflite", "edgetpu"}:
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):
                a[-1].bias.data[:] = 1.0
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)

    def decode_bboxes(self, bboxes, anchors):
        return dist2bbox(bboxes, anchors, xywh=not self.end2end, dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        B, A, _ = preds.shape
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, A))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, A))
        i = torch.arange(B)[..., None]
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)
