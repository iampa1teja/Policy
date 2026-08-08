import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class DepthwiseSepConv(nn.Module): 
    def __init__(self, ch): 
        super().__init__() 
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias = False)
        self.pw = nn.Conv2d(ch, ch, 1, bias = False) 
        self.bn = nn.BatchNorm2d(ch) 

    def forward(self, x): 
        return F.relu(self.bn(self.pw(self.dw(x))), inplace=True) 


class BiFPNLayer(nn.Module): 
    def __init__(self, ch: int, num_levels: int = 3, eps: float = 1e-4): 
        super().__init__()
        self.eps        = eps
        self.num_levels = num_levels
        self.td_weights = nn.ParameterList(
            [nn.Parameter(torch.ones(2)) for _ in range(num_levels - 1)])
        self.bu_weights = nn.ParameterList(
            [nn.Parameter(torch.ones(3)) for _ in range(num_levels - 1)])
        self.td_convs   = nn.ModuleList(
            [DepthwiseSepConv(ch) for _ in range(num_levels - 1)])
        self.bu_convs   = nn.ModuleList(
            [DepthwiseSepConv(ch) for _ in range(num_levels - 1)])

    def _fuse2(self, w, x1, x2):
        w1, w2 = F.relu(w[0]), F.relu(w[1])
        return (
            (w1 * x1 + w2 * x2)
            / (w1 + w2 + self.eps)
        )
    
    def _fuse3(self, w, x1, x2, x3): 
        w1, w2, w3 = F.relu(w[0]), F.relu(w[1]), F.relu(w[2]) 
        return (w1 * x1 + w2 * x2 + w3 * x3) / (w1 + w2 + w3 + self.eps) 

    def forward(self, features): 
        P = list(features)
        N = self.num_levels
        td     = [None] * N
        td[-1] = P[-1]
        for i in range(N - 2, -1, -1):
            up    = F.interpolate(td[i+1], size=P[i].shape[-2:], mode='nearest')
            td[i] = self.td_convs[i](self._fuse2(self.td_weights[i], P[i], up))
        out    = [None] * N
        out[0] = td[0]
        for i in range(1, N):
            down = F.adaptive_max_pool2d(out[i-1], output_size=td[i].shape[-2:])
            out[i] = self.bu_convs[i-1](self._fuse3(self.bu_weights[i-1], P[i], td[i], down))
        return out

class BiFPN(nn.Module): 
    def __init__(self, in_channels: list, out_ch: int = 256, num_layers: int = 3, eps: float = 1e-4): 
        super().__init__() 
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_ch, 1, bias = False), 
                nn.BatchNorm2d(out_ch), 
                nn.ReLU(inplace=True) 
            ) for c in in_channels
        ])
        self.layers = nn.ModuleList([
            BiFPNLayer(out_ch, num_levels=len(in_channels), eps = eps) 
            for _ in range(num_layers) 
        ])

    def forward(self, features): 
        x = [proj(f) for proj, f in zip(self.input_proj, features)]
        for layer in self.layers: 
            x = layer(x) 
        return x 
 