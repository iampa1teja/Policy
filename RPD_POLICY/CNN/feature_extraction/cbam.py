import torch 
import torch.nn as nn 

class ChannelAttention(nn.Module): 
    def __init__(self, ch: int, reduction: int = 16): 
        super().__init__() 
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.max_pool = nn.AdaptiveAvgPool2d(1) 
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, ch // reduction, 1, bias = False), 
            nn.ReLU(), 
            nn.Conv2d(ch // reduction, ch, 1, bias = False),
        )

    def forward(self, x): 
        return torch.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))) 

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True).values

        return torch.sigmoid(
            self.conv(torch.cat([avg, mx], dim=1))
        )

class CBAM(nn.Module): 
    def __init__(self, ch: int , reduction: int = 16):
        super().__init__() 
        self.ca = ChannelAttention(ch, reduction)  
        self.sa = SpatialAttention() 

    def forward(self, x): 
        x = x * self.ca(x) 
        x = x * self.sa(x) 
        return x 