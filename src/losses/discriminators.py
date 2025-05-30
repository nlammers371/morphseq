import torch.nn as nn, torch

# ---------- PatchGAN 70×70 (3-layer) ----------
class PatchD3(nn.Module):
    def __init__(self, in_ch=3, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ch, 4, 2, 1), nn.LeakyReLU(.2, True),
            nn.Conv2d(ch, ch*2, 4, 2, 1),  nn.BatchNorm2d(ch*2), nn.LeakyReLU(.2, True),
            nn.Conv2d(ch*2, ch*4, 4, 2, 1),nn.BatchNorm2d(ch*4), nn.LeakyReLU(.2, True),
            nn.Conv2d(ch*4, 1, 3, 1, 1)
        )
    def forward(self,x): return self.net(x)

# ---------- Multi-Scale PatchGAN ----------
class MultiScaleD(nn.Module):
    def __init__(self, in_ch=3, num_scales=3):
        super().__init__()
        self.scales = nn.ModuleList([PatchD3(in_ch) for _ in range(num_scales)])
        self.down = nn.AvgPool2d(3, 2, 1, count_include_pad=False)
    def forward(self, x):
        out=[]
        for D in self.scales:
            out.append(D(x))
            x = self.down(x)
        return out            # list of logits

# ---------- StyleGAN-2 Discriminator (minimal) ----------
class StyleGAN2D(nn.Module):
    def __init__(self, in_ch=3, channels=64, num_blocks=4):
        super().__init__()
        layers=[]
        c=channels
        for _ in range(num_blocks):
            layers += [
                nn.Conv2d(in_ch, c, 3, 1, 1), nn.LeakyReLU(.2, True),
                nn.Conv2d(c, c, 3, 1, 1),     nn.LeakyReLU(.2, True),
                nn.AvgPool2d(2)
            ]
            in_ch, c = c, min(c*2, 512)
        layers += [nn.Conv2d(c, c, 3, 1, 1), nn.LeakyReLU(.2, True)]
        self.features = nn.Sequential(*layers)
        self.final = nn.Sequential(
            nn.Conv2d(c+1, c, 3, 1, 1),  nn.LeakyReLU(.2, True),
            nn.Conv2d(c, 1, 4)
        )
    def forward(self, x):
        feat = self.features(x)
        # minibatch-std-dev trick
        std = feat.std(dim=0, keepdim=True).mean([1,2,3], keepdim=True)
        feat = torch.cat([feat, std.expand_as(feat[:,:1])], 1)
        return self.final(feat).view(-1)