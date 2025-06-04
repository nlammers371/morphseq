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



# ----------------------------------------------------------------------
# 1.  StyleGAN-2 discriminator, depth configurable
#    • 4 blocks  ≈ original “PatchD3-style”
#    • 8 blocks  ≈ heavy version for 48 GB
# ----------------------------------------------------------------------
class StyleGAN2Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(in_ch, out_ch, kernel_size, 1, kernel_size//2))
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(out_ch, out_ch, kernel_size, 1, kernel_size//2))
        self.down  = nn.AvgPool2d(2)
        self.act   = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return self.down(x)

class StyleGAN2D(nn.Module):
    """
    StyleGAN-2 discriminator with N residual-downsample blocks.
    num_blocks=4 (64→8)  : fits any GPU
    num_blocks=8 (256→1) : heavier, but <3 GB @288×128, batch-32, AMP
    """
    def __init__(self, in_ch=3, fmap_base=64, num_blocks=4):
        super().__init__()
        chans = [fmap_base * min(2**i, 16) for i in range(num_blocks+1)]
        self.blocks = nn.ModuleList()
        self.blocks.append(nn.utils.spectral_norm(
            nn.Conv2d(in_ch, chans[0], 1)))          # from-RGB

        for i in range(num_blocks):
            self.blocks.append(StyleGAN2Block(chans[i], chans[i+1]))

        self.final_conv = nn.utils.spectral_norm(
            nn.Conv2d(chans[-1], chans[-1], 3, 1, 1))
        self.final_dense = nn.utils.spectral_norm(
            nn.Linear(chans[-1], 1))

        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        x = self.act(self.final_conv(x))
        x = x.mean([2, 3])              # global pooling
        return self.final_dense(x).unsqueeze(-1).unsqueeze(-1)  # (B,1,1,1)


# ----------------------------------------------------------------------
# 2.  ResNet-50 full-image discriminator with Spectral Norm everywhere
#     About +1.9 GB over PatchD3.  Good when you want global context
# ----------------------------------------------------------------------
def sn_conv3x3(in_ch, out_ch, stride=1):
    return nn.utils.spectral_norm(
        nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False))

class SNBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = sn_conv3x3(in_ch, out_ch, stride)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = sn_conv3x3(out_ch, out_ch)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.act   = nn.LeakyReLU(0.2, True)
        self.down  = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                sn_conv3x3(in_ch, out_ch, stride),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        out = self.act(out + identity)
        return out

def make_layer(in_ch, out_ch, num_blocks, stride=1):
    layers = [SNBasicBlock(in_ch, out_ch, stride)]
    for _ in range(1, num_blocks):
        layers.append(SNBasicBlock(out_ch, out_ch))
    return nn.Sequential(*layers)

class ResNet50SN_D(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.conv1 = sn_conv3x3(in_ch, 64)
        self.bn1   = nn.BatchNorm2d(64)
        self.act   = nn.LeakyReLU(0.2, True)
        # ResNet-50 uses [3,4,6,3] bottleneck blocks; we use basic ≈34-layer depth
        self.layer1 = make_layer(64,  128, 3, stride=2)
        self.layer2 = make_layer(128, 256, 4, stride=2)
        self.layer3 = make_layer(256, 512, 6, stride=2)
        self.layer4 = make_layer(512, 512, 3, stride=2)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.utils.spectral_norm(nn.Linear(512, 1))

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x).unsqueeze(-1).unsqueeze(-1)  # (B,1,1,1)


# ----------------------------------------------------------------------
# 3.  Four-scale PatchGAN (70×70) – exactly like your MultiScaleD but 4 levels
# ----------------------------------------------------------------------
class FourScalePatchD(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.scales = nn.ModuleList([PatchD3(in_ch) for _ in range(4)])
        self.down   = nn.AvgPool2d(3, 2, 1, count_include_pad=False)

    def forward(self, x):
        outs = []
        for D in self.scales:
            outs.append(D(x))   # (B,1,h,w)
            x = self.down(x)
        return outs             # list of 4 tensors