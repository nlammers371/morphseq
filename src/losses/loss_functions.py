import torch
import torch.nn as nn
import torch.nn.functional as F
from src.losses.loss_configs import MetricLoss
from src.losses.discriminators import PatchD3, MultiScaleD, StyleGAN2D, FourScalePatchD, ResNet50SN_D, StyleGAN2DV0
from src.models.model_utils import ModelOutput
# from taming.modules.losses.vqperceptual import LPIPS
import lpips
from torch.amp import autocast

def calc_tv_loss(recon_x):
    # vertical diffs: shape [B, C, H-1, W]
    tv_v = torch.abs(recon_x[:, :, 1:, :] - recon_x[:, :, :-1, :])
    # horizontal diffs: shape [B, C, H, W-1]
    tv_h = torch.abs(recon_x[:, :, :, 1:] - recon_x[:, :, :, :-1])
    # mean over channel & spatial dims → [B]
    tv_loss = (
            tv_v.mean(dim=[1, 2, 3]) +
            tv_h.mean(dim=[1, 2, 3])
    )
    return tv_loss

def calc_pips_loss(self, x, recon_x) -> ModelOutput:

    if x.shape[1] == 1:
        in3 = x.repeat(1, 3, 1, 1)
        out3 = recon_x.repeat(1, 3, 1, 1)
        with autocast("cuda"):
            p_loss = self.perceptual_loss(in3, out3).view(x.shape[0])
    else:
        with autocast("cuda"):
            p_loss = self.perceptual_loss(
                x.contiguous(), recon_x.contiguous()
            ).view(x.shape[0])

    return p_loss.mean()



def recon_module(self, x, recon_x):
    if self.reconstruction_loss == "L1":
        recon_loss = torch.abs(
            recon_x.reshape(x.shape[0], -1) - x.reshape(x.shape[0], -1),
        ).mean(dim=-1)

    elif self.reconstruction_loss == "mse":
        recon_loss = (
                0.5
                * F.mse_loss(
            recon_x.reshape(x.shape[0], -1),
            x.reshape(x.shape[0], -1),
            reduction="none",
        ).mean(dim=-1)
        )

    elif self.reconstruction_loss == "bce":
        recon_loss = F.binary_cross_entropy(
            recon_x.reshape(x.shape[0], -1),
            x.reshape(x.shape[0], -1),
            reduction="none",
        ).mean(dim=-1)
    else:
        raise NotImplementedError
        
    return recon_loss

# def process_recon_loss(self, x, recon_x):
#
#     if self.pips_flag:
#         PIPS_loss = L1PIPS_module(self, x, recon_x)
#         recon_loss = PIPS_loss.nll_loss
#         px_loss = PIPS_loss.recon_loss
#         p_loss = PIPS_loss.pips_loss
#     elif (not self.pips_flag) and (self.reconstruction_loss == "L1"):  # we can also do just the L1 loss
#         self.pips_weight = 0
#         PIPS_loss = L1PIPS_module(self, x, recon_x)
#         recon_loss = PIPS_loss.nll_loss
#         px_loss = PIPS_loss.recon_los
#         p_loss = PIPS_loss.p_loss
#     elif self.reconstruction_loss in ["mse", "bce"]:
#         recon_loss = recon_module(self, x, recon_x)
#         px_loss = recon_loss
#         p_loss = torch.tensor([0.0])
#     else:
#         raise NotImplementedError
#
#     return recon_loss, px_loss, p_loss

class EVALPIPSLOSS(nn.Module):
    def __init__(self, cfg, force_gpu: bool = False):
        super().__init__()
        self.pips_net = cfg.eval_pips_net
        self.metric = lpips.LPIPS(net=self.pips_net)
        # instantiated on CPU
        if force_gpu and torch.cuda.is_available():
            self.metric.cuda()
            self._on_gpu = True
        else:
            self._on_gpu = False

        # Lightning may move sub-modules to GPU; guard against that:
    def _ensure_device(self):
        """
        Make sure the metric is back on the intended device (CPU by default)
        after Lightning's automatic `.to(device)` calls.
        """
        if not self._on_gpu and next(self.metric.parameters()).is_cuda:
            self.metric.cpu()

    def forward(self, model_input, model_output, batch_key="data"):
        """
        recon  : (N,C,H,W) in [-1,1]  – model output
        target : (N,C,H,W) in [-1,1]  – ground-truth image
        Returns: scalar LPIPS distance (mean over batch).
        """
        self._ensure_device()  # <-- safety check

        target = model_input[batch_key]
        recon = model_output.recon_x

        dev = next(self.metric.parameters()).device  # cpu *or* cuda
        target = target.to(dev, non_blocking=True)
        recon = recon.to(dev, non_blocking=True)

        return self.metric(recon, target).mean()



class VAELossBasic(nn.Module):

    def __init__(self, cfg, recon_logvar_init=0.0):
        super().__init__()

        # self.kld_weight = cfg.kld_weight
        self.reconstruction_loss = cfg.reconstruction_loss

        # only applies if we're not doing PIPS
        self.pips_flag = cfg.pips_flag
        self.schedule_pips = cfg.schedule_pips
        self.pips_weight = cfg.pips_weight
        self.pips_net = cfg.pips_net
        self.pips_cfg = cfg.pips_cfg
        self.tv_weight = cfg.tv_weight

        # GAN
        self.use_gan = cfg.use_gan
        self.gan_weight = cfg.gan_weight
        self.gan_net = cfg.gan_net
        self.schedule_gan = cfg.schedule_gan
        self.gan_cfg = cfg.gan_cfg

        # KLD
        self.schedule_kld = cfg.schedule_kld
        self.kld_weight = cfg.kld_weight
        self.kld_cfg = cfg.kld_cfg

        # ---- set up LPIPS (AlexNet backbone) ----
        self.perceptual_loss = lpips.LPIPS(net=self.pips_net)  # default is vgg, so net="alex"
        # freeze *all* its parameters:
        for p in self.perceptual_loss.parameters():
            p.requires_grad = False
        # put it in eval mode so BatchNorm / Dropout won’t update
        self.perceptual_loss.eval()
        # this is your learnable recon‐variance term:
        self.register_buffer('recon_logvar', torch.tensor(recon_logvar_init)) # NOPE. nn.Parameter(torch.tensor(recon_logvar_init))

        # --- set up GAN loss ----
        if self.use_gan:
            if cfg.gan_net == "patch":
                self.D = PatchD3(in_ch=cfg.input_dim[0])
            elif cfg.gan_net == "ms_patch":
                self.D = MultiScaleD(in_ch=cfg.input_dim[0])
            elif cfg.gan_net == "style2": # keeping temporarilly
                self.D = StyleGAN2DV0(in_ch=cfg.input_dim[0])
            elif cfg.gan_net == "style2_small":
                self.D = StyleGAN2D(in_ch=cfg.input_dim[0], num_blocks=4)
            elif cfg.gan_net == "style2_big":
                self.D = StyleGAN2D(in_ch=cfg.input_dim[0], num_blocks=8)
            elif cfg.gan_net == "resnet_sn":
                self.D = ResNet50SN_D(in_ch=cfg.input_dim[0])
            elif cfg.gan_net == "patch4scale":
                self.D = FourScalePatchD(in_ch=cfg.input_dim[0])
            else:
                raise ValueError("unknown gan_arch")

    def forward(self, model_input, model_output, batch_key="data"):

        # get model outputs
        x = model_input[batch_key]
        recon_x = model_output.recon_x
        logvar = model_output.logvar
        mu = model_output.mu

        # Standard Gaussian regularization
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

        # get PIXEL recon loss
        pixel_loss = recon_module(self, x, recon_x)

        # get Perceptual loss
        if self.pips_flag & (self.pips_weight > 0):
            pips_loss = calc_pips_loss(self, x, recon_x)
        else:
            pips_loss = 0

        # add adversarial loss
        gan_loss = 0  # default zero
        if self.use_gan and self.gan_weight > 0:
            pred_fake = self.D(recon_x)
            if isinstance(pred_fake, (list, tuple)):  # multi-scale
                gan_loss = sum([-p.mean() for p in pred_fake]) / len(pred_fake)
            else:
                gan_loss = -pred_fake.mean()

        # TV loss
        # tv_loss = 0
        # if self.tv_weight > 0:
        #     tv_loss = calc_tv_loss(recon_x)

        # Upscale recon and KLD to standardized pixel/latent sizes
        if self.reconstruction_loss != "L1":
            pixel_scale_factor = (128 * 288) / 100 # factor of 100 comes from shuffling the KLD resizing factor
        else:
            pixel_scale_factor = (128 * 288) / 10 / 100# accounts for fact that L1 loss ~10x size of L2

        kld_scale_factor = 1 #/ mu.shape[1]
        # calculate weighted loss components
        pixel_loss_w = pixel_loss.mean(dim=0) * pixel_scale_factor
        kld_loss_w = KLD.mean(dim=0) * kld_scale_factor

        # combine
        recon_loss = pixel_loss_w + self.kld_weight*kld_loss_w + self.pips_weight*pips_loss + self.gan_weight*gan_loss

        output = ModelOutput(
            loss=recon_loss, recon_loss=recon_loss,
            gan_loss=gan_loss, pips_loss=pips_loss, pixel_loss=pixel_loss_w, kld_loss=kld_loss_w
        )

        return output

    
class NTXentLoss(nn.Module):

    def __init__(self, cfg: MetricLoss, recon_logvar_init=0.0):
        super().__init__()

        # stash the whole config if you need it later
        self.cfg = cfg
        self.reconstruction_loss = cfg.reconstruction_loss  # only applies if we're not doing PIPS
        self.kld_weight = cfg.kld_weight
        self.bio_only_kld = cfg.bio_only_kld

        # PIPS stuff
        self.schedule_pips = cfg.schedule_pips
        self.pips_flag = cfg.pips_flag
        self.pips_weight = cfg.pips_weight
        self.pips_net = cfg.pips_net
        self.pips_cfg = cfg.pips_cfg

        self.tv_weight = cfg.tv_weight
        # KLD
        self.schedule_kld = cfg.schedule_kld
        self.kld_weight = cfg.kld_weight
        self.kld_cfg = cfg.kld_cfg

        # NT-Xent
        self.schedule_metric = cfg.schedule_metric
        self.metric_weight = cfg.metric_weight
        self.metric_cfg = cfg.metric_cfg

        # ---- set up LPIPS (AlexNet backbone) ----
        self.perceptual_loss = lpips.LPIPS(net=self.pips_net)  # default is vgg, so net="alex"
        # freeze *all* its parameters:
        for p in self.perceptual_loss.parameters():
            p.requires_grad = False
        # put it in eval mode so BatchNorm / Dropout won’t update
        self.perceptual_loss.eval()

        self.register_buffer('recon_logvar', torch.tensor(recon_logvar_init))


    def forward(self, model_input, model_output, batch_key="data"):

        # reshape x
        x = model_input[batch_key]
        # 1) split out the two views
        x0, x1 = x.unbind(dim=1)  # each is (B, C, H, W)
        # 2) stack them into a single 2B batch, *block-wise*
        # x_tall = torch.cat([x0, x1], dim=0)

        # get model output
        recon_x = model_output.recon_x
        logvar = model_output.logvar
        mu = model_output.mu

        # get metadata
        self_stats = model_input["self_stats"]
        other_stats = model_input["other_stats"]
        # hpf_deltas = model_output.hpf_deltas

        # calculate reconstruction error
        recon_loss, px_loss, p_loss = process_recon_loss(self, x0, recon_x)
        if self.pips_flag:
            recon_scale_factor = (128*288)
        else:
            recon_scale_factor = (128 * 288) / 10

        #/ (x.shape[-1] * x.shape[-2])
        kld_scale_factor = 100 #/ mu.shape[1] # does not account for possibility of bio-only. Simpler. Not sure which is better
        # Calculate cross-entropy wrpt a standard multivariate Gaussian
        if self.bio_only_kld:
            b_indices = self.cfg.biological_indices
            KLD = -0.5 * torch.mean(1 + logvar[:, b_indices] - mu[:, b_indices].pow(2) -
                                   logvar[:, b_indices].exp(), dim=-1)
        else:
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

        metric_loss = self._nt_xent_loss_euclidean(features=mu,
                                                   self_stats=self_stats,
                                                   other_stats=other_stats)

        # calculate weighted loss components
        metric_loss_w = self.cfg.metric_weight * metric_loss
        recon_loss_w = recon_loss.mean(dim=0) * recon_scale_factor
        kld_loss_w = self.cfg.kld_weight * KLD.mean(dim=0) * kld_scale_factor  #* latent_weight

        output = ModelOutput(
            loss=(recon_loss_w + kld_loss_w + metric_loss_w) / (recon_scale_factor + kld_scale_factor),
            recon_loss=recon_loss.mean(dim=0),
            KLD=KLD.mean(dim=0),
            metric_loss=metric_loss,
            pixel_loss=px_loss.mean(dim=0),
            pips_loss=p_loss.mean(dim=0),
        )

        return output
    

    def _nt_xent_loss_euclidean(self, features, self_stats=None, other_stats=None, n_views=2):

        temperature = self.cfg.temperature
        # into the contrastive loss
        features = features[:, self.cfg.biological_indices]
        device = features.device
        # infer batch size
        batch_size = int(features.shape[0] / n_views)

        # EUCLIDEAN
        pair_matrix = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        pair_matrix = (pair_matrix.unsqueeze(0) == pair_matrix.unsqueeze(1)).float()
        mask = torch.eye(pair_matrix.shape[0], dtype=torch.bool).to(device)
        pair_matrix[mask] = -1  # exclude self comparisons
        # target_matrix = target_matrix.to(self.device)

        dist_matrix = torch.cdist(features, features, p=2).pow(2)

        # normalize distances to fall on scale similar to cosine. For stability, we will use expectation for 2 sigma of an isotropic gaussian as our "max"
        N = self.cfg.latent_dim_bio / 2
        sigma = N  # sigma^1 for an ND isotropic Gaussian
        dist_normed = (-(dist_matrix / sigma).pow(
            0.5) + self.cfg.margin) / temperature  # Effectively a shifted z score. Note that large distances are permitted to go below -1

        # Generate matrix containing pos/neg pair info
        target_matrix = torch.zeros(pair_matrix.shape, dtype=torch.float32)
        if self_stats is not None:
            age_vec = torch.cat([self_stats[1], other_stats[1]], axis=0)

            age_deltas = torch.abs(age_vec.unsqueeze(-1) - age_vec.unsqueeze(0))
            age_bool = age_deltas <= (self.cfg.time_window + 1.5)  # add an extra neutral "buffer" of 1.5 hrs

            pert_cross = torch.zeros_like(age_bool, dtype=torch.bool)
            pert_bool = torch.ones_like(age_bool, dtype=torch.bool)
            # if self.cfg.time_only_flag == 1:
            #     pass
            if self.cfg.self_target_prob < 1.0:
                pert_vec = torch.cat([self_stats[2], other_stats[2]], axis=0)

                # get class relationships
                metric_array = torch.tensor(self.cfg.metric_array).type(torch.int8)
                metric_matrix = metric_array.clone().to(pert_vec.device)
                metric_matrix = metric_matrix[pert_vec, :]
                metric_matrix = metric_matrix[:, pert_vec]

                pert_bool = metric_matrix == 1  # positive examples #pert_vec.unsqueeze(-1) == pert_vec.unsqueeze(0)  # avoid like perturbations
                pert_cross = metric_matrix == -1
            else:
                pass

            extra_match_flags = (age_bool & pert_bool).type(torch.bool)  # extra positives
            target_matrix[extra_match_flags] = 1
            target_matrix[pert_cross] = -1

        target_matrix[pair_matrix == 1] = 1
        target_matrix[pair_matrix == -1] = -1

        # pass to device
        target_matrix = target_matrix.to(device)

        # call multiclass nt_xent loss
        loss_euc = self._nt_xent_loss_multiclass(dist_normed, target_matrix)

        return loss_euc
    
    
    def _nt_xent_loss_multiclass(self, logits_tempered, target):

        # Exclude cross-matches from everything
        logits_tempered[target == -1] = -torch.inf # exclude flagged instances (self pair and cross-matched pairs)

        # exclude negative pairs from numerator
        logits_num = logits_tempered.clone()
        logits_num[target == 0] = -torch.inf # exclude all negative pairs from numerator

        # calculate loss for each entry in the batch
        numerator = torch.logsumexp(logits_num, axis=1)
        denominator = torch.logsumexp(logits_tempered, axis=1)
        loss = -(numerator - denominator)

        return torch.mean(loss)
