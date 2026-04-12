import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.losses.discriminators import PatchD3, MultiScaleD, StyleGAN2D, FourScalePatchD, ResNet50SN_D, StyleGAN2DV0
from src.core.models.model_utils import ModelOutput
import lpips
from torch.amp import autocast


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions, no state)
# ---------------------------------------------------------------------------

def calc_tv_loss(recon_x):
    tv_v = torch.abs(recon_x[:, :, 1:, :] - recon_x[:, :, :-1, :])
    tv_h = torch.abs(recon_x[:, :, :, 1:] - recon_x[:, :, :, :-1])
    return tv_v.mean(dim=[1, 2, 3]) + tv_h.mean(dim=[1, 2, 3])


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
    elif self.reconstruction_loss == "L2":
        recon_loss = (
            0.5 * F.mse_loss(
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


def _build_discriminator(cfg):
    """Instantiate the GAN discriminator selected by *cfg.gan_net*.

    Centralises the architecture-selection logic that was previously
    duplicated inside both ``VAELossBasic`` and ``NTXentLoss``.
    """
    _D_MAP = {
        "patch":       lambda: PatchD3(in_ch=cfg.input_dim[0]),
        "ms_patch":    lambda: MultiScaleD(in_ch=cfg.input_dim[0]),
        "style2":      lambda: StyleGAN2DV0(in_ch=cfg.input_dim[0]),       # default StyleGAN-2 variant
        "style2_small":lambda: StyleGAN2D(in_ch=cfg.input_dim[0], num_blocks=4),
        "style2_big":  lambda: StyleGAN2D(in_ch=cfg.input_dim[0], num_blocks=7),
        "resnet_sn":   lambda: ResNet50SN_D(in_ch=cfg.input_dim[0]),
        "patch4scale": lambda: FourScalePatchD(in_ch=cfg.input_dim[0]),
    }
    if cfg.gan_net not in _D_MAP:
        raise ValueError(
            f"Unknown gan_net {cfg.gan_net!r}; must be one of {list(_D_MAP)}"
        )
    return _D_MAP[cfg.gan_net]()


# ---------------------------------------------------------------------------
# Evaluation-only LPIPS (frozen, never contributes to gradients)
# ---------------------------------------------------------------------------

class EVALPIPSLOSS(nn.Module):
    def __init__(self, cfg, force_gpu: bool = False):
        super().__init__()
        self.pips_net = cfg.eval_pips_net
        self.metric = lpips.LPIPS(net=self.pips_net)
        if force_gpu and torch.cuda.is_available():
            self.metric.cuda()
            self._on_gpu = True
        else:
            self._on_gpu = False

    def _ensure_device(self):
        if not self._on_gpu and next(self.metric.parameters()).is_cuda:
            self.metric.cpu()

    def forward(self, model_input, model_output, batch_key="data"):
        """
        recon  : (N,C,H,W)  – model output
        target : (N,C,H,W)  – ground-truth image
        Returns: scalar LPIPS distance (mean over batch).
        """
        self._ensure_device()
        target = model_input[batch_key]
        recon  = model_output.recon_x
        dev = next(self.metric.parameters()).device
        target = target.to(dev, non_blocking=True)
        recon  = recon.to(dev, non_blocking=True)
        return self.metric(recon, target).mean()


# ---------------------------------------------------------------------------
# Shared base class  (LPIPS setup + GAN setup + shared forward logic)
# ---------------------------------------------------------------------------

class _VAELossBase(nn.Module):
    """Internal base class shared by VAELossBasic and NTXentLoss.

    Holds all common state (LPIPS, discriminator, weight schedules) and
    exposes ``_compute_vae_terms`` which returns the four core loss
    components that both subclasses need.

    Subclasses only need to implement ``forward``.
    """

    def __init__(self, cfg, recon_logvar_init: float = 0.0):
        super().__init__()

        # --- Pixel reconstruction ---
        self.reconstruction_loss = cfg.reconstruction_loss

        # --- Perceptual (LPIPS) ---
        self.pips_flag     = cfg.pips_flag
        self.schedule_pips = cfg.schedule_pips
        self.pips_weight   = cfg.pips_weight
        self.pips_net      = cfg.pips_net
        self.pips_cfg      = cfg.pips_cfg
        self.tv_weight     = cfg.tv_weight

        # --- GAN ---
        self.use_gan      = cfg.use_gan
        self.gan_weight   = cfg.gan_weight
        self.gan_net      = cfg.gan_net
        self.schedule_gan = cfg.schedule_gan
        self.gan_cfg      = cfg.gan_cfg

        # --- KLD ---
        self.schedule_kld = cfg.schedule_kld
        self.kld_weight   = cfg.kld_weight
        self.kld_cfg      = cfg.kld_cfg

        # --- LPIPS (frozen perceptual network) ---
        self.perceptual_loss = lpips.LPIPS(net=self.pips_net)
        for p in self.perceptual_loss.parameters():
            p.requires_grad = False
        self.perceptual_loss.eval()

        # Learnable recon log-variance (currently kept as a non-gradient buffer)
        self.register_buffer("recon_logvar", torch.tensor(recon_logvar_init))

        # --- Discriminator (optional) ---
        if self.use_gan:
            self.D = _build_discriminator(cfg)

    # -----------------------------------------------------------------------
    # Pixel-scale normalisation
    # -----------------------------------------------------------------------
    # The raw pixel loss is a per-pixel mean, which for a 128×288 image is
    # tiny (~0.001).  We rescale to the same ballpark as the KLD term
    # (which operates on latent vectors of dimension ~64) so that the
    # loss weights (kld_weight, pips_weight, gan_weight) are interpretable
    # as straightforward multipliers rather than needing to compensate for
    # a 36 000× scale difference.
    #
    # Derivation:
    #   pixel_scale = (H × W) / 100
    # The factor of 100 is an empirically chosen constant that sets the
    # typical reconstruction loss to O(1) for an untrained model on
    # normalised [0, 1] images, matching the KLD scale.  For L1 we apply
    # an additional /10 because L1 ≈ 10× L2 for Gaussian noise.
    def _pixel_scale(self) -> float:
        if self.reconstruction_loss != "L1":
            return (128 * 288) / 100
        else:
            return (128 * 288) / 10 / 100

    # -----------------------------------------------------------------------
    # Shared computation of the four core VAE loss terms
    # -----------------------------------------------------------------------
    def _compute_vae_terms(self, x, recon_x, mu, logvar):
        """Compute the four core VAE loss terms.

        Parameters
        ----------
        x       : (B, C, H, W) ground-truth image
        recon_x : (B, C, H, W) reconstruction
        mu      : (B, D) posterior mean
        logvar  : (B, D) posterior log-variance

        Returns
        -------
        pixel_loss_w : scalar  – pixel reconstruction loss (scaled)
        kld_loss_w   : scalar  – KL divergence (scale factor = 1)
        pips_loss    : scalar  – perceptual LPIPS loss (or 0)
        gan_loss     : scalar  – generator adversarial loss (or 0)
        """
        # Standard Gaussian KL: -0.5 * E[1 + logvar - mu² - exp(logvar)]
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

        pixel_loss = recon_module(self, x, recon_x)

        pips_loss = 0
        if self.pips_flag and self.pips_weight > 0:
            pips_loss = calc_pips_loss(self, x, recon_x)

        gan_loss = 0
        if self.use_gan and self.gan_weight > 0:
            pred_fake = self.D(recon_x)
            if isinstance(pred_fake, (list, tuple)):   # multi-scale discriminator
                gan_loss = sum([-p.mean() for p in pred_fake]) / len(pred_fake)
            else:
                gan_loss = -pred_fake.mean()

        pixel_loss_w = pixel_loss.mean() * self._pixel_scale()
        kld_loss_w   = KLD.mean()        # kld_scale_factor = 1 (weights are in kld_weight)

        return pixel_loss_w, kld_loss_w, pips_loss, gan_loss


# ---------------------------------------------------------------------------
# VAELossBasic — plain VAE loss
# ---------------------------------------------------------------------------

class VAELossBasic(_VAELossBase):
    """Standard VAE loss: pixel + KLD + optional LPIPS + optional GAN."""

    def forward(self, model_input, model_output, batch_key="data"):
        x       = model_input[batch_key]
        recon_x = model_output.recon_x
        mu      = model_output.mu
        logvar  = model_output.logvar

        pixel_loss_w, kld_loss_w, pips_loss, gan_loss = self._compute_vae_terms(
            x, recon_x, mu, logvar
        )

        recon_loss = (
            pixel_loss_w
            + self.kld_weight  * kld_loss_w
            + self.pips_weight * pips_loss
            + self.gan_weight  * gan_loss
        )

        return ModelOutput(
            loss=recon_loss,
            recon_loss=recon_loss,
            gan_loss=gan_loss,
            pips_loss=pips_loss,
            pixel_loss=pixel_loss_w,
            kld_loss=kld_loss_w,
        )


# ---------------------------------------------------------------------------
# NTXentLoss — VAE loss + NT-Xent contrastive metric learning
# ---------------------------------------------------------------------------

class NTXentLoss(_VAELossBase):
    """VAE loss extended with NT-Xent (Euclidean) contrastive metric learning.

    The contrastive term acts only on the *biological* latent dimensions
    (``biological_indices``); the nuisance dimensions are regularised by
    the standard KLD alone.
    """

    def __init__(self, cfg, recon_logvar_init: float = 0.0):
        super().__init__(cfg, recon_logvar_init)

        # NT-Xent specific
        self.schedule_metric    = cfg.schedule_metric
        self.metric_weight      = cfg.metric_weight
        self.metric_cfg         = cfg.metric_cfg
        self.biological_indices = cfg.biological_indices
        self.nuisance_indices   = cfg.nuisance_indices
        self.cfg                = cfg   # keep whole config for convenience

    def forward(self, model_input, model_output, batch_key="data"):
        x = model_input[batch_key]
        x0, _ = x.unbind(dim=1)   # split paired views; reconstruct from view 0 only

        pixel_loss_w, kld_loss_w, pips_loss, gan_loss = self._compute_vae_terms(
            x0, model_output.recon_x, model_output.mu, model_output.logvar
        )

        metric_loss = self._nt_xent_loss_euclidean(
            features=model_output.mu,
            self_stats=model_input["self_stats"],
            other_stats=model_input["other_stats"],
        )

        total_loss = (
            pixel_loss_w
            + self.kld_weight   * kld_loss_w
            + self.pips_weight  * pips_loss
            + self.gan_weight   * gan_loss
            + self.metric_weight * metric_loss
        )

        return ModelOutput(
            loss=total_loss,
            recon_loss=total_loss,
            metric_loss=metric_loss,
            gan_loss=gan_loss,
            pips_loss=pips_loss,
            pixel_loss=pixel_loss_w,
            kld_loss=kld_loss_w,
        )

    # -----------------------------------------------------------------------
    # NT-Xent implementation
    # -----------------------------------------------------------------------

    def _nt_xent_loss_euclidean(self, features, self_stats=None, other_stats=None, n_views=2):

        temperature = self.cfg.temperature
        features    = features[:, self.cfg.biological_indices]
        device      = features.device
        batch_size  = int(features.shape[0] / n_views)

        # Positive-pair indicator matrix (1 = positive, -1 = exclude self)
        pair_matrix = torch.cat([torch.arange(batch_size) for _ in range(n_views)], dim=0)
        pair_matrix = (pair_matrix.unsqueeze(0) == pair_matrix.unsqueeze(1)).float()
        mask = torch.eye(pair_matrix.shape[0], dtype=torch.bool, device=device)
        pair_matrix[mask] = -1

        # Normalised squared Euclidean distance
        dist_matrix = torch.cdist(features, features, p=2).pow(2)
        N     = self.cfg.latent_dim_bio / 2
        sigma = N
        dist_normed = (-(dist_matrix / sigma).pow(0.5) + self.cfg.margin) / temperature

        # Build target matrix: 1 = positive, 0 = negative, -1 = exclude
        target_matrix = torch.zeros(pair_matrix.shape, dtype=torch.float32)
        if self_stats is not None:
            age_vec    = torch.cat([self_stats[1], other_stats[1]], axis=0)
            age_deltas = torch.abs(age_vec.unsqueeze(-1) - age_vec.unsqueeze(0))
            age_bool   = age_deltas <= (self.cfg.time_window + 1.5)

            pert_cross = torch.zeros_like(age_bool, dtype=torch.bool)
            pert_bool  = torch.ones_like(age_bool,  dtype=torch.bool)

            if self.cfg.self_target_prob < 1.0:
                pert_vec      = torch.cat([self_stats[2], other_stats[2]], axis=0)
                metric_array  = torch.tensor(self.cfg.metric_array).type(torch.int8)
                metric_matrix = metric_array.clone().to(pert_vec.device)
                metric_matrix = metric_matrix[pert_vec, :][:, pert_vec]
                pert_bool  = metric_matrix == 1
                pert_cross = metric_matrix == -1

            target_matrix[age_bool & pert_bool] = 1
            target_matrix[pert_cross]            = -1

        target_matrix[pair_matrix == 1]  = 1
        target_matrix[pair_matrix == -1] = -1
        target_matrix = target_matrix.to(device)

        return self._nt_xent_loss_multiclass(dist_normed, target_matrix)

    def _nt_xent_loss_multiclass(self, logits_tempered, target):
        logits_tempered[target == -1] = -torch.inf   # exclude flagged pairs

        logits_num = logits_tempered.clone()
        logits_num[target == 0] = -torch.inf          # exclude negatives from numerator

        numerator   = torch.logsumexp(logits_num,      axis=1)
        denominator = torch.logsumexp(logits_tempered, axis=1)
        return torch.mean(-(numerator - denominator))
