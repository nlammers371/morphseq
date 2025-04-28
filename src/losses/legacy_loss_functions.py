import torch
import torch.nn as nn
import torch.nn.functional as F
from src.losses.loss_configs import MetricLoss
from src.models.model_utils import ModelOutput

def recon_module(self, x, recon_x):
    if self.reconstruction_loss == "mse":
        recon_loss = (
                0.5
                * F.mse_loss(
            recon_x.reshape(x.shape[0], -1),
            x.reshape(x.shape[0], -1),
            reduction="none",
        ).sum(dim=-1)
        )

    elif self.reconstruction_loss == "bce":

        recon_loss = F.binary_cross_entropy(
            recon_x.reshape(x.shape[0], -1),
            x.reshape(x.shape[0], -1),
            reduction="none",
        ).sum(dim=-1)
        
    return recon_loss


class VAELossBasic(nn.Module):

    def __init__(self, kld_weight=1.0, reconstruction_loss="mse"):
        super().__init__()

        self.kld_weight = kld_weight
        self.reconstruction_loss = reconstruction_loss

    def forward(self, model_input, model_output, batch_key="data"):

        x = model_input[batch_key]
        recon_x = model_output.recon_x
        logvar = model_output.logvar
        mu = model_output.mu

        recon_loss = recon_module(self, x, recon_x)

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

        output = ModelOutput(
            loss=recon_loss.mean(dim=0) + self.kld_weight*KLD.mean(dim=0),
            recon_loss=recon_loss.mean(dim=0),
            KLD=KLD.mean(dim=0)
        )

        return output
    
class NTXentLoss(nn.Module):

    def __init__(self, cfg: MetricLoss):
        super().__init__()

        # stash the whole config if you need it later
        self.cfg = cfg
        self.reconstruction_loss = cfg.reconstruction_loss
        self.kld_weight = cfg.kld_weight
        self.bio_only_kld = cfg.bio_only_kld

    def forward(self,  model_input, model_output, batch_key="data"):

        # reshape inputs
        x = model_input[batch_key]
        # 1) split out the two views
        x0, x1 = x.unbind(dim=1)  # each is (B, C, H, W)
        # 2) stack them into a single 2B batch, *block-wise*
        x_tall = torch.cat([x0, x1], dim=0)

        # get model output
        recon_x = model_output.recon_x
        logvar = model_output.logvar
        mu = model_output.mu

        # get metadata
        self_stats = model_input["self_stats"]
        other_stats = model_input["other_stats"]
        # hpf_deltas = model_output.hpf_deltas

        # calculate reconstruction error
        recon_loss = recon_module(self, x_tall, recon_x)
        recon_scale_factor = (128*288) / (x.shape[-1] * x.shape[-2])
        kld_scale_factor = 64 / mu.shape[1] # does not account for possibility of bio-only. Simpler. Not sure which is better
        # Calculate cross-entropy wrpt a standard multivariate Gaussian
        if self.bio_only_kld:
            b_indices = self.cfg.biological_indices
            KLD = -0.5 * torch.sum(1 + logvar[:, b_indices] - mu[:, b_indices].pow(2) -
                                   logvar[:, b_indices].exp(), dim=-1)
        else:
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

        metric_loss = self._nt_xent_loss_euclidean(features=mu,
                                                   self_stats=self_stats,
                                                   other_stats=other_stats)

        # calculate weighted loss components
        metric_loss_w = self.cfg.metric_weight * metric_loss
        recon_loss_w = recon_loss.mean(dim=0) * recon_scale_factor
        kld_loss_w = self.cfg.kld_weight * KLD.mean(dim=0) * kld_scale_factor  #* latent_weight

        output = ModelOutput(
            loss=recon_loss_w + kld_loss_w + metric_loss_w,
            recon_loss=recon_loss_w,
            KLD=kld_loss_w,
            metric_loss=metric_loss_w
        )

        return output
    

    def _nt_xent_loss_euclidean(self, features, self_stats=None, other_stats=None, n_views=2):

        temperature = self.cfg.temperature
        # remove latent dimensions that are intended to capture nuisance variability--these should not factor
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
            0.5) + 1) / temperature  # Effectively a shifted z score. Note that large distances are permitted to go below -1

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
