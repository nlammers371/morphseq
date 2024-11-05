import os
from typing import Optional
import ntpath
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import TripletMarginLoss
import pandas as pd
from pythae.data.datasets import BaseDataset
from ..base import BaseAE
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from ..nn.default_architectures import Encoder_VAE_MLP
from .seq_vae_config import SeqVAEConfig


class SeqVAE(BaseAE):
    """Variational Autoencoder model with sequential and metric constraints

    Args:
        model_config (VAEConfig): The Variational Autoencoder configuration setting the main
        parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
            self,
            model_config: SeqVAEConfig,
            encoder: Optional[BaseEncoder] = None,
            decoder: Optional[BaseDecoder] = None,
    ):

        BaseAE.__init__(self, model_config=model_config, decoder=decoder)

        self.model_name = "SeqVAE"
        self.metric_loss_type = model_config.metric_loss_type
        self.latent_dim = model_config.latent_dim
        self.zn_frac = model_config.zn_frac  # number of nuisance latent dimensions
        self.temperature = model_config.temperature
        self.margin = model_config.margin
        self.distance_metric = model_config.distance_metric
        self.orth_flag = model_config.orth_flag  # indicates whether or not to impose orthogonality constraint
        self.beta = model_config.beta
        self.metric_weight = model_config.metric_weight
        # calculate number of "biological" and "nuisance" latent variables
        self.latent_dim_nuisance = torch.tensor(np.floor(self.latent_dim * self.zn_frac))
        self.latent_dim_biological = self.latent_dim - self.latent_dim_nuisance
        self.nuisance_indices = torch.arange(self.latent_dim_nuisance, dtype=torch.int)
        self.biological_indices = torch.arange(self.latent_dim_nuisance, self.latent_dim, dtype=torch.int)
        self.model_config = model_config
        self.contrastive_flag = True

        #### NEW STUFF
        self.mu_cache = {}
        self.logvar_cache = {}
        self.iter_i = 0


        if encoder is None:
            if model_config.input_dim is None:
                raise AttributeError(
                    "No input dimension provided !"
                    "'input_dim' parameter of BaseAEConfig instance must be set to 'data_shape' "
                    "where the shape of the data is (C, H, W ..). Unable to build encoder "
                    "automatically"
                )

            encoder = Encoder_VAE_MLP(model_config)
            self.model_config.uses_default_encoder = True

        else:
            self.model_config.uses_default_encoder = False

        self.set_encoder(encoder)

    def forward(self, inputs: BaseDataset, **kwargs):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """

        x = inputs["data"]
        # Check input to see if it is 5 dmensional, if so, then the model is being
        self.contrastive_flag = True
        self.triplet_flag = False
        if (len(x.shape) != 5): # This pertains to post-training runs
            # raise Warning("Model did not receive contrastive pairs. No contrastive loss will be calculated.")
            self.contrastive_flag = False
        elif (x.shape[1] == 3):
            self.contrastive_flag = False
            self.triplet_flag = True

        if self.contrastive_flag:
            
            # raise Exception("Contrastive loss is not currently supported")
        
            x0 = torch.reshape(x[:, 0, :, :, :],
                               (x.shape[0], x.shape[2], x.shape[3], x.shape[4]))  # first set of images
            x1 = torch.reshape(x[:, 1, :, :, :], (
                x.shape[0], x.shape[2], x.shape[3], x.shape[4]))  # second set with matched contrastive pairs

            encoder_output0 = self.encoder(x0)
            encoder_output1 = self.encoder(x1)

            mu0, log_var0 = encoder_output0.embedding, encoder_output0.log_covariance
            mu1, log_var1 = encoder_output1.embedding, encoder_output1.log_covariance

            std0 = torch.exp(0.5 * log_var0)
            std1 = torch.exp(0.5 * log_var1)

            z0, eps0 = self._sample_gauss(mu0, std0)
            recon_x0 = self.decoder(z0)["reconstruction"]

            z1, eps1 = self._sample_gauss(mu1, std1)
            recon_x1 = self.decoder(z1)["reconstruction"]

            # combine
            x_out = torch.cat([x0, x1], axis=0)
            recon_x_out = torch.cat([recon_x0, recon_x1], axis=0)
            mu_out = torch.cat([mu0, mu1], axis=0)
            log_var_out = torch.cat([log_var0, log_var1], axis=0)
            z_out = torch.cat([z0, z1], axis=0)

            loss, recon_loss, kld, nt_xent = self.loss_function(recon_x_out, x_out, mu_out, log_var_out,
                                                                inputs["weight_hpf"], inputs["self_stats"],
                                                                inputs["other_stats"])  # , labels=y)

        elif self.triplet_flag:
            self.iter_i += 1

            xa = torch.reshape(x[:, 0, :, :, :],
                               (x.shape[0], x.shape[2], x.shape[3], x.shape[4]))  # first set of images
            xp = torch.reshape(x[:, 1, :, :, :], (
                                x.shape[0], x.shape[2], x.shape[3], x.shape[4]))  # second set with matched contrastive pairs
            xn = torch.reshape(x[:, 2, :, :, :], (
                                x.shape[0], x.shape[2], x.shape[3], x.shape[4]))

            #### NEW STUFF
            # ida = inputs["index"][0].numpy().astype(int)
            # idn = inputs["index"][1].numpy().astype(int)
            # idp = inputs["index"][2].numpy().astype(int)

            # do_flags_a = ~np.isin(ida, list(self.mu_cache.keys()))
            # do_flags_p = ~np.isin(idp, list(self.mu_cache.keys())) | np.isin(idp, ida) # overwrite in case of self-comparisons...we don't want to compare an image to its twin
            # do_flags_n = ~np.isin(idn, list(self.mu_cache.keys())) & ~np.isin(idn, ida) & ~np.isin(idn, idp)

            # # if self.iter_i == 5:
            # #     print("check")
            # ########
            # # do anchor first
            # if np.any(do_flags_a):
            #     encoder_output_a = self.encoder(xa[do_flags_a])
            #     # update cache
            #     self.mu_cache.update(zip(ida[do_flags_a], encoder_output_a.embedding.detach()))
            #     self.logvar_cache.update(zip(ida[do_flags_a], encoder_output_a.log_covariance.detach()))
            # # get encodings
            # mua = torch.stack([self.mu_cache[a] for a in ida]).clone().detach().requires_grad_(True)
            # log_vara = torch.stack([self.logvar_cache[a] for a in ida]).clone().detach().requires_grad_(True)

            # ########
            # # do positive
            # if np.any(do_flags_p):
            #     encoder_output_p = self.encoder(xp[do_flags_p])
            #     # update cache
            #     self.mu_cache.update(zip(idp[do_flags_p], encoder_output_p.embedding.detach()))
            #     self.logvar_cache.update(zip(idp[do_flags_p], encoder_output_p.log_covariance.detach()))
            # # get encodings
            # mup = torch.stack([self.mu_cache[p] for p in idp]).clone().detach().requires_grad_(True)
            # log_varp = torch.stack([self.logvar_cache[p] for p in idp]).clone().detach().requires_grad_(True)

            # ########
            # # do negative
            # if np.any(do_flags_n):
            #     encoder_output_n = self.encoder(xn[do_flags_n])
            #     # update cache
            #     self.mu_cache.update(zip(idn[do_flags_n], encoder_output_n.embedding.detach()))
            #     self.logvar_cache.update(zip(idn[do_flags_n], encoder_output_n.log_covariance.detach()))
            # # get encodings
            # mun = torch.stack([self.mu_cache[n] for n in idn]).clone().detach().requires_grad_(True)
            # log_varn = torch.stack([self.logvar_cache[n] for n in idn]).clone().detach().requires_grad_(True)
            # print(np.sum(do_flags_n))
            #### END

            encoder_output_a = self.encoder(xa)
            encoder_output_p = self.encoder(xp)
            encoder_output_n = self.encoder(xn)

            mua, log_vara = encoder_output_a.embedding, encoder_output_a.log_covariance
            mup, log_varp = encoder_output_p.embedding, encoder_output_p.log_covariance
            mun, log_varn = encoder_output_n.embedding, encoder_output_n.log_covariance


            stda = torch.exp(0.5 * log_vara)
            stdp = torch.exp(0.5 * log_varp)
            stdn = torch.exp(0.5 * log_varn)

            za, epsa = self._sample_gauss(mua, stda)
            recon_xa = self.decoder(za)["reconstruction"]

            zp, epsp = self._sample_gauss(mup, stdp)
            recon_xp = self.decoder(zp)["reconstruction"]

            zn, epsn = self._sample_gauss(mun, stdn)
            recon_xn = self.decoder(zn)["reconstruction"]

            # combine
            x_out = torch.cat([xa, xp, xn], axis=0)
            recon_x_out = torch.cat([recon_xa, recon_xp, recon_xn], axis=0)
            # mu_out = torch.cat([mua, mup, mun], axis=0)
            log_var_out = torch.cat([log_vara, log_varp, log_varn], axis=0)
            z_out = torch.cat([za, zp, zn], axis=0)

            loss, recon_loss, kld, nt_xent = self.loss_function(recon_x_out, x_out, z_out, log_var_out,
                                                                None, None, None)  # , labels=y)

        else:
            encoder_output = self.encoder(x)

            mu, log_var = encoder_output.embedding, encoder_output.log_covariance

            std = torch.exp(0.5 * log_var)

            z_out, eps = self._sample_gauss(mu, std)
            recon_x_out = self.decoder(z_out)["reconstruction"]

            loss, recon_loss, kld, nt_xent = self.loss_function(recon_x_out, x, mu, log_var, torch.ones(x.shape[0]),
                                                                None, None)  # labels=y)  # , z_out,
            # weight_matrix)

        output = ModelOutput(
            recon_loss=recon_loss,
            reg_loss=kld,
            ntxent_loss=nt_xent,
            loss=loss,
            recon_x=recon_x_out,
            z=z_out,
        )

        return output

    def loss_function(self, recon_x, x, mu, log_var, hpf_deltas, self_stats, other_stats):  # , labels=None):

        # if labels is not None:
        #     labels = self.clean_path_names(labels)

        # calculate reconstruction error
        if self.model_config.reconstruction_loss == "mse":
            recon_loss = (
                    0.5
                    * F.mse_loss(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).mean(dim=-1)
            )

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = F.binary_cross_entropy(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).mean(dim=-1)

        # Calculate cross-entropy wrpt a standard multivariate Gaussian
        KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        if self.contrastive_flag:

            # if self.distance_metric == "cosine":
            #     metric_loss = self.nt_xent_loss(features=mu, self_stats=self_stats, other_stats=other_stats)

            # elif self.distance_metric == "euclidean":
            metric_loss = self.nt_xent_loss_euclidean(features=mu, self_stats=self_stats,
                                                           other_stats=other_stats, temp_weights=hpf_deltas)
                
            # else:
            #     raise Exception("Invalid distance metric was passed to model.")

        elif self.triplet_flag:
            metric_loss = self.triplet_loss(features=mu)
        else:
            metric_loss = torch.tensor(0)

        recon_weight = (128*288) #/ (recon_x.shape[2]*recon_x.shape[3]) # this holds relative image recon weight constant
        latent_weight = 100 # standardize KLD loss

        # calculate weighted loss components
        metric_loss_w = self.metric_weight * metric_loss 
        recon_loss_w = recon_weight * recon_loss.mean(dim=0) 
        kld_loss_w = self.beta * KLD.mean(dim=0) * latent_weight

        return recon_loss_w + kld_loss_w + metric_loss_w, recon_loss_w, kld_loss_w, metric_loss_w
    
    # NT-Xent addapted for euclidean distances
    def nt_xent_loss_euclidean(self, features, temp_weights, self_stats=None, other_stats=None, n_views=2):

        temperature = self.temperature
        # remove latent dimensions that are intended to capture nuisance variability--these should not factor
        # into the contrastive loss
        features = features[:, self.biological_indices]

        # infer batch size
        batch_size = int(features.shape[0] / n_views)

        # EUCLIDEAN
        pair_matrix = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        pair_matrix = (pair_matrix.unsqueeze(0) == pair_matrix.unsqueeze(1)).float()
        mask = torch.eye(pair_matrix.shape[0], dtype=torch.bool).to(self.device)
        pair_matrix[mask] = -1  # exclude self comparisons
        # target_matrix = target_matrix.to(self.device)

        dist_matrix = torch.cdist(features, features, p=2).pow(2)

        # normalize distances to fall on scale similar to cosine. For stability, we will use expectation for 2 sigma of an isotropic gaussian as our "max"
        N = self.latent_dim_biological/2
        sigma = N # sigma^1 for an ND isotropic Gaussian
        dist_normed = (-(dist_matrix / sigma).pow(0.5) + 1) / temperature # Effectively a shifted z score. Note that large distances are permitted to go below -1

        # Apply temperature parameter
        target_matrix = torch.zeros(pair_matrix.shape, dtype=torch.float32)
        if self_stats is not None:
            age_vec = torch.cat([self_stats[1], other_stats[1]], axis=0)

            age_deltas = torch.abs(age_vec.unsqueeze(-1) - age_vec.unsqueeze(0))
            age_bool = age_deltas <= (self.model_config.time_window + 1.5)  # add an extra neutral "buffer" of 1.5 hrs
            if self.model_config.time_only_flag == 1:
                pert_bool = torch.ones_like(age_bool, dtype=torch.bool)
            elif self.model_config.self_target_prob < 1.0:
                pert_vec = torch.cat([self_stats[2], other_stats[2]], axis=0)
                pert_bool = pert_vec.unsqueeze(-1) == pert_vec.unsqueeze(0)  # avoid like perturbations
            else:
                # pert_vec = torch.cat([self_stats[0], other_stats[0]], axis=0)
                # pert_bool = pert_vec.unsqueeze(-1) == pert_vec.unsqueeze(0)  # avoid same embryo
                pert_bool = torch.ones_like(age_bool, dtype=torch.bool)

            cross_match_flags = age_bool & pert_bool  # things to keep out of the denomenator
            target_matrix[cross_match_flags] = 1

        target_matrix[pair_matrix == 1] = 1
        target_matrix[pair_matrix == -1] = -1
        target_matrix = target_matrix.to(self.device)

        # call multiclass nt_xent loss
        loss_euc = self.nt_xent_loss_multiclass(dist_normed, target_matrix)

        return loss_euc


    def nt_xent_loss_multiclass(self, logits_tempered, target):

        # Apply temperature parameter
        logits_tempered[target == -1] = -torch.inf # exclude flagged instances (self pair and cross-matched pairs)
        logits_num = logits_tempered.clone()
        logits_num[target == 0] = -torch.inf # exclude all negative pairs from numerator

        # calculate loss for each entry in the batch
        numerator = torch.logsumexp(logits_num, axis=1)
        denominator = torch.logsumexp(logits_tempered, axis=1)
        loss = -(numerator - denominator)

        return torch.mean(loss)

    def triplet_loss(self, features):

        # subset to just the biological partition
        features = features[:, self.biological_indices]
        margin = self.margin
        
        sigma = np.sqrt(self.latent_dim_biological/2) # scale to account for latent space dimensionality
        features_norm = features / sigma

        # infer batch size
        batch_size = int(features.shape[0] / 3)
        triplet_loss = TripletMarginLoss(margin=margin, p=2, eps=1e-7)
        trip_loss = triplet_loss(features_norm[0:batch_size, :],
                                 features_norm[batch_size:2*batch_size, :],
                                 features_norm[2*batch_size:, :])

        return trip_loss
        # pair_matrix[mask] = -1  # exclude self comparisons


     # def nt_xent_loss(self, features, self_stats, other_stats, n_views=2):

    #     temperature = self.temperature

    #     # remove latent dimensions that are intended to capture nuisance variability--these should not factor
    #     # into the contrastive loss
    #     features = features[:, self.biological_indices]

    #     # infer batch size
    #     batch_size = int(features.shape[0] / n_views)

    #     # COS approach
    #     # Normalize each latent vector. This simplifies the process of calculating cosie differences
    #     features_norm = F.normalize(features, dim=1)

    #     # Due to above normalization, sim matrix entries are same as cosine differences
    #     similarity_matrix = torch.matmul(features_norm, features_norm.T)

    #     # EUCLIDEAN
    #     pair_matrix = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    #     pair_matrix = (pair_matrix.unsqueeze(0) == pair_matrix.unsqueeze(1)).float()
    #     mask = torch.eye(pair_matrix.shape[0], dtype=torch.bool).to(self.device)
    #     pair_matrix[mask] = -1  # exclude self comparisons
    #     # target_matrix = target_matrix.to(self.device)


    #     # Apply temperature parameter
    #     logits = similarity_matrix / temperature

    #     target_matrix = torch.zeros(pair_matrix.shape, dtype=torch.float32)
    #     if self_stats is not None:
    #         age_vec = torch.cat([self_stats[1], other_stats[1]], axis=0)

    #         age_deltas = torch.abs(age_vec.unsqueeze(-1) - age_vec.unsqueeze(0))
    #         age_bool = age_deltas <= (self.model_config.time_window + 1.5)  # note extra buffer
    #         if self.model_config.self_target_prob < 1.0:
    #             pert_vec = torch.cat([self_stats[2], other_stats[2]], axis=0)
    #             pert_bool = pert_vec.unsqueeze(-1) == pert_vec.unsqueeze(0)  # avoid like perturbations
    #         else:
    #             pert_vec = torch.cat([self_stats[0], other_stats[0]], axis=0)
    #             pert_bool = pert_vec.unsqueeze(-1) == pert_vec.unsqueeze(0)  # avoid same embryo

    #         cross_match_flags = age_bool & pert_bool
    #         target_matrix[cross_match_flags] = -1

    #     target_matrix[pair_matrix == 1] = 1
    #     target_matrix[pair_matrix == -1] = -1
    #     target_matrix = target_matrix.to(self.device)

    #     loss = self.nt_xent_loss_multiclass(logits, target_matrix)

    #     return loss

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def clean_path_names(self, path_list):
        path_list_out = []
        for path in path_list:
            head, tail = ntpath.split(path)
            path_list_out.append(tail[:-4])

        return path_list_out

    def get_nll(self, data, n_samples=1, batch_size=100):
        """
        Function computed the estimate negative log-likelihood of the model. It uses importance
        sampling method with the approximate posterior distribution. This may take a while.

        Args:
            data (torch.Tensor): The input data from which the log-likelihood should be estimated.
                Data must be of shape [Batch x n_channels x ...]
            n_samples (int): The number of importance samples to use for estimation
            batch_size (int): The batchsize to use to avoid memory issues
        """

        if n_samples <= batch_size:
            n_full_batch = 1
        else:
            n_full_batch = n_samples // batch_size
            n_samples = batch_size

        log_p = []

        for i in range(len(data)):
            x = data[i].unsqueeze(0)

            log_p_x = []

            for j in range(n_full_batch):
                x_rep = torch.cat(batch_size * [x])

                encoder_output = self.encoder(x_rep)
                mu, log_var = encoder_output.embedding, encoder_output.log_covariance

                std = torch.exp(0.5 * log_var)
                z, _ = self._sample_gauss(mu, std)

                log_q_z_given_x = -0.5 * (
                        log_var + (z - mu) ** 2 / torch.exp(log_var)
                ).sum(dim=-1)
                log_p_z = -0.5 * (z ** 2).sum(dim=-1)

                recon_x = self.decoder(z)["reconstruction"]

                if self.model_config.reconstruction_loss == "mse":

                    log_p_x_given_z = -0.5 * F.mse_loss(
                        recon_x.reshape(x_rep.shape[0], -1),
                        x_rep.reshape(x_rep.shape[0], -1),
                        reduction="none",
                    ).sum(dim=-1) - torch.tensor(
                        [np.prod(self.input_dim) / 2 * np.log(np.pi * 2)]
                    ).to(
                        data.device
                    )  # decoding distribution is assumed unit variance  N(mu, I)

                elif self.model_config.reconstruction_loss == "bce":

                    log_p_x_given_z = -F.binary_cross_entropy(
                        recon_x.reshape(x_rep.shape[0], -1),
                        x_rep.reshape(x_rep.shape[0], -1),
                        reduction="none",
                    ).sum(dim=-1)

                log_p_x.append(
                    log_p_x_given_z + log_p_z - log_q_z_given_x
                )  # log(2*pi) simplifies

            log_p_x = torch.cat(log_p_x)

            log_p.append((torch.logsumexp(log_p_x, 0) - np.log(len(log_p_x))).item())
        return np.mean(log_p)

    # def nt_xent_loss_euclidean_orig(self, features, n_views=2):

    #     temperature = self.temperature

    #     # remove latent dimensions that are intended to capture nuisance variability--these should not factor
    #     # into the contrastive loss
    #     features = features[:, self.biological_indices]

    #     # infer batch size
    #     batch_size = int(features.shape[0] / n_views)

    #     # EUCLIDEAN
    #     labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    #     labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    #     labels = labels.to(self.device)

    #     dist_matrix = torch.cdist(features, features, p=2)
    #     mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
    #     labels = labels[~mask].view(labels.shape[0], -1)
    #     dist_matrix = dist_matrix[~mask].view(dist_matrix.shape[0], -1).pow(2)

    #     # select and combine multiple positives
    #     positives_euc = dist_matrix[labels.bool()].view(labels.shape[0], -1)

    #     # select only the negatives the negatives
    #     negatives_euc = dist_matrix[~labels.bool()].view(dist_matrix.shape[0], -1)

    #     # Construct logits matrix with positive examples as firs column
    #     distances_euc = torch.cat([positives_euc, negatives_euc], dim=1)

    #     # These labels tell the cross-entropy function that the positive example for each row is in the first column (col=0)
    #     labels = torch.zeros(distances_euc.shape[0], dtype=torch.long).to(self.device)

    #     # Apply temperature parameter
    #     distances_euc = -distances_euc / temperature

    #     # initialize cross entropy loss
    #     loss_fun = torch.nn.CrossEntropyLoss()
    #     loss_euc = loss_fun(distances_euc, labels)

    #     # target_matrix = torch.zeros(distances_euc.shape)
    #     # target_matrix[:, 0] = 1
    #     # loss_new = self.nt_xent_loss_multiclass(distances_euc, target_matrix)

    #     # Now from alternative direction
    #     # EUCLIDEAN
    #     # pair_matrix = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    #     # pair_matrix = (pair_matrix.unsqueeze(0) == pair_matrix.unsqueeze(1)).float()
    #     # mask = torch.eye(pair_matrix.shape[0], dtype=torch.bool).to(self.device)
    #     # pair_matrix[mask] = -1  # exclude self comparisons
    #     # # target_matrix = target_matrix.to(self.device)
    #     #
    #     # dist_matrix2 = torch.cdist(features, features, p=2).pow(2)
    #     #
    #     # # Apply temperature parameter
    #     # distances_euc2 = -dist_matrix2 / temperature
    #     # target_matrix2 = torch.zeros(pair_matrix.shape, dtype=torch.float32)
    #     #
    #     # target_matrix2[pair_matrix == 1] = 1
    #     # target_matrix2[pair_matrix == -1] = -1
    #     # target_matrix2 = target_matrix2.to(self.device)
    #     #
    #     # loss_new2 = self.nt_xent_loss_multiclass(distances_euc2, target_matrix2)

    #     return loss_euc

    # def nt_xent_loss_cosine_og(self, features, n_views=2):

    #     temperature = self.temperature

    #     # remove latent dimensions that are intended to capture nuisance variability--these should not factor
    #     # into the contrastive loss
    #     features = features[:, self.biological_indices]

    #     # infer batch size
    #     batch_size = int(features.shape[0] / n_views)

    #     labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    #     labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    #     labels = labels.to(self.device)

    #     # COS approach
    #     # Normalize each latent vector. This simplifies the process of calculating cosie differences
    #     features_norm = F.normalize(features, dim=1)

    #     # Due to above normalization, sim matrix entries are same as cosine differences
    #     similarity_matrix = torch.matmul(features_norm, features_norm.T)
    #     # assert similarity_matrix.shape == (
    #     #     n_views * batch_size, n_views * batch_size)
    #     # assert similarity_matrix.shape == labels.shape

    #     # discard the main diagonal, since this is the comparison of image with itself
    #     mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
    #     labels = labels[~mask].view(labels.shape[0], -1)
    #     similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    #     # assert similarity_matrix.shape == labels.shape

    #     # select and combine multiple positives
    #     positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    #     # select only the negatives the negatives
    #     negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    #     # Construct logits matrix with positive examples as firs column
    #     logits = torch.cat([positives, negatives], dim=1)

    #     # These labels tell the cross-entropy function that the positive example for each row is in the first column (col=0)
    #     labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

    #     # Apply temperature parameter
    #     logits = logits / temperature

    #     # initialize cross entropy loss
    #     loss_fun = torch.nn.CrossEntropyLoss()

    #     loss = loss_fun(logits, labels)

    #     return loss