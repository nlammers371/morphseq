from typing import Optional
import ntpath
import numpy as np
import torch
import torch.nn.functional as F
from pythae.data.datasets import BaseDataset
from src.vae.models import BaseAE
from src.vae.models import ModelOutput
from src.vae.models import BaseDecoder, BaseEncoder
from src.vae.models.nn.default_architectures import Encoder_VAE_MLP
from .metric_vae_config import MetricVAEConfig


class MetricVAE(BaseAE):
    """Vanilla Variational Autoencoder model.

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
            model_config: MetricVAEConfig,
            encoder: Optional[BaseEncoder] = None,
            decoder: Optional[BaseDecoder] = None,
    ):

        BaseAE.__init__(self, model_config=model_config, decoder=decoder)

        self.model_name = "MetricVAE"
        self.latent_dim = model_config.latent_dim
        self.zn_frac = model_config.zn_frac  # number of nuisance latent dimensions
        self.temperature = model_config.temperature
        self.distance_metric = model_config.distance_metric
        # self.gamma = model_config.gamma # weight factor for orth weight
        self.orth_flag = model_config.orth_flag  # indicates whether or not to impose orthogonality constraint
        self.contrastive_flag = True
        self.beta = model_config.beta
        # calculate number of "biological" and "nuisance" latent variables
        self.latent_dim_nuisance = torch.tensor(np.floor(self.latent_dim * self.zn_frac))
        self.latent_dim_biological = self.latent_dim - self.latent_dim_nuisance
        self.nuisance_indices = torch.arange(self.latent_dim_nuisance, dtype=torch.int)
        self.biological_indices = torch.arange(self.latent_dim_nuisance, self.latent_dim, dtype=torch.int)
        self.model_config = model_config
        # self.class_key = model_config.class_key
        # self.class_ignorance_flag = model_config.class_ignorance_flag
        # self.time_ignorance_flag = model_config.time_ignorance_flag
        # self.time_similarity_threshold = model_config.time_similarity_threshold
        # self.class_key_path = model_config.class_key_path
        # self.class_key = pd.read_csv(self.class_key_path)

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
        # y = None
        # if "label" in inputs:
        #     y = list(inputs["label"][0])

        # Check input to see if it is 5 dmensional, if so, then the model is being
        if (len(x.shape) != 5) or (x.shape[1] != 2):
            # raise Warning("Model did not receive contrastive pairs. No contrastive loss will be calculated.")
            self.contrastive_flag = False

        if self.contrastive_flag:
            x0 = torch.reshape(x[:, 0, :, :, :],
                               (x.shape[0], x.shape[2], x.shape[3], x.shape[4]))  # first set of images
            x1 = torch.reshape(x[:, 1, :, :, :], (
                x.shape[0], x.shape[2], x.shape[3], x.shape[4]))  # second set with matched contrastive pairs

            encoder_output0 = self.encoder(x0)
            encoder_output1 = self.encoder(x1)

            mu0, log_var0 = encoder_output0.embedding, encoder_output0.log_covariance
            mu1, log_var1 = encoder_output1.embedding, encoder_output1.log_covariance

            # weigth_matrix = None
            # if self.orth_flag:
            #     weight_matrix = encoder_output0.weight_matrix

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
            # if y is not None:
            #     y = y * 2

            loss, recon_loss, kld, nt_xent = self.loss_function(recon_x_out, x_out, mu_out, log_var_out)#, labels=y)

        else:
            encoder_output = self.encoder(x)

            mu, log_var = encoder_output.embedding, encoder_output.log_covariance

            std = torch.exp(0.5 * log_var)

            z_out, eps = self._sample_gauss(mu, std)
            recon_x_out = self.decoder(z_out)["reconstruction"]

            loss, recon_loss, kld, nt_xent = self.loss_function(recon_x_out, x, mu, log_var) #labels=y)  # , z_out,
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

    def loss_function(self, recon_x, x, mu, log_var): #, labels=None):

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
            ).sum(dim=-1)
            )

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = F.binary_cross_entropy(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        # Calculate cross-entropy wrpt a standard multivariate Gaussian
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        if self.contrastive_flag:

            if self.distance_metric == "cosine":
                nt_xent_loss = self.nt_xent_loss(features=mu)
            elif self.distance_metric == "euclidean":
                nt_xent_loss = self.nt_xent_loss_euclidean(features=mu)
            else:
                raise Exception("Invalid distance metric was passed to model.")
        else:
            nt_xent_loss = torch.tensor(0)

        # if self.class_ignorance_flag:
        #     ntx_knowledge_loss = self.calculate_knowledge_loss(features=mu, labels=labels)
        # else:
        #     ntx_knowledge_loss = torch.tensor(0)


        return recon_loss.mean(dim=0) + self.beta*KLD.mean(dim=0) + nt_xent_loss, recon_loss.mean(
            dim=0), KLD.mean(
            dim=0), nt_xent_loss

    def nt_xent_loss(self, features, n_views=2):

        temperature = self.temperature

        # remove latent dimensions that are intended to capture nuisance variability--these should not factor
        # into the contrastive loss
        features = features[:, self.biological_indices]

        # infer batch size
        batch_size = int(features.shape[0] / n_views)

        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        # COS approach
        # Normalize each latent vector. This simplifies the process of calculating cosie differences
        features_norm = F.normalize(features, dim=1)

        # Due to above normalization, sim matrix entries are same as cosine differences
        similarity_matrix = torch.matmul(features_norm, features_norm.T)
        # assert similarity_matrix.shape == (
        #     n_views * batch_size, n_views * batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal, since this is the comparison of image with itself
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        # Construct logits matrix with positive examples as firs column
        logits = torch.cat([positives, negatives], dim=1)

        # These labels tell the cross-entropy function that the positive example for each row is in the first column (col=0)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        # Apply temperature parameter
        logits = logits / temperature

        # initialize cross entropy loss
        loss_fun = torch.nn.CrossEntropyLoss()

        loss = loss_fun(logits, labels)

        return loss

    def nt_xent_loss_euclidean(self, features, n_views=2):

        temperature = self.temperature

        # remove latent dimensions that are intended to capture nuisance variability--these should not factor
        # into the contrastive loss
        features = features[:, self.biological_indices]

        # infer batch size
        batch_size = int(features.shape[0] / n_views)

        # EUCLIDEAN
        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        dist_matrix = torch.cdist(features, features, p=2)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        dist_matrix = dist_matrix[~mask].view(dist_matrix.shape[0], -1).pow(2)

        # select and combine multiple positives
        positives_euc = dist_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives_euc = dist_matrix[~labels.bool()].view(dist_matrix.shape[0], -1)

        # Construct logits matrix with positive examples as firs column
        distances_euc = torch.cat([positives_euc, negatives_euc], dim=1)

        # These labels tell the cross-entropy function that the positive example for each row is in the first column (col=0)
        labels = torch.zeros(distances_euc.shape[0], dtype=torch.long).to(self.device)

        # Apply temperature parameter
        distances_euc = -distances_euc / temperature

        # initialize cross entropy loss
        loss_fun = torch.nn.CrossEntropyLoss()
        loss_euc = loss_fun(distances_euc, labels)

        return loss_euc

    # def calculate_knowledge_loss(self, features, labels, n_views=2, dt_thresh=1.5):
    #
    #     if self.class_key is None:
    #         raise Exception("Nuisance metric learning requested, but not class information was passed to model.")
    #     else:
    #         class_key = self.class_key
    #
    #     if self.time_ignorance_flag:
    #         dt_thresh = self.time_similarity_threshold
    #         assert dt_thresh >= 0
    #
    #     features = features[:, self.nuisance_indices]
    #
    #     # calculate euclidean distances
    #     # temperature = self.temperature
    #     dist_matrix = torch.cdist(features, features, p=2)
    #     # logits = dist_matrix / temperature
    #
    #     # use class key to calculate embryos that are close enough wrpt time and perturbation to count as positive
    #     # examples
    #     class_key_batch = pd.DataFrame(labels, columns=["snip_id"])
    #     class_key_batch = class_key_batch.merge(class_key, how="left", on="snip_id")
    #
    #     if self.time_ignorance_flag:
    #         # calculate time pairs
    #         time_tensor = torch.tensor(class_key_batch["predicted_stage_hpf"].values).to(self.device)
    #         tdist_matrix = torch.cdist(time_tensor[:, np.newaxis], time_tensor[:, np.newaxis], p=2)
    #         tbool_matrix = tdist_matrix <= dt_thresh
    #     else:
    #         tbool_matrix = torch.ones((dist_matrix.shape))
    #
    #     if self.class_ignorance_flag:
    #         # calculate class pairs
    #         class_tensor = torch.tensor(class_key_batch["perturbation_id"].values).to(self.device)
    #         cbool_matrix = (class_tensor.unsqueeze(0) == class_tensor.unsqueeze(1)).float()
    #     else:
    #         cbool_matrix = torch.ones((dist_matrix.shape))
    #
    #     # construct master target matrix
    #     # batch_size = int(features.shape[0] / n_views)
    #
    #     # EUCLIDEAN
    #     # batch_labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0).to(self.device)
    #     # target_matrix = (batch_labels.unsqueeze(0) == batch_labels.unsqueeze(1)).float()
    #     # labels = labels.to(self.device)
    #
    #     target_matrix = (torch.multiply(cbool_matrix, tbool_matrix) == 0).type(torch.float32)
    #     # target_matrix = (torch.multiply(cbool_matrix, tbool_matrix) > 0).type(torch.float32)
    #
    #     mask = torch.eye(features.shape[0], dtype=torch.bool)  # .to(self.device)
    #     target_matrix[mask == 1] = -1  # this excludes self-pairs from all calculations
    #
    #     # call multiclass nt_xent loss
    #     loss = self.nt_xent_loss_multiclass(dist_matrix, target_matrix, repel_flag=False)
    #
    #     return loss

    # def nt_xent_loss_multiclass(self, logits, target, repel_flag=False):
    #     # a multiclass version of the NT-Xent loss function
    #     logit_sign = -1
    #     if repel_flag:
    #         logit_sign = 1
    #
    #     temperature = self.temperature
    #
    #     # Apply temperature parameter
    #     logits_tempered = logit_sign * logits / temperature
    #     logits_tempered[target == -1] = -torch.inf
    #     logits_num = logits_tempered.clone()
    #     logits_num[target == 0] = -torch.inf
    #
    #     # calculate loss for each entry in the batch
    #     numerator = -torch.logsumexp(logits_num, axis=1)
    #     denominator = torch.logsumexp(logits_tempered, axis=1)
    #     loss = numerator + denominator
    #
    #     return torch.mean(loss)

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