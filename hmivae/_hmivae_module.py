from typing import Iterable, Optional  # Dict, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from _hmivae_base_components import DecoderHMIVAE, EncoderHMIVAE

# from scvi import _CONSTANTS
# from scvi.distributions import ZeroInflatedNegativeBinomial
# from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
# from scvi.nn import one_hot
# from torch.distributions import Normal
# from torch.distributions import kl_divergence as kl

torch.backends.cudnn.benchmark = True


# class HMIVAE(BaseModuleClass):
#     """
#     Variational auto-encoder model.

#     Here we implement a basic version of scVI's underlying VAE [Lopez18]_.
#     This implementation is for instructional purposes only.

#     Parameters
#     ----------
#     n_input
#         Number of input genes
#     library_log_means
#         1 x n_batch array of means of the log library sizes. Parameterizes prior on library size if
#         not using observed library size.
#     library_log_vars
#         1 x n_batch array of variances of the log library sizes. Parameterizes prior on library size if
#         not using observed library size.
#     n_batch
#         Number of batches, if 0, no batch correction is performed.
#     n_hidden
#         Number of nodes per hidden layer
#     n_latent
#         Dimensionality of the latent space
#     n_layers
#         Number of hidden layers used for encoder and decoder NNs
#     dropout_rate
#         Dropout rate for neural networks
#     """

#     def __init__(
#         self,
#         n_input: int,
#         n_batch: int = 0,
#         n_hidden: int = 128,
#         n_latent: int = 10,
#         n_layers: int = 1,
#         dropout_rate: float = 0.1,
#     ):
#     # def __init__(
#     #     self,
#     #     input_exp_dim: int,
#     #     input_corr_dim: int,
#     #     input_morph_dim: int,
#     #     input_spcont_dim: int,
#     #     E_me: int,
#     #     E_cr: int,
#     #     E_mr: int,
#     #     E_sc: int,
#     #     n_latent: int = 10,
#     #     n_batch: int = 0,
#     #     n_hidden: int = 1,
#     # ):
#         super().__init__()
#         self.n_latent = n_latent
#         self.n_batch = n_batch
#         # this is needed to comply with some requirement of the VAEMixin class
#         self.latent_distribution = "normal"

#         self.register_buffer(
#             "library_log_means", torch.from_numpy(library_log_means).float()
#         )
#         self.register_buffer(
#             "library_log_vars", torch.from_numpy(library_log_vars).float()
#         )

#         # setup the parameters of your generative model, as well as your inference model
#         self.px_r = torch.nn.Parameter(torch.randn(n_input))
#         # z encoder goes from the n_input-dimensional data to an n_latent-d
#         # latent space representation
#         self.z_encoder = EncoderHMIVAE(
#             n_input,
#             n_latent,
#             n_layers=n_layers,
#             n_hidden=n_hidden,
#             dropout_rate=dropout_rate,
#         )
#         # l encoder goes from n_input-dimensional data to 1-d library size
#         self.l_encoder = EncoderHMIVAE(
#             n_input,
#             1,
#             n_layers=1,
#             n_hidden=n_hidden,
#             dropout_rate=dropout_rate,
#         )
#         # decoder goes from n_latent-dimensional space to n_input-d data
#         self.decoder = DecoderHMIVAE(
#             n_latent,
#             n_input,
#             n_layers=n_layers,
#             n_hidden=n_hidden,
#         )

#     def _get_inference_input(self, tensors):
#         """Parse the dictionary to get appropriate args"""
#         x = tensors[_CONSTANTS.X_KEY]

#         input_dict = dict(x=x)
#         return input_dict

#     def _get_generative_input(self, tensors, inference_outputs):
#         z = inference_outputs["z"]
#         library = inference_outputs["library"]

#         input_dict = {
#             "z": z,
#             "library": library,
#         }
#         return input_dict

#     @auto_move_data
#     def inference(self, x):
#         """
#         High level inference method.

#         Runs the inference (encoder) model.
#         """
#         # log the input to the variational distribution for numerical stability
#         x_ = torch.log(1 + x)
#         # get variational parameters via the encoder networks
#         qz_m, qz_v, z = self.z_encoder(x_)
#         ql_m, ql_v, library = self.l_encoder(x_)

#         outputs = dict(z=z, qz_m=qz_m, qz_v=qz_v, ql_m=ql_m, ql_v=ql_v, library=library)
#         return outputs

#     @auto_move_data
#     def generative(self, z, library):
#         """Runs the generative model."""

#         # form the parameters of the ZINB likelihood
#         px_scale, _, px_rate, px_dropout = self.decoder("gene", z, library)
#         px_r = torch.exp(self.px_r)

#         return dict(
#             px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout
#         )

#     def loss(
#         self,
#         tensors,
#         inference_outputs,
#         generative_outputs,
#         kl_weight: float = 1.0,
#     ):
#         x = tensors[_CONSTANTS.X_KEY]
#         qz_m = inference_outputs["qz_m"]
#         qz_v = inference_outputs["qz_v"]
#         ql_m = inference_outputs["ql_m"]
#         ql_v = inference_outputs["ql_v"]
#         px_rate = generative_outputs["px_rate"]
#         px_r = generative_outputs["px_r"]
#         px_dropout = generative_outputs["px_dropout"]

#         mean = torch.zeros_like(qz_m)
#         scale = torch.ones_like(qz_v)

#         kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(
#             dim=1
#         )

#         batch_index = tensors[_CONSTANTS.BATCH_KEY]
#         n_batch = self.library_log_means.shape[1]
#         local_library_log_means = F.linear(
#             one_hot(batch_index, n_batch), self.library_log_means
#         )
#         local_library_log_vars = F.linear(
#             one_hot(batch_index, n_batch), self.library_log_vars
#         )

#         kl_divergence_l = kl(
#             Normal(ql_m, torch.sqrt(ql_v)),
#             Normal(local_library_log_means, torch.sqrt(local_library_log_vars)),
#         ).sum(dim=1)

#         reconst_loss = (
#             -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
#             .log_prob(x)
#             .sum(dim=-1)
#         )

#         kl_local_for_warmup = kl_divergence_z
#         kl_local_no_warmup = kl_divergence_l

#         weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

#         loss = torch.mean(reconst_loss + weighted_kl_local)

#         kl_local = dict(
#             kl_divergence_l=kl_divergence_l, kl_divergence_z=kl_divergence_z
#         )
#         kl_global = torch.tensor(0.0)
#         return LossRecorder(loss, reconst_loss, kl_local, kl_global)

#     @torch.no_grad()
#     def sample(
#         self,
#         tensors,
#         n_samples=1,
#         library_size=1,
#     ) -> np.ndarray:
#         r"""
#         Generate observation samples from the posterior predictive distribution.

#         The posterior predictive distribution is written as :math:`p(\hat{x} \mid x)`.

#         Parameters
#         ----------
#         tensors
#             Tensors dict
#         n_samples
#             Number of required samples for each cell
#         library_size
#             Library size to scale scamples to

#         Returns
#         -------
#         x_new : :py:class:`torch.Tensor`
#             tensor with shape (n_cells, n_genes, n_samples)
#         """
#         inference_kwargs = dict(n_samples=n_samples)
#         _, generative_outputs, = self.forward(
#             tensors,
#             inference_kwargs=inference_kwargs,
#             compute_loss=False,
#         )

#         px_r = generative_outputs["px_r"]
#         px_rate = generative_outputs["px_rate"]
#         px_dropout = generative_outputs["px_dropout"]

#         dist = ZeroInflatedNegativeBinomial(
#             mu=px_rate, theta=px_r, zi_logits=px_dropout
#         )

#         if n_samples > 1:
#             exprs = dist.sample().permute(
#                 [1, 2, 0]
#             )  # Shape : (n_cells_batch, n_genes, n_samples)
#         else:
#             exprs = dist.sample()

#         return exprs.cpu()

#     @torch.no_grad()
#     @auto_move_data
#     def marginal_ll(self, tensors, n_mc_samples):
#         sample_batch = tensors[_CONSTANTS.X_KEY]
#         batch_index = tensors[_CONSTANTS.BATCH_KEY]

#         to_sum = torch.zeros(sample_batch.size()[0], n_mc_samples)

#         for i in range(n_mc_samples):
#             # Distribution parameters and sampled variables
#             inference_outputs, _, losses = self.forward(tensors)
#             qz_m = inference_outputs["qz_m"]
#             qz_v = inference_outputs["qz_v"]
#             z = inference_outputs["z"]
#             ql_m = inference_outputs["ql_m"]
#             ql_v = inference_outputs["ql_v"]
#             library = inference_outputs["library"]

#             # Reconstruction Loss
#             reconst_loss = losses.reconstruction_loss

#             # Log-probabilities
#             n_batch = self.library_log_means.shape[1]
#             local_library_log_means = F.linear(
#                 one_hot(batch_index, n_batch), self.library_log_means
#             )
#             local_library_log_vars = F.linear(
#                 one_hot(batch_index, n_batch), self.library_log_vars
#             )
#             p_l = (
#                 Normal(local_library_log_means, local_library_log_vars.sqrt())
#                 .log_prob(library)
#                 .sum(dim=-1)
#             )

#             p_z = (
#                 Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v))
#                 .log_prob(z)
#                 .sum(dim=-1)
#             )
#             p_x_zl = -reconst_loss
#             q_z_x = Normal(qz_m, qz_v.sqrt()).log_prob(z).sum(dim=-1)
#             q_l_x = Normal(ql_m, ql_v.sqrt()).log_prob(library).sum(dim=-1)

#             to_sum[:, i] = p_z + p_l + p_x_zl - q_z_x - q_l_x

#         batch_log_lkl = torch.logsumexp(to_sum, dim=-1) - np.log(n_mc_samples)
#         log_lkl = torch.sum(batch_log_lkl).item()
#         return log_lkl


class hmiVAE(pl.LightningModule):
    """
    Variational Autoencoder for hmiVAE based on pytorch-lightning.
    """

    def __init__(
        self,
        input_exp_dim: int,
        input_corr_dim: int,
        input_morph_dim: int,
        input_spcont_dim: int,
        E_me: int = 32,
        E_cr: int = 32,
        E_mr: int = 32,
        E_sc: int = 32,
        latent_dim: int = 10,
        n_hidden: int = 1,
    ):
        super().__init__()
        # hidden_dim = E_me + E_cr + E_mr + E_sc

        self.encoder = EncoderHMIVAE(
            input_exp_dim,
            input_corr_dim,
            input_morph_dim,
            input_spcont_dim,
            E_me,
            E_cr,
            E_mr,
            E_sc,
            latent_dim,
        )

        self.decoder = DecoderHMIVAE(
            latent_dim,
            E_me,
            E_cr,
            E_mr,
            E_sc,
            input_exp_dim,
            input_corr_dim,
            input_morph_dim,
            input_spcont_dim,
        )

    def reparameterization(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(log_std)

        # sampling from encoded distribution
        z_samples = mu + eps * std

        return z_samples

    def KL_div(self, enc_x_mu, enc_x_logstd, z):
        """Takes in the encoded x mu and sigma, and the z sampled from
        q, and outputs the KL-Divergence term in ELBO"""

        p = torch.distributions.Normal(
            torch.zeros_like(enc_x_mu), torch.ones_like(enc_x_logstd)
        )
        enc_x_std = torch.exp(enc_x_logstd)
        q = torch.distributions.Normal(enc_x_mu, enc_x_std + 1e-6)

        log_q_zx = q.log_prob(z)
        log_p_z = p.log_prob(z)

        kl = log_q_zx - log_p_z
        kl = kl.sum(-1)

        return kl

    def em_recon_loss(
        self,
        dec_x_mu_exp,
        dec_x_logstd_exp,
        dec_x_mu_corr,
        dec_x_logstd_corr,
        dec_x_mu_morph,
        dec_x_logstd_morph,
        dec_x_mu_spcont,
        dec_x_logstd_spcont,
        y,
        s,
        m,
        c,
        weights=None,
    ):
        """Takes in the parameters output from the decoder,
        and the original input x, and gives the reconstruction
        loss term in ELBO
        dec_x_mu_exp: torch.Tensor, decoded means for protein expression feature
        dec_x_logstd_exp: torch.Tensor, decoded log std for protein expression feature
        dec_x_mu_corr: torch.Tensor, decoded means for correlation feature
        dec_x_logstd_corr: torch.Tensor, decoded log std for correlations feature
        dec_x_mu_morph: torch.Tensor, decoded means for morphology feature
        dec_x_logstd_morph: torch.Tensor, decoded log std for morphology feature
        dec_x_mu_spcont: torch.Tensor, decoded means for spatial context feature
        dec_x_logstd_spcont: torch.Tensor, decoded log std for spatial context feature
        y: torch.Tensor, original mean expression input
        s: torch.Tensor, original correlation input
        m: torch.Tensor, original morphology input
        c: torch.Tensor, original cell context input
        weights: torch.Tensor, weights calculated from decoded means for protein expression feature
        """

        dec_x_std_exp = torch.exp(dec_x_logstd_exp)
        dec_x_std_corr = torch.exp(dec_x_logstd_corr)
        dec_x_std_morph = torch.exp(dec_x_logstd_morph)
        dec_x_std_spcont = torch.exp(dec_x_logstd_spcont)
        p_rec_exp = torch.distributions.Normal(dec_x_mu_exp, dec_x_std_exp + 1e-6)
        p_rec_corr = torch.distributions.Normal(dec_x_mu_corr, dec_x_std_corr + 1e-6)
        p_rec_morph = torch.distributions.Normal(dec_x_mu_morph, dec_x_std_morph + 1e-6)
        p_rec_spcont = torch.distributions.Normal(
            dec_x_mu_spcont, dec_x_std_spcont + 1e-6
        )

        log_p_xz_exp = p_rec_exp.log_prob(y)
        log_p_xz_morph = p_rec_morph.log_prob(m)
        log_p_xz_spcont = p_rec_spcont.log_prob(c)  # already dense matrix

        if weights is None:
            log_p_xz_corr = p_rec_corr.log_prob(s)
        else:
            log_p_xz_corr = torch.mul(
                weights, p_rec_corr.log_prob(s)
            )  # does element-wise multiplication

        log_p_xz_exp = log_p_xz_exp.sum(-1)
        log_p_xz_corr = log_p_xz_corr.sum(-1)
        log_p_xz_morph = log_p_xz_morph.sum(-1)
        log_p_xz_spcont = log_p_xz_spcont.sum(-1)

        return log_p_xz_exp, log_p_xz_corr, log_p_xz_morph, log_p_xz_spcont

    def neg_ELBO(
        self,
        enc_x_mu,
        enc_x_logstd,
        dec_x_mu_exp,
        dec_x_logstd_exp,
        dec_x_mu_corr,
        dec_x_logstd_corr,
        dec_x_mu_morph,
        dec_x_logstd_morph,
        dec_x_mu_spcont,
        dec_x_logstd_spcont,
        z,
        y,
        s,
        m,
        c,
        weights=None,
    ):
        kl_div = self.KL_div(enc_x_mu, enc_x_logstd, z)

        recon_lik_me, recon_lik_corr, recon_lik_mor, recon_lik_sc = self.em_recon_loss(
            dec_x_mu_exp,
            dec_x_logstd_exp,
            dec_x_mu_corr,
            dec_x_logstd_corr,
            dec_x_mu_morph,
            dec_x_logstd_morph,
            dec_x_mu_spcont,
            dec_x_logstd_spcont,
            y,
            s,
            m,
            c,
            weights,
        )
        return (
            kl_div,
            recon_lik_me,
            recon_lik_corr,
            recon_lik_mor,
            recon_lik_sc,
        )

    def loss(self, kl_div, recon_loss, beta: float = 1.0):

        return beta * kl_div.mean() - recon_loss.mean()

    def training_step(
        self,
        train_batch,
        spatial_context,
        batch_idx,
        categories: Optional[Iterable[int]] = None,
        corr_weights=False,
        recon_weights=np.array([1.0, 1.0, 1.0, 1.0]),
        beta=1.0,
    ):
        """
        Carries out the training step.
        train_batch: torch.Tensor. Training data,
        spatial_context: torch.Tensor. Matrix with old mu_z integrated neighbours information,
        corr_weights: numpy.array. Array with weights for the correlations for each cell.
        recon_weights: numpy.array. Array with weights for each view during loss calculation.
        beta: float. Coefficient for KL-Divergence term in ELBO.
        """
        Y = train_batch[0]
        S = train_batch[1]
        M = train_batch[2]
        spatial_context = train_batch[3]

        mu_z, log_std_z = self.encoder(Y, S, M, spatial_context)

        z_samples = self.reparameterization(mu_z, log_std_z)

        # decoding
        (
            mu_x_exp_hat,
            log_std_x_exp_hat,
            mu_x_corr_hat,
            log_std_x_corr_hat,
            mu_x_morph_hat,
            log_std_x_morph_hat,
            mu_x_spcont_hat,
            log_std_x_spcont_hat,
            weights,
        ) = self.decoder(z_samples)

        (
            kl_div,
            recon_lik_me,
            recon_lik_corr,
            recon_lik_mor,
            recon_lik_sc,
        ) = self.neg_ELBO(
            mu_z,
            log_std_z,
            mu_x_exp_hat,
            log_std_x_exp_hat,
            mu_x_corr_hat,
            log_std_x_corr_hat,
            mu_x_morph_hat,
            log_std_x_morph_hat,
            mu_x_spcont_hat,
            log_std_x_spcont_hat,
            z_samples,
            Y,
            S,
            M,
            spatial_context,
            weights,
        )

        recon_loss = (
            recon_weights[0] * recon_lik_me
            + recon_weights[1] * recon_lik_corr
            + recon_weights[2] * recon_lik_mor
            + recon_weights[3] * recon_lik_sc
        )

        loss = self.loss(kl_div, recon_loss, beta=beta)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return (
            loss,
            kl_div.mean().item(),
            recon_loss.mean().item(),
            recon_lik_me.mean().item(),
            recon_lik_corr.mean().item(),
            recon_lik_mor.mean().item(),
            recon_lik_sc.mean().item(),
        )

    def test_step(
        self,
        test_batch,
        spatial_context,
        batch_idx,
        corr_weights=False,
        recon_weights=np.array([1.0, 1.0, 1.0, 1.0]),
        beta=1.0,
    ):
        """
        Carries out the validation/test step.
        test_batch: torch.Tensor. Validation/test data,
        spatial_context: torch.Tensor. Matrix with old mu_z integrated neighbours information,
        corr_weights: numpy.array. Array with weights for the correlations for each cell.
        recon_weights: numpy.array. Array with weights for each view during loss calculation.
        beta: float. Coefficient for KL-Divergence term in ELBO.
        """
        Y = test_batch[0]
        S = test_batch[1]
        M = test_batch[2]
        spatial_context = test_batch[3]

        mu_z, log_std_z, z1 = self.encode(Y, S, M, spatial_context)

        z_samples = self.reparameterization(mu_z, log_std_z)

        # decoding
        (
            mu_x_exp_hat,
            log_std_x_exp_hat,
            mu_x_corr_hat,
            log_std_x_corr_hat,
            mu_x_morph_hat,
            log_std_x_morph_hat,
            mu_x_spcont_hat,
            log_std_x_spcont_hat,
            weights,
        ) = self.decode(z_samples)

        (
            kl_div,
            recon_lik_me,
            recon_lik_corr,
            recon_lik_mor,
            recon_lik_sc,
        ) = self.neg_ELBO(
            mu_z,
            log_std_z,
            mu_x_exp_hat,
            log_std_x_exp_hat,
            mu_x_corr_hat,
            log_std_x_corr_hat,
            mu_x_morph_hat,
            log_std_x_morph_hat,
            mu_x_spcont_hat,
            log_std_x_spcont_hat,
            z_samples,
            Y,
            S,
            M,
            spatial_context,
            weights,
        )

        recon_loss = (
            recon_weights[0] * recon_lik_me
            + recon_weights[1] * recon_lik_corr
            + recon_weights[2] * recon_lik_mor
            + recon_weights[3] * recon_lik_sc
        )

        loss = self.loss(kl_div, recon_loss, beta=beta)

        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return (
            loss,
            kl_div.mean().item(),
            recon_loss.mean().item(),
            recon_lik_me.mean().item(),
            recon_lik_corr.mean().item(),
            recon_lik_mor.mean().item(),
            recon_lik_sc.mean().item(),
        )

    def configure_optimizers(self):
        """Optimizer"""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def __get_input_embeddings__(
        self, x_mean, x_correlations, x_morphology, x_spatial_context
    ):
        """
        Returns the view-specific embeddings.
        """
        h_mean = F.elu(self.input_exp(x_mean))
        h_mean2 = F.elu(self.exp_hidden(h_mean))

        h_correlations = F.elu(self.input_corr(x_correlations))
        h_correlations2 = F.elu(self.corr_hidden(h_correlations))

        h_morphology = F.elu(self.input_morph(x_morphology))
        h_morphology2 = F.elu(self.morph_hidden(h_morphology))

        h_spatial_context = F.elu(self.input_spatial_context(x_spatial_context))
        h_spatial_context2 = F.elu(self.spatial_context_hidden(h_spatial_context))

        return h_mean2, h_correlations2, h_morphology2, h_spatial_context2
