from typing import Optional, Sequence  # List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

# import hmivae
from hmivae._hmivae_base_components import DecoderHMIVAE, EncoderHMIVAE

# from anndata import AnnData

torch.backends.cudnn.benchmark = True


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
        n_covariates: int = 0,
        use_covs: bool = False,
        n_hidden: int = 1,
        batch_correct: bool = True,
    ):
        super().__init__()
        # hidden_dim = E_me + E_cr + E_mr + E_sc
        self.n_covariates = n_covariates

        self.batch_correct = batch_correct

        self.use_covs = use_covs

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
            n_covariates=n_covariates,
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
            n_covariates=n_covariates,
        )

        self.save_hyperparameters(ignore=["adata", "cat_list"])

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
        # weights=None,
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
        log_p_xz_corr = p_rec_corr.log_prob(s)
        log_p_xz_morph = p_rec_morph.log_prob(m)
        log_p_xz_spcont = p_rec_spcont.log_prob(c)  # already dense matrix

        # if weights is None:
        #     log_p_xz_corr = p_rec_corr.log_prob(s)
        # else:
        #     log_p_xz_corr = torch.mul(
        #         weights, p_rec_corr.log_prob(s)
        #     )  # does element-wise multiplication

        log_p_xz_exp = log_p_xz_exp.sum(-1)
        log_p_xz_corr = log_p_xz_corr.sum(-1)
        log_p_xz_morph = log_p_xz_morph.sum(-1)
        log_p_xz_spcont = log_p_xz_spcont.sum(-1)

        return (
            log_p_xz_exp,
            log_p_xz_corr,
            log_p_xz_morph,
            log_p_xz_spcont,
        )

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
        # weights=None,
    ):
        kl_div = self.KL_div(enc_x_mu, enc_x_logstd, z)

        (
            recon_lik_me,
            recon_lik_corr,
            recon_lik_mor,
            recon_lik_sc,
        ) = self.em_recon_loss(
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
            # weights,
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

        if self.use_covs:
            categories = train_batch[5]
        else:
            categories = torch.Tensor([])

        if self.batch_correct:
            one_hot = train_batch[4]

            cov_list = torch.cat([one_hot, categories], 1).float()
        else:
            cov_list = torch.Tensor([])

        mu_z, log_std_z = self.encoder(Y, S, M, spatial_context, cov_list)

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
            # weights,
        ) = self.decoder(z_samples, cov_list)

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
            # weights,
        )

        recon_loss = (
            recon_weights[0] * recon_lik_me
            + recon_weights[1] * recon_lik_corr
            + recon_weights[2] * recon_lik_mor
            + recon_weights[3] * recon_lik_sc
        )

        loss = self.loss(kl_div, recon_loss, beta=beta)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return {
            "loss": loss,
            "kl_div": kl_div.mean().item(),
            "recon_loss": recon_loss.mean().item(),
            "recon_lik_me": recon_lik_me.mean().item(),
            "recon_lik_corr": recon_lik_corr.mean().item(),
            "recon_lik_mor": recon_lik_mor.mean().item(),
            "recon_lik_sc": recon_lik_sc.mean().item(),
        }

    def validation_step(
        self,
        test_batch,
        n_other_cat: int = 0,
        L_iter: int = 100,
        corr_weights=False,
        recon_weights=np.array([1.0, 1.0, 1.0, 1.0]),
        beta=1.0,
    ):
        """---> Add random one-hot encoding
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
        batch_idx = test_batch[-1]
        test_loss = []

        if self.use_covs:
            categories = test_batch[5]
            n_classes = self.n_covariates - categories.shape[1]
        else:
            categories = torch.Tensor([])
            n_classes = self.n_covariates

        for i in range(L_iter):

            if self.batch_correct:
                one_hot = self.random_one_hot(
                    n_classes=n_classes, n_samples=len(batch_idx)
                )

                cov_list = torch.cat([one_hot, categories], 1).float()
            else:
                cov_list = torch.Tensor([])

            mu_z, log_std_z = self.encoder(Y, S, M, spatial_context, cov_list)

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
                # weights,
            ) = self.decoder(z_samples, cov_list)

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
                # weights,
            )

            recon_loss = (
                recon_weights[0] * recon_lik_me
                + recon_weights[1] * recon_lik_corr
                + recon_weights[2] * recon_lik_mor
                + recon_weights[3] * recon_lik_sc
            )

            loss = self.loss(kl_div, recon_loss, beta=beta)

            test_loss.append(loss)

        self.log(
            "test_loss",
            sum(test_loss) / L_iter,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )  # log the average test loss over all the iterations

        return {
            "loss": sum(test_loss) / L_iter,
            "kl_div": kl_div.mean().item(),
            "recon_loss": recon_loss.mean().item(),
            "recon_lik_me": recon_lik_me.mean().item(),
            "recon_lik_corr": recon_lik_corr.mean().item(),
            "recon_lik_mor": recon_lik_mor.mean().item(),
            "recon_lik_sc": recon_lik_sc.mean().item(),
        }

    def configure_optimizers(self):
        """Optimizer"""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    @torch.no_grad()
    def get_input_embeddings(
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

    @torch.no_grad()
    def inference(
        self,
        data,
        indices: Optional[Sequence[int]] = None,
        give_mean: bool = True,
        # idx = None,
    ) -> np.ndarray:
        """
        Return the latent representation of each cell.
        """
        Y = data.Y
        S = data.S
        M = data.M
        C = data.C
        if self.use_covs:
            categories = data.BKG
            n_cats = categories.shape[1]
        else:
            categories = torch.Tensor([])
            n_cats = 0

        if self.batch_correct:
            one_hot = data.samples_onehot
            if one_hot.shape[1] < self.n_covariates - n_cats:
                zeros_pad = torch.Tensor(
                    np.zeros(
                        [
                            one_hot.shape[0],
                            (self.n_covariates - n_cats) - one_hot.shape[1],
                        ]
                    )
                )
                one_hot = torch.cat([one_hot, zeros_pad], 1)
            else:
                one_hot = one_hot

            cov_list = torch.cat([one_hot, categories], 1).float()
        else:
            cov_list = torch.Tensor([])

        if give_mean:
            mu_z, _ = self.encoder(Y, S, M, C, cov_list)

            return mu_z.numpy()
        else:
            mu_z, log_std_z = self.encoder(Y, S, M, C, cov_list)
            z = self.reparameterization(mu_z, log_std_z)

            return z.numpy()

    @torch.no_grad()
    def random_one_hot(self, n_classes: int, n_samples: int):
        """
        Generates a random one hot encoded matrix.
        From:  https://stackoverflow.com/questions/45093615/random-one-hot-matrix-in-numpy
        """

        return torch.Tensor(np.eye(n_classes)[np.random.choice(n_classes, n_samples)])
