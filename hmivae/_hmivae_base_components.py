import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderHMIVAE(nn.Module):
    """Encoder for the case in which data is merged after initial encoding
    input_exp_dim: Dimension for the original mean expression input
    input_corr_dim: Dimension for the original correlations input
    input_morph_dim: Dimension for the original morphology input
    input_spcont_dim: Dimension for the original spatial context input
    E_me: Dimension for the encoded mean expressions input
    E_cr: Dimension for the encoded correlations input
    E_mr: Dimension for the encoded morphology input
    E_sc: Dimension for the encoded spatial context input
    latent_dim: Dimension of the encoded output
    n_hidden: Number of hidden layers, default=1
    """

    def __init__(
        self,
        input_exp_dim,
        input_corr_dim,
        input_morph_dim,
        input_spcont_dim,
        E_me,
        E_cr,
        E_mr,
        E_sc,
        latent_dim,
        n_hidden=1,
    ):
        super().__init__()
        hidden_dim = E_me + E_cr + E_mr + E_sc

        self.input_exp = nn.Linear(input_exp_dim, E_me)
        self.exp_hidden = nn.Linear(E_me, E_me)

        self.input_corr = nn.Linear(input_corr_dim, E_cr)
        self.corr_hidden = nn.Linear(E_cr, E_cr)
        self.input_morph = nn.Linear(input_morph_dim, E_mr)
        self.morph_hidden = nn.Linear(E_mr, E_mr)
        self.input_spatial_context = nn.Linear(input_spcont_dim, E_sc)
        self.spatial_context_hidden = nn.Linear(E_sc, E_sc)

        self.linear = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for i in range(n_hidden)]
        )

        self.mu_z = nn.Linear(hidden_dim, latent_dim)
        self.std_z = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x_mean, x_correlations, x_morphology, x_spatial_context):
        h_mean = F.elu(self.input_exp(x_mean))
        h_mean2 = F.elu(self.exp_hidden(h_mean))

        h_correlations = F.elu(self.input_corr(x_correlations))
        h_correlations2 = F.elu(self.corr_hidden(h_correlations))

        h_morphology = F.elu(self.input_morph(x_morphology))
        h_morphology2 = F.elu(self.morph_hidden(h_morphology))

        # z1 = torch.cat([h_mean2, h_correlations2, h_morphology2], 1)

        h_spatial_context = F.elu(self.input_spatial_context(x_spatial_context))
        h_spatial_context2 = F.elu(self.spatial_context_hidden(h_spatial_context))

        h = torch.cat([h_mean2, h_correlations2, h_morphology2, h_spatial_context2], 1)

        for net in self.linear:
            h = F.elu(net(h))

        mu_z = self.mu_z(h)

        log_std_z = self.std_z(h)

        return mu_z, log_std_z


class DecoderHMIVAE(nn.Module):
    """
    Decoder for the case where data is merged after inital encoding
    latent_dim: Dimension of the encoded input
    E_me: Dimension for the encoded mean expressions input
    E_cr: Dimension for the encoded correlations input
    E_mr: Dimension for the encoded morphology input
    E_sc: Dimension for the encoded spatial context input
    input_exp_dim: Dimension for the decoded mean expression output
    input_corr_dim: Dimension for the decoded correlations output
    input_morph_dim: Dimension for the decoded morphology input
    input_spcont_dim: Dimension for the decoded spatial context input
    n_hidden: Number of hidden layers, default=1
    """

    def __init__(
        self,
        latent_dim,
        E_me,
        E_cr,
        E_mr,
        E_sc,
        input_exp_dim,
        input_corr_dim,
        input_morph_dim,
        input_spcont_dim,
        n_hidden=1,
    ):
        super().__init__()
        hidden_dim = E_me + E_cr + E_mr + E_sc
        self.E_me = E_me
        self.E_cr = E_cr
        self.E_mr = E_mr
        self.E_sc = E_sc
        self.input = nn.Linear(latent_dim, hidden_dim)
        self.linear = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for i in range(n_hidden)]
        )
        # mean expression
        self.exp_hidden = nn.Linear(E_me, E_me)
        self.mu_x_exp = nn.Linear(E_me, input_exp_dim)
        self.std_x_exp = nn.Linear(E_me, input_exp_dim)

        # correlations/co-localizations
        self.corr_hidden = nn.Linear(E_cr, E_cr)
        self.mu_x_corr = nn.Linear(E_cr, input_corr_dim)
        self.std_x_corr = nn.Linear(E_cr, input_corr_dim)

        # morphology
        self.morph_hidden = nn.Linear(E_mr, E_mr)
        self.mu_x_morph = nn.Linear(E_mr, input_morph_dim)
        self.std_x_morph = nn.Linear(E_mr, input_morph_dim)

        # spatial context
        self.spatial_context_hidden = nn.Linear(E_sc, E_sc)
        self.mu_x_spcont = nn.Linear(E_sc, input_spcont_dim)
        self.std_x_spcont = nn.Linear(E_sc, input_spcont_dim)

    def forward(self, z):
        out = F.elu(self.input(z))
        for net in self.linear:
            out = F.elu(net(out))

        h2_mean = F.elu(self.exp_hidden(out[:, 0 : self.E_me]))
        h2_correlations = F.elu(
            self.corr_hidden(out[:, self.E_me : self.E_me + self.E_cr])
        )
        h2_morphology = F.elu(
            self.morph_hidden(
                out[:, self.E_me + self.E_cr : self.E_me + self.E_cr + self.E_mr]
            )
        )
        h2_spatial_context = F.elu(
            self.spatial_context_hidden(out[:, self.E_me + self.E_cr + self.E_mr :])
        )

        mu_x_exp = self.mu_x_exp(h2_mean)
        std_x_exp = self.std_x_exp(h2_mean)

        # if self.use_weights:
        #     with torch.no_grad():
        #         weights = self.get_corr_weights_per_cell(
        #             mu_x_exp.detach()
        #         )  # calculating correlation weights
        # else:
        #     weights = None

        mu_x_corr = self.mu_x_corr(h2_correlations)
        std_x_corr = self.std_x_corr(h2_correlations)

        mu_x_morph = self.mu_x_morph(h2_morphology)
        std_x_morph = self.std_x_morph(h2_morphology)

        mu_x_spatial_context = self.mu_x_spcont(h2_spatial_context)
        std_x_spatial_context = self.std_x_spcont(h2_spatial_context)

        return (
            mu_x_exp,
            std_x_exp,
            mu_x_corr,
            std_x_corr,
            mu_x_morph,
            std_x_morph,
            mu_x_spatial_context,
            std_x_spatial_context,
            # weights,
        )
