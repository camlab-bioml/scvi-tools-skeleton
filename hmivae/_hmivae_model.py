import logging
from typing import List, Optional

import anndata as ad
import numpy as np
import pytorch_lightning as pl
import torch
from anndata import AnnData
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# import hmivae
import hmivae._hmivae_module as module
import hmivae.ScModeDataloader as ScModeDataloader

logger = logging.getLogger(__name__)


class hmivaeModel(pl.LightningModule):
    """
    Skeleton for an scvi-tools model.

    Please use this skeleton to create new models.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~mypackage.MyModel.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    **model_kwargs
        Keyword args for :class:`~mypackage.MyModule`
    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> mypackage.MyModel.setup_anndata(adata, batch_key="batch")
    >>> vae = mypackage.MyModel(adata)
    >>> vae.train()
    >>> adata.obsm["X_mymodel"] = vae.get_latent_representation()
    """

    def __init__(
        self,
        adata: AnnData,
        input_exp_dim: int,
        input_corr_dim: int,
        input_morph_dim: int,
        input_spcont_dim: int,
        E_me: int = 32,
        E_cr: int = 32,
        E_mr: int = 32,
        E_sc: int = 32,
        latent_dim: int = 10,
        use_covs: bool = False,
        n_hidden: int = 1,
        cofactor: float = 1.0,
        batch_correct: bool = True,
        **model_kwargs,
    ):
        # super(hmivaeModel, self).__init__(adata)
        super().__init__()

        self.adata = adata
        self.use_covs = use_covs

        if self.use_covs:
            self.keys = []
            for key in adata.obsm.keys():

                if key not in ["correlations", "morphology", "xy"]:
                    self.keys.append(key)

        else:
            self.keys = None

        (
            self.train_batch,
            self.test_batch,
            self.n_covariates,
            # self.cov_list,
        ) = self.setup_anndata(
            adata=self.adata,
            protein_correlations_obsm_key="correlations",
            cell_morphology_obsm_key="morphology",
            continuous_covariate_keys=self.keys,
            cofactor=cofactor,
            image_correct=batch_correct,
        )

        # self.summary_stats provides information about anndata dimensions and other tensor info
        self.module = module.hmiVAE(
            input_exp_dim=input_exp_dim,
            input_corr_dim=input_corr_dim,
            input_morph_dim=input_morph_dim,
            input_spcont_dim=input_spcont_dim,
            E_me=E_me,
            E_cr=E_cr,
            E_mr=E_mr,
            E_sc=E_sc,
            latent_dim=latent_dim,
            n_covariates=self.n_covariates,
            n_hidden=n_hidden,
            use_covs=self.use_covs,
            # cat_list=self.cov_list,
            batch_correct=batch_correct,
            **model_kwargs,
        )
        self._model_summary_string = (
            "hmiVAE model with the following parameters: \n n_latent:{}, "
            "n_protein_expression:{}, n_correlation:{}, n_morphology:{}, n_spatial_context:{}, "
            "use_covariates:{} "
        ).format(
            latent_dim,
            input_exp_dim,
            input_corr_dim,
            input_morph_dim,
            input_spcont_dim,
            use_covs,
        )
        # necessary line to get params that will be used for saving/loading
        # self.init_params_ = self._get_init_params(locals())

        logger.info("The model has been initialized")

    def train(
        self,
        max_epochs=100,
        check_val_every_n_epoch=5,
    ):  # misnomer, both train and test are here (either rename or separate)

        early_stopping = EarlyStopping(monitor="test_loss", mode="min", patience=2)

        wandb_logger = WandbLogger(project="hmiVAE_init_trial_runs")

        trainer = Trainer(
            max_epochs=max_epochs,
            check_val_every_n_epoch=check_val_every_n_epoch,
            callbacks=[early_stopping],
            logger=wandb_logger,
            # gradient_clip_val=2.0,
        )

        trainer.fit(self.module, self.train_batch, self.test_batch)

    @torch.no_grad()
    def get_latent_representation(
        self,
        protein_correlations_obsm_key: str,
        cell_morphology_obsm_key: str,
        continuous_covariate_keys: Optional[List[str]] = None,  # default is self.keys
        cofactor: float = 1.0,
        is_trained_model: Optional[bool] = False,
        batch_correct: Optional[bool] = True,
    ) -> AnnData:
        """
        Gives the latent representation of each cell.
        """
        if is_trained_model:
            (adata_train, adata_test, data_train, data_test,) = self.setup_anndata(
                self.adata,
                protein_correlations_obsm_key,
                cell_morphology_obsm_key,
                continuous_covariate_keys=self.keys,
                cofactor=cofactor,
                is_trained_model=is_trained_model,
                image_correct=batch_correct,
            )

            adata_train.obsm["VAE"] = self.module.inference(data_train)
            adata_test.obsm["VAE"] = self.module.inference(data_test)

            return ad.concat([adata_train, adata_test], uns_merge="first")
        else:
            raise Exception(
                "No latent representation to produce! Model is not trained!"
            )

    # @setup_anndata_dsp.dedent
    @staticmethod
    def setup_anndata(
        # self,
        adata: AnnData,
        protein_correlations_obsm_key: str,
        cell_morphology_obsm_key: str,
        protein_correlations_names_uns_key: Optional[str] = None,
        cell_morphology_names_uns_key: Optional[str] = None,
        image_correct: bool = True,
        batch_size: Optional[int] = 128,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        layer: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[
            List[str]
        ] = None,  # obsm keys for other categories
        cofactor: float = 1.0,
        train_prop: Optional[float] = 0.75,
        apply_winsorize: Optional[bool] = True,
        arctanh_corrs: Optional[bool] = False,
        is_trained_model: Optional[bool] = False,
        random_seed: Optional[int] = 1234,
        copy: bool = False,
    ) -> Optional[AnnData]:
        """
        %(summary)s.
        Takes in an AnnData object and returns the train and test loaders.
        Parameters
        ----------
        %(param_adata)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_layer)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        %(param_copy)s

        Returns
        -------
        %(returns)s
        """
        N_PROTEINS = adata.shape[1]
        N_MORPHOLOGY = len(adata.uns["names_morphology"])

        if continuous_covariate_keys is not None:
            cat_list = []
            for cat_key in continuous_covariate_keys:
                category = adata.obsm[cat_key]
                cat_list.append(category)
            cat_list = np.concatenate(cat_list, 1)
            n_cats = cat_list.shape[1]
            # if apply_winsorize:
            #     for i in range(cat_list.shape[1]):
            #         cat_list[:, i] = winsorize(cat_list[:, i], limits=[0, 0.01])

            adata.obsm["background_covs"] = cat_list
        else:
            n_cats = 0

        adata.X = np.arcsinh(adata.X / cofactor)

        if apply_winsorize:
            for i in range(N_PROTEINS):
                adata.X[:, i] = winsorize(adata.X[:, i], limits=[0, 0.01])
            for i in range(N_MORPHOLOGY):
                adata.obsm[cell_morphology_obsm_key][:, i] = winsorize(
                    adata.obsm[cell_morphology_obsm_key][:, i], limits=[0, 0.01]
                )

        if arctanh_corrs:
            adata.obsm[protein_correlations_obsm_key] = np.arctanh(
                adata.obsm[protein_correlations_obsm_key]
            )

        samples_list = (
            adata.obs["Sample_name"].unique().tolist()
        )  # samples in the adata

        samples_train, samples_test = train_test_split(
            samples_list, train_size=train_prop, random_state=random_seed
        )

        adata_train = adata.copy()[adata.obs["Sample_name"].isin(samples_train), :]

        adata_test = adata.copy()[adata.obs["Sample_name"].isin(samples_test), :]

        data_train = ScModeDataloader.ScModeDataloader(adata_train)
        data_test = ScModeDataloader.ScModeDataloader(adata_test, data_train.scalers)

        loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
        loader_test = DataLoader(data_test, batch_size=batch_size)

        if image_correct:
            n_samples = len(samples_train)
        else:
            n_samples = 0

        if is_trained_model:

            return (
                adata_train,
                adata_test,
                data_train,
                data_test,
            )

        else:

            return (
                loader_train,
                loader_test,
                n_samples + n_cats,
            )
