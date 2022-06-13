import logging
from typing import List, Optional

import numpy as np
import pytorch_lightning as pl
from _hmivae_module import hmiVAE
from anndata import AnnData
from pytorch_lightning.trainer import Trainer
from scipy.stats.mstats import winsorize
from ScModeDataloader import ScModeDataloader
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# from scvi.data import setup_anndata
# from scvi.model._utils import _init_library_size
# from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
# from scvi.utils import setup_anndata_dsp


logger = logging.getLogger(__name__)


# class hmivaeModel(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
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
        n_hidden: int = 1,
        **model_kwargs,
    ):
        # super(hmivaeModel, self).__init__(adata)
        super().__init__()

        # library_log_means, library_log_vars = _init_library_size(
        #     adata, self.summary_stats["n_batch"]
        # )

        self.train_batch, self.test_batch = self.setup_anndata(
            adata=adata,
            protein_correlations_obsm_key="correlations",
            cell_morphology_obsm_key="morphology",
        )

        # self.summary_stats provides information about anndata dimensions and other tensor info

        self.module = hmiVAE(
            input_exp_dim=input_exp_dim,
            input_corr_dim=input_corr_dim,
            input_morph_dim=input_morph_dim,
            input_spcont_dim=input_spcont_dim,
            E_me=E_me,
            E_cr=E_cr,
            E_mr=E_mr,
            E_sc=E_sc,
            latent_dim=latent_dim,
            n_hidden=n_hidden,
            **model_kwargs,
        )
        self._model_summary_string = (
            "hmiVAE model with the following parameters: \nn_latent:{}"
            "n_protein_expression:{}, n_correlation:{}, n_morphology:{}, n_spatial_context:{}"
        ).format(
            latent_dim,
            input_exp_dim,
            input_corr_dim,
            input_morph_dim,
            input_spcont_dim,
        )
        # necessary line to get params that will be used for saving/loading
        self.init_params_ = self._get_init_params(locals())

        logger.info("The model has been initialized")

    def train(self):

        trainer = Trainer()

        trainer.fit(self.module, self.train_batch)  # training, add wandb
        trainer.test(dataloaders=self.test_batch)  # test, add wandb

        return trainer()

    # @setup_anndata_dsp.dedent
    @staticmethod
    def setup_anndata(
        # self,
        adata: AnnData,
        protein_correlations_obsm_key: str,
        cell_morphology_obsm_key: str,
        # cell_spatial_context_obsm_key: str,
        protein_correlations_names_uns_key: Optional[str] = None,
        cell_morphology_names_uns_key: Optional[str] = None,
        batch_size: Optional[int] = 128,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        layer: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        cofactor: float = 1.0,
        train_prop: Optional[float] = 0.75,
        apply_winsorize: Optional[bool] = True,
        arctanh_corrs: Optional[bool] = False,
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
        # N_TOTAL_CELLS = adata.shape[0]
        N_PROTEINS = adata.shape[1]
        # N_CORRELATIONS = len(adata.uns["names_correlations"])
        N_MORPHOLOGY = len(adata.uns["names_morphology"])

        # N_TOTAL_FEATURES = N_PROTEINS + N_CORRELATIONS + N_MORPHOLOGY
        # if cofactor is not None:
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

        # train_size = int(np.floor(len(samples_list) * train_prop))
        # test_size = len(samples_list) - train_size

        samples_train, samples_test = train_test_split(
            samples_list, train_size=train_prop, random_state=random_seed
        )

        adata_train = adata.copy()[adata.obs["Sample_name"].isin(samples_train), :]
        adata_test = adata.copy()[adata.obs["Sample_name"].isin(samples_test), :]

        data_train = ScModeDataloader(adata_train)
        data_test = ScModeDataloader(adata_test, data_train.scalers)

        loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
        loader_test = DataLoader(data_test, batch_size=batch_size, shuffle=True)

        return loader_train, loader_test
