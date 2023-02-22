import inspect
import logging
from typing import List, Literal, Optional, Union

import anndata as ad
import numpy as np
import pytorch_lightning as pl
import torch
from anndata import AnnData
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.progress import RichProgressBar
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
        E_cov: int = 10,
        latent_dim: int = 10,
        use_covs: bool = False,
        use_weights: bool = True,
        n_covariates: Optional[Union[None, int]] = None,
        cohort: Optional[Union[None, str]] = None,
        n_hidden: int = 1,
        cofactor: float = 1.0,
        beta_scheme: Optional[Literal["constant", "warmup"]] = "warmup",
        batch_correct: bool = True,
        is_trained_model: bool = False,
        batch_size: Optional[int] = 1234,
        random_seed: Optional[int] = 1234,
        leave_out_view: Optional[
            Union[None, Literal["expression", "correlation", "morphology", "spatial"]]
        ] = None,
        output_dir: str = ".",
        **model_kwargs,
    ):
        # super(hmivaeModel, self).__init__(adata)
        super().__init__()

        self.output_dir = output_dir
        # self.adata = adata
        self.use_covs = use_covs
        self.use_weights = use_weights
        self.leave_out_view = leave_out_view
        self.is_trained_model = is_trained_model
        self.random_seed = random_seed
        self.name = f"{cohort}_rs{random_seed}_nh{n_hidden}_bs{batch_size}_hd{E_me}_ls{latent_dim}"

        if self.use_covs:
            self.keys = []
            for key in adata.obsm.keys():
                # print(key)
                if key not in ["correlations", "morphology", "spatial", "xy"]:
                    self.keys.append(key)

            if n_covariates is None:
                raise ValueError("`n_covariates` cannot be None when `use_covs`==True")
            else:
                n_covariates = n_covariates

            # print("n_keys", len(self.keys))
        else:
            self.keys = None
            if n_covariates is None:
                n_covariates = 0
            else:
                n_covariates = 0
                print("`n_covariates` automatically set to 0 when use_covs == False")

        (
            self.train_batch,
            self.test_batch,
            n_samples,
            self.features_config,
            # self.cov_list,
        ) = self.setup_anndata(
            adata=adata,
            protein_correlations_obsm_key="correlations",
            cell_morphology_obsm_key="morphology",
            continuous_covariate_keys=self.keys,
            cofactor=cofactor,
            image_correct=batch_correct,
            batch_size=batch_size,
            random_seed=random_seed,
        )

        n_covariates += n_samples

        print("n_covs", n_covariates)

        # for batch in self.train_batch:
        #     print('Y', torch.mean(batch[0],1))
        #     print('S', torch.mean(batch[1],1))
        #     print('M', torch.mean(batch[2],1))
        #     print('C', torch.mean(batch[3],1))
        #     print('one-hot', batch[4])
        #     print('covariates', torch.mean(batch[5],1))
        #     break

        # for batch in self.test_batch:
        #     print('Y_test', torch.mean(batch[0],1))
        #     print('S_test', torch.mean(batch[1],1))
        #     print('M_test', torch.mean(batch[2],1))
        #     print('C_test', torch.mean(batch[3],1))
        #     print('one-hot_test', batch[4])
        #     print('covariates_test', torch.mean(batch[5],1))
        #     break

        # print("cov_list", self.cov_list.shape)

        # print("self.adata", self.adata.X)

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
            E_cov=E_cov,
            latent_dim=latent_dim,
            n_covariates=n_covariates,
            n_hidden=n_hidden,
            use_covs=self.use_covs,
            use_weights=self.use_weights,
            beta_scheme=beta_scheme,
            batch_correct=batch_correct,
            leave_out_view=leave_out_view,
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
        self.init_params_ = self._get_init_params(locals())

        logger.info("The model has been initialized")

    def _get_init_params(self, locals):
        """
        Taken from: https://github.com/scverse/scvi-tools/blob/main/scvi/model/base/_base_model.py
        """

        init = self.__init__
        sig = inspect.signature(init)
        parameters = sig.parameters.values()

        init_params = [p.name for p in parameters]
        all_params = {p: locals[p] for p in locals if p in init_params}

        non_var_params = [p.name for p in parameters if p.kind != p.VAR_KEYWORD]
        non_var_params = {k: v for (k, v) in all_params.items() if k in non_var_params}
        var_params = [p.name for p in parameters if p.kind == p.VAR_KEYWORD]
        var_params = {k: v for (k, v) in all_params.items() if k in var_params}

        user_params = {"kwargs": var_params, "non_kwargs": non_var_params}

        return user_params

    def train(
        self,
        max_epochs=15,
        check_val_every_n_epoch=1,
        config=None,
    ):  # misnomer, both train and test/val are here (either rename or separate)

        # with wandb.init(config=config):
        #     config=wandb.config

        pl.seed_everything(self.random_seed)

        early_stopping = EarlyStopping(monitor="recon_lik_test", mode="max", patience=1)

        cb_chkpt = ModelCheckpoint(
            dirpath=f"{self.output_dir}",
            monitor="recon_lik_test",
            mode="max",
            save_top_k=1,
            filename="{epoch}_{step}_{recon_lik_test:.3f}",
        )

        cb_progress = RichProgressBar()
        # wandb.finish()
        # wandb_logger = WandbLogger(log_model="all")

        if self.leave_out_view is None:

            wandb_logger = WandbLogger(
                project="hmivae_hyperparameter_runs",
                name=self.name,
                config=self.features_config,
            )
        else:
            wandb_logger = WandbLogger(
                project="hmivae_ablation", config=self.features_config
            )

        trainer = Trainer(
            max_epochs=max_epochs,
            check_val_every_n_epoch=check_val_every_n_epoch,
            callbacks=[early_stopping, cb_progress, cb_chkpt],
            logger=wandb_logger,
            # overfit_batches=0.01,
            gradient_clip_val=2.0,
            accelerator="auto",
            devices="auto",
            log_every_n_steps=1,
            # limit_train_batches=0.1,
            # limit_val_batches=0.1,
        )

        trainer.fit(self.module, self.train_batch, self.test_batch)

        # wandb.finish()

    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: AnnData,
        protein_correlations_obsm_key: str,
        cell_morphology_obsm_key: str,
        continuous_covariate_keys: Optional[List[str]] = None,
        cofactor: float = 1.0,
        is_trained_model: Optional[bool] = False,
        batch_correct: Optional[bool] = True,
        use_covs: Optional[bool] = True,
    ) -> AnnData:
        """
        Gives the latent representation of each cell.
        """
        if is_trained_model:
            (
                adata_train,
                adata_test,
                data_train,
                data_test,
                n_covariates,
                # cat_list,
                # train_idx,
                # test_idx,
            ) = self.setup_anndata(
                adata,
                protein_correlations_obsm_key,
                cell_morphology_obsm_key,
                continuous_covariate_keys=continuous_covariate_keys,
                cofactor=cofactor,
                is_trained_model=is_trained_model,
                image_correct=batch_correct,
            )

            adata_train.obsm["VAE"] = self.module.inference(
                data_train,
                n_covariates=n_covariates,
                use_covs=use_covs,
                batch_correct=batch_correct,
            )  # idx=train_idx)
            adata_test.obsm["VAE"] = self.module.inference(
                data_test,
                n_covariates=n_covariates,
                use_covs=use_covs,
                batch_correct=batch_correct,
            )  # idx=test_idx)

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
        # cell_spatial_context_obsm_key: str,
        protein_correlations_names_uns_key: Optional[str] = None,
        cell_morphology_names_uns_key: Optional[str] = None,
        image_correct: bool = True,
        batch_size: Optional[int] = 4321,
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

        # print("adata in setup_adata", adata.X)

        if continuous_covariate_keys is not None:
            cat_list = []
            for cat_key in continuous_covariate_keys:
                # print(cat_key)
                # print(f"{cat_key} shape:", adata.obsm[cat_key].shape)
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

        features_ranges = {
            "Train expression min/max": (data_train.Y.min(), data_train.Y.max()),
            "Train correlation min/max": (data_train.S.min(), data_train.S.max()),
            "Train morphology min/max": (data_train.M.min(), data_train.M.max()),
            "Train spatial context min/max": (data_train.C.min(), data_train.C.max()),
            "Test expression min/max": (data_test.Y.min(), data_test.Y.max()),
            "Test correlation min/max": (data_test.S.min(), data_test.S.max()),
            "Test morphology min/max": (data_test.M.min(), data_test.M.max()),
            "Test spatial context min/max": (data_test.C.min(), data_test.C.max()),
        }

        loader_train = DataLoader(
            data_train, batch_size=batch_size, shuffle=True, num_workers=64
        )
        loader_test = DataLoader(
            data_test, batch_size=batch_size, num_workers=64
        )  # shuffle=True)

        if image_correct:
            n_samples = len(samples_train)
            # print("n_samples", n_samples)
            # print("cat_list", cat_list.shape)
            print("one-hot+covs", n_samples + n_cats)
        else:
            n_samples = 0
            print("n_cats", n_cats)

        if is_trained_model:
            return (
                adata_train,
                adata_test,
                data_train,
                data_test,
                n_cats + n_samples,
            )

        else:

            return (
                loader_train,
                loader_test,
                n_samples,
                features_ranges,
            )
