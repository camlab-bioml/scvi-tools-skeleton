# import argparse
# import os

# import time

# import anndata as ad
# import numpy as np
# import scanpy as sc
# import torch

# import wandb
# from pl_vae_classes_and_func_refact import *
# from pytorch_lightning import Trainer

# from scipy.stats.mstats import winsorize
# from ScModeDataloader import ScModeDataloader

# from sklearn.model_selection import train_test_split


# def sparse_numpy_to_torch(adj_mat):
#     """Construct sparse torch tensor
#     Need to do csr -> coo
#     then follow https://stackoverflow.com/questions/50665141/converting-a-scipy-coo-matrix-to-pytorch-sparse-tensor
#     """
#     adj_mat_coo = adj_mat.tocoo()

#     values = adj_mat_coo.data
#     indices = np.vstack((adj_mat_coo.row, adj_mat_coo.col))

#     i = torch.LongTensor(indices)
#     v = torch.FloatTensor(values)
#     shape = adj_mat_coo.shape

#     return torch.sparse_coo_tensor(i, v, shape)


# parser = argparse.ArgumentParser(description="Run emVAE, em2LVAE, dm2LVAE and dmVAE")

# parser.add_argument(
#     "--input_h5ad",
#     type=str,
#     required=True,
#     help="h5ad file that contains mean expression and correlation information for one or more samples",
# )

# parser.add_argument("--lr", type=float, help="Learning rate for VAEs", default=0.001)

# parser.add_argument(
#     "--random_seed",
#     type=int,
#     required=False,
#     help="Random seed for VAE initialization",
#     default=1234,
# )

# parser.add_argument(
#     "--train_ratio",
#     type=float,
#     help="Ratio of the full dataset to be treated as the training set",
#     default=0.75,
# )

# parser.add_argument("--subset_to", type=int, help="Data subset size")

# parser.add_argument(
#     "--winsorize", type=int, help="0 or 1 to denote False or True", default=1
# )

# parser.add_argument("--cofactor", type=float, help="Value for cofactor", default=5.0)

# # parser.add_argument(
# #     "--n_proteins", type=int, required=True, help="Number of proteins in the dataset"
# # )

# parser.add_argument(
#     "--use_weights", type=int, help="0 or 1 to denote False or True", default=0
# )

# parser.add_argument(
#     "--apply_arctanh", type=int, help="0 or 1 to denote False or True", default=0
# )

# parser.add_argument("--cohort", type=str, help="Name of cohort", default="None")

# parser.add_argument("--beta", type=float, help="beta value for B-VAE", default=1.0)

# parser.add_argument("--n_epochs", type=int, help="number of epochs", default=200)

# parser.add_argument(
#     "--apply_KLwarmup",
#     type=int,
#     help="0 or 1 as False or True, to apply a KL-warmup scheme, if not, then BETA is used as given",
#     default=1,
# )

# parser.add_argument(
#     "--regress_out_patient",
#     type=int,
#     help="0 or 1 as False or True, to regress out patient effects, default is False",
#     default=0,
# )

# parser.add_argument(
#     "--KL_limit",
#     type=float,
#     help="Max limit for the coefficient of the KL-Div term",
#     default=0.3,
# )

# parser.add_argument(
#     "--output_dir", type=str, help="Directory to store the outputs", default="."
# )

# args = parser.parse_args()


# adata = sc.read_h5ad(args.input_h5ad)

# COFACTOR = args.cofactor

# RANDOM_SEED = args.random_seed

# N_EPOCHS = args.n_epochs
# N_HIDDEN = 2
# HIDDEN_LAYER_SIZE_Eme = 8
# HIDDEN_LAYER_SIZE_Ecr = 8
# HIDDEN_LAYER_SIZE_Emr = 8
# N_SPATIAL_CONTEXT = (
#     HIDDEN_LAYER_SIZE_Eme + HIDDEN_LAYER_SIZE_Ecr + HIDDEN_LAYER_SIZE_Emr
# )
# HIDDEN_LAYER_SIZE_Esc = 8  # keeping this consistent with the previous Basel analysis

# LATENT_DIM = 10
# BATCH_SIZE = 256
# CELLS_CUTOFF = 500

# N_TOTAL_CELLS = adata.shape[0]
# N_PROTEINS = adata.shape[1]
# N_CORRELATIONS = len(adata.uns["names_correlations"])
# N_MORPHOLOGY = len(adata.uns["names_morphology"])

# N_TOTAL_FEATURES = N_PROTEINS + N_CORRELATIONS + N_MORPHOLOGY

# BETA = args.beta  ## beta for beta-vae

# TRAIN_PROP = args.train_ratio  # set the training set ratio

# lr = args.lr  # set the learning rate

# # log_py = {}
# # elbo_losses = {}


# if args.output_dir is not None:
#     output_dir = args.output_dir
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
# else:
#     output_dir = "."


# # Set up the data
# np.random.seed(RANDOM_SEED)

# if args.subset_to is not None:
#     print("Subsetting samples")
#     samples = adata.obs.Sample_name.unique().to_list()
#     inds = np.random.choice(samples, args.subset_to)
#     adata = adata[adata.obs.Sample_name.isin(inds)]
# else:
#     adata = adata


# # adata.obs = adata.obs.reset_index()
# # adata.obs.columns = ["index", "Sample_name", "cell_id"]


# if adata.X.shape[0] > 705000:
#     sample_drop_lst = []
#     for sample in adata.obs[
#         "Sample_name"
#     ].unique():  # if sample has less than 500 cells, drop it
#         if (
#             adata.obs.query("Sample_name==@sample").shape[0] < CELLS_CUTOFF
#         ):  # true for <235 samples out of all samples
#             sample_drop_lst.append(sample)

#     adata_sub = adata.copy()[
#         ~adata.obs.Sample_name.isin(sample_drop_lst), :
#     ]  # select all rows except those that belong to samples w cells < CELLS_CUTOFF

#     adata_sub.obs = adata_sub.obs.reset_index()
#     if "level_0" in adata_sub.obs.columns:
#         adata_sub.obs = adata_sub.obs.drop(columns=["level_0"])

# else:
#     adata_sub = adata


# print("Preprocessing data views")

# if args.cofactor is not None:
#     adata_sub.X = np.arcsinh(adata_sub.X / COFACTOR)


# if args.winsorize == 1:
#     for i in range(N_PROTEINS):
#         adata_sub.X[:, i] = winsorize(adata_sub.X[:, i], limits=[0, 0.01])
#     for i in range(N_MORPHOLOGY):
#         adata_sub.obsm["morphology"][:, i] = winsorize(
#             adata_sub.obsm["morphology"][:, i], limits=[0, 0.01]
#         )

# if args.apply_arctanh == 1:
#     adata_sub.obsm["correlations"] = np.arctanh(adata_sub.obsm["correlations"])


# adata_sub.obs["Sample_name"] = adata_sub.obs["Sample_name"].astype(
#     str
# )  # have to do this otherwise it will contain the ones that were removed

# if args.regress_out_patient:
#     print("Regressing out patient effect")
#     sc.pp.regress_out(adata_sub, "Sample_name")

# samples_list = adata_sub.obs["Sample_name"].unique().tolist()  # samples in the adata


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# train_size = int(np.floor(len(samples_list) * TRAIN_PROP))
# test_size = len(samples_list) - train_size

# # separate images/samples as train or test *only* (this is different from before, when we separated cells into train/test)

# print("Setting up train and test data")

# samples_train, samples_test = train_test_split(
#     samples_list, train_size=TRAIN_PROP, random_state=RANDOM_SEED
# )

# adata_train = adata_sub.copy()[adata_sub.obs["Sample_name"].isin(samples_train), :]
# adata_test = adata_sub.copy()[adata_sub.obs["Sample_name"].isin(samples_test), :]


# data_train = ScModeDataloader(adata_train)
# data_test = ScModeDataloader(adata_test, data_train.scalers)

# loader_train = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
# loader_test = DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=True)


# model = hmiVAE(
#     N_PROTEINS,
#     N_CORRELATIONS,
#     N_MORPHOLOGY,
#     N_SPATIAL_CONTEXT,
#     HIDDEN_LAYER_SIZE_Eme,
#     HIDDEN_LAYER_SIZE_Ecr,
#     HIDDEN_LAYER_SIZE_Emr,
#     HIDDEN_LAYER_SIZE_Esc,
#     LATENT_DIM,
# )

# trainer = Trainer()

# wandb.init(
#     project="vae_new_morphs",
#     entity="sayub",
#     config={
#         "learning_rate": lr,
#         "epochs": N_EPOCHS,
#         "batch_size": BATCH_SIZE,
#         "use_weights": int(corr_weights),
#         "use_arctanh": int(args.apply_arctanh),
#         "n_cells": len(data_train),
#         "method": "emVAE",
#         "cohort": args.cohort,
#         "BETA": BETA,
#         "cofactor": COFACTOR,
#         "regress_out_patient": args.regress_out_patient,
#         "apply_KL_warmup": args.apply_KLwarmup,
#         "KL_max_limit": args.KL_limit,
#     },
# )


# # start_time_em = time.time()


# # ## Reconstruction weights
# # r = np.array([1., 1., 1., 0.])

# # for n in range(N_EPOCHS):
# #     if args.apply_KLwarmup:
# #         if n>5:
# #             new_beta = BETA + 0.05
# #             BETA = min(new_beta, args.KL_limit)
# #     else:
# #         BETA = BETA

# #     if n > 5:
# #         spcont_r = r[3]+0.1
# #         r[3] = min(spcont_r, 1.0)
# #         if n < 30:
# #             print(r)
# #         #r = np.array([1., 1., 1., 1.])

# #     optimizer_em.zero_grad()

# #     train_losses_em = 0
# #     num_batches_em = 0

# for num, batch in enumerate(loader_train):
#     data = {
#         "Y": batch[0],
#         "S": batch[1],
#         "M": batch[2],
#         "A": spatial_context[batch[-1], :],
#     }

#     em_train_loss = train_test_run(
#         data=data,
#         spatial_context=data["A"],
#         method="EM",
#         method_enc=enc_em,
#         method_dec=dec_em,
#         n_proteins=N_PROTEINS,
#         latent_dim=LATENT_DIM,
#         corr_weights=corr_weights,
#         recon_weights=r,
#         beta=BETA,
#     )

#     ## Update gradients and weights
#     em_train_loss[0].backward()

#     torch.nn.utils.clip_grad_norm_(parameters, 2.0)

#     optimizer_em.step()

#     train_losses_em += em_train_loss[0].detach().item()
#     num_batches_em += 1


# # Update old mean embeddings (once per epoch)
# with torch.no_grad():
#     mu_z_old, _, z1 = enc_em(data_train.Y, data_train.S, data_train.M, z1)  # mu_z_old)

# wandb.log(
#     {
#         "train_neg_elbo": train_losses_em / num_batches_em,
#         "kl_div": em_train_loss[1],
#         "recon_lik": em_train_loss[2],
#         "recon_lik_me": em_train_loss[3],
#         "recon_lik_corr": em_train_loss[4],
#         "recon_lik_mor": em_train_loss[5],
#         "recon_lik_spcont": em_train_loss[6],
#         "mu_z_max": em_train_loss[8],
#         "log_std_z_max": em_train_loss[9],
#         "mu_z_min": em_train_loss[10],
#         "log_std_z_min": em_train_loss[11],
#         "mu_x_exp_hat_max": em_train_loss[12],
#         "log_std_x_exp_hat_max": em_train_loss[13],
#         "mu_x_exp_hat_min": em_train_loss[14],
#         "log_std_x_exp_hat_min": em_train_loss[15],
#         "mu_x_corr_hat_max": em_train_loss[16],
#         "log_std_x_corr_hat_max": em_train_loss[17],
#         "mu_x_corr_hat_min": em_train_loss[18],
#         "log_std_x_corr_hat_min": em_train_loss[19],
#         "mu_x_morph_hat_max": em_train_loss[20],
#         "log_std_x_morph_hat_max": em_train_loss[21],
#         "mu_x_morph_hat_min": em_train_loss[22],
#         "log_std_x_morph_hat_min": em_train_loss[23],
#         "mu_x_spcont_hat_max": em_train_loss[24],
#         "log_std_x_spcont_hat_max": em_train_loss[25],
#         "mu_x_spcont_hat_min": em_train_loss[26],
#         "log_std_x_spcont_hat_min": em_train_loss[27],
#     }
# )

# # Now compute test metrics

# with torch.no_grad():
#     spatial_context_test = torch.smm(
#         adj_mat_test_tensor, z1_test  # mu_z_old_test
#     ).to_dense()

#     test_data = {
#         "Y": data_test.Y,
#         "S": data_test.S,
#         "M": data_test.M,
#         "A": spatial_context_test,
#     }

#     em_test_loss = train_test_run(
#         data=test_data,
#         spatial_context=test_data["A"],
#         method="EM",
#         method_enc=enc_em,
#         method_dec=dec_em,
#         n_proteins=N_PROTEINS,
#         latent_dim=LATENT_DIM,
#         corr_weights=corr_weights,
#         recon_weights=r,
#         beta=BETA,
#     )

#     # mu_z_old_test = em_test_loss[7]
#     z1_test = em_test_loss[7]

#     wandb.log(
#         {
#             "test_neg_elbo": em_test_loss[0],
#             "test_kl_div": em_test_loss[1],
#             "test_recon_lik": em_test_loss[2],
#             "test_recon_lik_me": em_test_loss[3],
#             "test_recon_lik_corr": em_test_loss[4],
#             "test_recon_lik_mor": em_test_loss[5],
#             "test_recon_lik_spcont": em_test_loss[6],
#         }
#     )

# stop_time_em = time.time()
# wandb.finish()
# print("emVAE done training")

# # adata_train.obsm['emVAE_final_spatial_context'] = mu_z_old_train_em.detach().numpy()
# # adata_test.obsm['emVAE_final_spatial_context'] = mu_z_old_test_em.detach().numpy()

# # eval_em_spatial_context= torch.smm(
# #             adj_mat_test_tensor, mu_z_old_test_em
# #         ).to_dense()

# with torch.no_grad():
#     mu_z, _, z1 = enc_em(data_train.Y, data_train.S, data_train.M, z1)
#     mu_z_test, _, z1_test = enc_em(data_test.Y, data_test.S, data_test.M, z1_test)

# adata_train.obsm["VAE"] = mu_z.numpy()
# adata_test.obsm["VAE"] = mu_z_test.numpy()

# adata_train.obsm["spatial_context"] = z1.numpy()
# adata_test.obsm["spatial_context"] = z1_test.numpy()

# adata_train.write(os.path.join(output_dir, "adata_train.h5ad"))
# adata_test.write(os.path.join(output_dir, "adata_test.h5ad"))

# torch.save(enc_em.state_dict(), os.path.join(output_dir, "emVAE_encoder.pt"))
# torch.save(dec_em.state_dict(), os.path.join(output_dir, "emVAE_decoder.pt"))


# print("All done!")
