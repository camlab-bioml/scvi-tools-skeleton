## run with hmivae

import argparse
import os
import time
from collections import OrderedDict

import numpy as np
import pandas as pd

# import phenograph
import scanpy as sc
import squidpy as sq
import torch
import wandb
from rich.progress import (  # track,
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
from statsmodels.api import OLS, add_constant

# import hmivae
from hmivae._hmivae_model import hmivaeModel


def create_cluster_dummy(adata, cluster_col, cluster):
    # n_clusters = len(adata.obs[cluster_col].unique().tolist())
    x = np.zeros([adata.X.shape[0], 1])

    for cell in adata.obs.index:
        # cell_cluster = int(adata.obs[cluster_col][cell])
        # print(type(cell), type(cluster))

        if adata.obs[cluster_col][int(cell)] == cluster:
            x[int(cell)] = 1

    return x


def get_feature_matrix(adata, scale_values=False, cofactor=1, weights=True):

    correlations = adata.obsm["correlations"]
    if weights:
        correlations = np.multiply(
            correlations, adata.obsm["weights"]
        )  # multiply weights with correlations

    if scale_values:
        morphology = adata.obsm["morphology"]
        for i in range(adata.obsm["morphology"].shape[1]):
            morphology[:, i] = winsorize(
                adata.obsm["morphology"][:, i], limits=[0, 0.01]
            )

        expression = np.arcsinh(adata.X / cofactor)
        for j in range(adata.X.shape[1]):
            expression[:, j] = winsorize(expression[:, j], limits=[0, 0.01])
    else:
        morphology = adata.obsm["morphology"]
        expression = adata.X

    y = StandardScaler().fit_transform(
        np.concatenate([expression, correlations, morphology], axis=1)
    )

    var_names = np.concatenate(
        [
            adata.var_names,
            adata.uns["names_correlations"],
            adata.uns["names_morphology"],
        ]
    )

    return y, var_names


def rank_features_in_groups(adata, group_col, scale_values=False, cofactor=1):

    progress = Progress(
        TextColumn(f"[progress.description]Ranking features in {group_col} groups"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )
    ranked_features_in_groups = {}
    dfs = []
    # create the feature matrix for entire adata
    y, var_names = get_feature_matrix(
        adata, scale_values=scale_values, cofactor=cofactor
    )
    y = add_constant(y)  # add intercept

    with progress:

        for group in progress.track(adata.obs[group_col].unique()):
            ranked_features_in_groups[str(group)] = {}
            x = create_cluster_dummy(adata, group_col, group)
            mod = OLS(x, y)
            res = mod.fit()

            df_values = pd.DataFrame(
                res.tvalues[1:],  # remove the intercept value
                index=var_names,
                columns=[f"t_value_{group}"],
            ).sort_values(by=f"t_value_{group}", ascending=False)

            ranked_features_in_groups[str(group)]["names"] = df_values.index.to_list()
            ranked_features_in_groups[str(group)]["t_values"] = df_values[
                f"t_value_{group}"
            ].to_list()

            # print('df index:', df_values.index.tolist())

            dfs.append(df_values)

    fc_df = pd.concat(
        dfs, axis=1
    ).sort_index()  # index is sorted as alphabetical! (order with original var_names is NOT maintained!)

    fc_df.index = fc_df.index.map(str)
    fc_df.columns = fc_df.columns.map(str)

    adata.uns[f"{group_col}_ranked_features_in_groups"] = ranked_features_in_groups
    adata.uns[f"{group_col}_feature_scores"] = fc_df

    # return adata


def top_common_features(df, top_n_features=10):

    sets_list = []

    for i in df.columns:
        abs_sorted_col = df[i].map(abs).sort_values(ascending=False)
        for j in abs_sorted_col.index.to_list()[0:top_n_features]:
            sets_list.append(j)

    common_features = list(set(sets_list))

    common_feat_df = df.loc[common_features]

    return common_feat_df


parser = argparse.ArgumentParser(description="Run hmiVAE")

parser.add_argument(
    "--adata", type=str, required=True, help="AnnData file with all the inputs"
)

parser.add_argument(
    "--include_all_views",
    type=int,
    help="Run model using all views",
    default=1,
    choices=[0, 1],
)

parser.add_argument(
    "--remove_view",
    type=str,
    help="Name of view to leave out. One of ['expression', 'correlation', 'morphology', 'spatial']. Must be given when `include_all_views` is False",
    default=None,
    choices=["expression", "correlation", "morphology", "spatial"],
)

parser.add_argument(
    "--use_covs",
    type=bool,
    help="True/False for using background covariates",
    default=True,
)

parser.add_argument(
    "--use_weights",
    type=bool,
    help="True/False for using correlation weights",
    default=True,
)

parser.add_argument(
    "--batch_correct",
    type=bool,
    help="True/False for using one-hot encoding for batch correction",
    default=True,
)

parser.add_argument(
    "--batch_size",
    type=int,
    help="Batch size for train/test data, default=1234",
    default=1234,
)

parser.add_argument(
    "--hidden_dim_size",
    type=int,
    help="Size for view-specific hidden layers",
    default=32,
)

parser.add_argument(
    "--latent_dim",
    type=int,
    help="Size for the final latent representation layer",
    default=10,
)

parser.add_argument(
    "--n_hidden",
    type=int,
    help="Number of hidden layers",
    default=1,
)

parser.add_argument(
    "--beta_scheme",
    type=str,
    help="Scheme to use for beta vae",
    default="warmup",
    choices=["constant", "warmup"],
)

parser.add_argument(
    "--use_linear_decoder",
    type=bool,
    help="For using a linear decoder: True or False",
    default=False,
)

parser.add_argument(
    "--cofactor", type=float, help="Cofactor for arcsinh transformation", default=1.0
)

parser.add_argument(
    "--random_seed",
    type=int,
    help="Random seed for weights initialization",
    default=1234,
)

parser.add_argument("--cohort", type=str, help="Cohort name", default="cohort")

parser.add_argument(
    "--output_dir", type=str, help="Directory to store the outputs", default="."
)

args = parser.parse_args()

log_file = open(
    os.path.join(
        args.output_dir,
        f"{args.cohort}_nhidden{args.n_hidden}_hiddendim{args.hidden_dim_size}_latentdim{args.latent_dim}_betascheme{args.beta_scheme}_randomseed{args.random_seed}_run_log.txt",
    ),
    "w+",
)

raw_adata = sc.read_h5ad(args.adata)

# print("connections", adata.obsp["connectivities"])
# print("raw adata X min,max", raw_adata.X.max(), raw_adata.X.min())
# print("raw adata corrs min,max", raw_adata.obsm['correlations'].max(), raw_adata.obsm['correlations'].min())
# print("raw adata morph min,max", raw_adata.obsm['morphology'].max(), raw_adata.obsm['morphology'].min())

L = [
    f"raw adata X, max: {raw_adata.X.max()}, min: {raw_adata.X.min()} \n",
    f"raw adata correlations, max: {raw_adata.obsm['correlations'].max()}, min: {raw_adata.obsm['correlations'].min()} \n",
    f"raw adata morphology, max: {raw_adata.obsm['morphology'].max()}, min: {raw_adata.obsm['morphology'].min()} \n",
]

log_file.writelines(L)
n_total_features = (
    raw_adata.X.shape[1]
    + raw_adata.obsm["correlations"].shape[1]
    + raw_adata.obsm["morphology"].shape[1]
)

log_file.write(f"Total number of features:{n_total_features} \n")
log_file.write(f"Total number of cells:{raw_adata.X.shape[0]} \n")

print("Set up the model")

start = time.time()


E_me, E_cr, E_mr, E_sc = [
    args.hidden_dim_size,
    args.hidden_dim_size,
    args.hidden_dim_size,
    args.hidden_dim_size,
]
input_exp_dim, input_corr_dim, input_morph_dim, input_spcont_dim = [
    raw_adata.shape[1],
    raw_adata.obsm["correlations"].shape[1],
    raw_adata.obsm["morphology"].shape[1],
    n_total_features,
]
keys = []
if args.use_covs:
    cat_list = []

    for key in raw_adata.obsm.keys():
        # print(key)
        if key not in ["correlations", "morphology", "spatial", "xy"]:
            keys.append(key)
    for cat_key in keys:
        # print(cat_key)
        # print(f"{cat_key} shape:", adata.obsm[cat_key].shape)
        category = raw_adata.obsm[cat_key]
        cat_list.append(category)
    cat_list = np.concatenate(cat_list, 1)
    n_covariates = cat_list.shape[1]
    E_cov = args.hidden_dim_size
else:
    n_covariates = 0
    E_cov = 0

model = hmivaeModel(
    adata=raw_adata,
    input_exp_dim=input_exp_dim,
    input_corr_dim=input_corr_dim,
    input_morph_dim=input_morph_dim,
    input_spcont_dim=input_spcont_dim,
    E_me=E_me,
    E_cr=E_cr,
    E_mr=E_mr,
    E_sc=E_sc,
    E_cov=E_cov,
    latent_dim=args.latent_dim,
    cofactor=args.cofactor,
    use_covs=args.use_covs,
    cohort=args.cohort,
    use_weights=args.use_weights,
    beta_scheme=args.beta_scheme,
    linear_decoder=args.use_linear_decoder,
    n_covariates=n_covariates,
    batch_correct=args.batch_correct,
    batch_size=args.batch_size,
    random_seed=args.random_seed,
    n_hidden=args.n_hidden,
    leave_out_view=args.remove_view,
    output_dir=args.output_dir,
)


print("Start training")


model.train()

wandb.finish()

model_checkpoint = [
    i for i in os.listdir(args.output_dir) if ".ckpt" in i
]  # should only be 1 -- saved best model

print("model_checkpoint", model_checkpoint)

load_chkpt = torch.load(os.path.join(args.output_dir, model_checkpoint[0]))

state_dict = load_chkpt["state_dict"]
# print(state_dict)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    # print("key", k)
    if "weight" or "bias" in k:
        # print("changing", k)
        name = "module." + k  # add `module.`
        # print("new name", name)
    else:
        # print("staying same", k)
        name = k
    new_state_dict[name] = v
# load params

load_chkpt["state_dict"] = new_state_dict

# torch.save(os.path.join(args.output_dir, model_checkpoint[0]))

model = hmivaeModel(
    adata=raw_adata,
    input_exp_dim=input_exp_dim,
    input_corr_dim=input_corr_dim,
    input_morph_dim=input_morph_dim,
    input_spcont_dim=input_spcont_dim,
    E_me=E_me,
    E_cr=E_cr,
    E_mr=E_mr,
    E_sc=E_sc,
    E_cov=E_cov,
    latent_dim=args.latent_dim,
    cofactor=args.cofactor,
    use_covs=args.use_covs,
    use_weights=args.use_weights,
    beta_scheme=args.beta_scheme,
    linear_decoder=args.use_linear_decoder,
    n_covariates=n_covariates,
    batch_correct=args.batch_correct,
    batch_size=args.batch_size,
    random_seed=args.random_seed,
    n_hidden=args.n_hidden,
    leave_out_view=args.remove_view,
    output_dir=args.output_dir,
)
model.load_state_dict(new_state_dict, strict=False)


# model.load_from_checkpoint(os.path.join(args.output_dir, model_checkpoint[0]), adata=raw_adata)

print("Best model loaded from checkpoint")

stop = time.time()

log_file.write(f"All training done in {(stop-start)/60} minutes \n")

starta = time.time()

adata = model.get_latent_representation(  # use the best model to get the latent representations
    adata=raw_adata,
    protein_correlations_obsm_key="correlations",
    cell_morphology_obsm_key="morphology",
    continuous_covariate_keys=keys,
    is_trained_model=True,
    batch_correct=args.batch_correct,
)

print("Doing cluster and neighbourhood enrichment analysis")

print("Clustering using integrated space")

sc.pp.neighbors(
    adata, n_neighbors=100, use_rep="VAE", key_added="vae"
)  # 100 nearest neighbours, will be used in downstream tests -- keep with PG

# sc.pp.neighbors(adata, n_neighbors=100, use_rep="VAE", key_added="vae_100")

sc.tl.leiden(adata, neighbors_key="vae")

print("Clustering using specific views")

sc.pp.neighbors(
    adata, n_neighbors=100, use_rep="expression_embedding", key_added="expression"
)  # 100 nearest neighbours, will be used in downstream tests -- keep with PG

# sc.pp.neighbors(adata, n_neighbors=100, use_rep="VAE", key_added="vae_100")

sc.tl.leiden(
    adata,
    neighbors_key="expression",
    key_added="expression_leiden",
    random_state=args.random_seed,
)

sc.pp.neighbors(
    adata, n_neighbors=100, use_rep="correlation_embedding", key_added="correlation"
)

sc.tl.leiden(
    adata,
    neighbors_key="correlation",
    key_added="correlation_leiden",
    random_state=args.random_seed,
)

sc.pp.neighbors(
    adata, n_neighbors=100, use_rep="morphology_embedding", key_added="morphology"
)

sc.tl.leiden(
    adata,
    neighbors_key="morphology",
    key_added="morphology_leiden",
    random_state=args.random_seed,
)

sc.pp.neighbors(
    adata,
    n_neighbors=100,
    use_rep="spatial_context_embedding",
    key_added="spatial_context",
)

sc.tl.leiden(
    adata,
    neighbors_key="spatial_context",
    key_added="spatial_context_leiden",
    random_state=args.random_seed,
)


sc.tl.umap(adata, neighbors_key="vae", random_state=args.random_seed)


# print("Ranking features across cluster")
# start1 = time.time()
# if "cell_id" not in adata.obs.columns:
#     print("Reset index to get cell_id column")
#     adata.obs = adata.obs.reset_index()


# # ranked_dict, fc_df =
# rank_features_in_groups(
#     adata,
#     "leiden",
#     scale_values=False,
#     cofactor=args.cofactor,
# )  # no scaling required because using adata_train and test which have already been normalized and winsorized -- StandardScaler still applied
# fc_df = adata.uns["leiden_feature_scores"]

# #ranked_dict_pg, fc_df_pg =
# rank_features_in_groups(
#     adata, "PhenoGraph_clusters", scale_values=True, cofactor=args.cofactor,
# )
# fc_df_pg = adata.uns["PhenoGraph_clusters_feature_scores"]
# stop1 = time.time()

# print(f"\t ===> Finished ranking features across clusters in {stop1-start1} seconds")

# print("Sorting most common features")

# print('fc_df', fc_df)

# top5_leiden = top_common_features(fc_df)

# if args.include_all_views:

#     top5_leiden.to_csv(
#         os.path.join(
#             args.output_dir, f"{args.cohort}_top5_features_across_clusters_leiden.tsv"
#         ),
#         sep="\t",
#     )

# else:
#     top5_leiden.to_csv(
#         os.path.join(
#             args.output_dir,
#             f"{args.cohort}_top5_features_across_clusters_leiden_remove_{args.remove_view}.tsv",
#         ),
#         sep="\t",
#     )

# top5_pg = top_common_features(fc_df_pg)

# if args.include_all_views:

#     top5_pg.to_csv(
#         os.path.join(args.output_dir, f"{args.cohort}_top5_features_across_clusters_pg.tsv"),
#         sep="\t",
#     )

# else:
#     top5_pg.to_csv(
#         os.path.join(args.output_dir, f"{args.cohort}_top5_features_across_clusters_pg_remove_{args.remove_view}.tsv"),
#         sep="\t",
#     )

print("Neighbourhood enrichment analysis")

# sq.gr.co_occurrence(adata, cluster_key="leiden")  # if it works, it works

sq.gr.spatial_neighbors(adata)
sq.gr.nhood_enrichment(adata, cluster_key="leiden")
# sq.gr.nhood_enrichment(adata, cluster_key="PhenoGraph_clusters")


# sc.pl.umap(adata, color=['leiden', 'DNA1', 'panCK', 'CD45', 'Vimentin', 'CD3', 'CD20'], show=False, save=f"_{args.cohort}")

stopa = time.time()

log_file.write(f"All analysis done in {(stopa-starta)/60} minutes")

log_file.close()

# print("old", adata.uns.keys())

# new_uns = {str(k):v for k,v in adata.uns.items()}

# print("new", new_uns.keys())

# # adata.uns = new_uns

if args.include_all_views:
    adata.obs.to_csv(
        os.path.join(args.output_dir, f"{args.cohort}_clusters.tsv"), sep="\t"
    )
    adata.write(os.path.join(args.output_dir, f"{args.cohort}_adata_new.h5ad"))

else:
    adata.obs.to_csv(
        os.path.join(
            args.output_dir, f"{args.cohort}_remove_{args.remove_view}_clusters.tsv"
        ),
        sep="\t",
    )
    adata.write(
        os.path.join(
            args.output_dir, f"{args.cohort}_adata_remove_{args.remove_view}.h5ad"
        )
    )
