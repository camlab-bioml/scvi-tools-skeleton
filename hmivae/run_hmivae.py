## run with hmivae
import argparse
import os
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import phenograph
import scanpy as sc
import seaborn as sns
import squidpy as sq
import torch
import wandb
from anndata import AnnData
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
from hmivae.ScModeDataloader import ScModeDataloader


def arrange_features(vars_lst, adata):

    arranged_features = {"E": [], "C": [], "M": [], "S": []}
    orig_list = vars_lst

    for i in orig_list:

        if i in adata.var_names:
            arranged_features["E"].append(i)
        elif i in adata.uns["names_morphology"]:
            arranged_features["M"].append(i)
        elif i in adata.uns["names_correlations"]:
            arranged_features["C"].append(i)
        else:
            arranged_features["S"].append(i)
    arr_list = [
        *np.sort(arranged_features["E"]).tolist(),
        *np.sort(arranged_features["C"]).tolist(),
        *np.sort(arranged_features["M"]).tolist(),
        *np.sort(arranged_features["S"]).tolist(),
    ]
    return arranged_features, arr_list


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

print("===> Clustering using integrated space")

sc.pp.neighbors(
    adata, n_neighbors=100, use_rep="VAE", key_added="vae"
)  # 100 nearest neighbours, will be used in downstream tests -- keep with PG


sc.tl.leiden(adata, neighbors_key="vae")

print("===> Clustering using specific views")

print("Expression")

sc.pp.neighbors(
    adata, n_neighbors=100, use_rep="expression_embedding", key_added="expression"
)  # 100 nearest neighbours, will be used in downstream tests -- keep with PG

sc.tl.leiden(
    adata,
    neighbors_key="expression",
    key_added="expression_leiden",
    random_state=args.random_seed,
    resolution=0.5,
)  # expression wasn't too bad

print("Correlation")

sc.pp.neighbors(
    adata, n_neighbors=100, use_rep="correlation_embedding", key_added="correlation"
)

sc.tl.leiden(
    adata,
    neighbors_key="correlation",
    key_added="correlation_leiden",
    random_state=args.random_seed,
)  # probably no need to change correlation because there were few anyways

print("Morphology")

sc.pp.neighbors(
    adata, n_neighbors=100, use_rep="morphology_embedding", key_added="morphology"
)

sc.tl.leiden(
    adata,
    neighbors_key="morphology",
    key_added="morphology_leiden",
    random_state=args.random_seed,
    resolution=0.1,
)  # pull it way down because there were LOTS of clusters

print("Spatial context")

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
    resolution=0.5,
)

print("===> Creating UMAPs")

print("Integrated space")

sc.tl.umap(adata, neighbors_key="vae", random_state=args.random_seed)

adata.obsm["X_umap_int"] = adata.obsm["X_umap"].copy()

print("Expression")

sc.tl.umap(adata, neighbors_key="expression", random_state=args.random_seed)

adata.obsm["X_umap_exp"] = adata.obsm["X_umap"].copy()

print("Correlations")

sc.tl.umap(adata, neighbors_key="correlation", random_state=args.random_seed)

adata.obsm["X_umap_corr"] = adata.obsm["X_umap"].copy()

print("Morphology")

sc.tl.umap(adata, neighbors_key="morphology", random_state=args.random_seed)

adata.obsm["X_umap_morph"] = adata.obsm["X_umap"].copy()

print("Spatial context")

sc.tl.umap(adata, neighbors_key="spatial_context", random_state=args.random_seed)

adata.obsm["X_umap_spct"] = adata.obsm["X_umap"].copy()
# ranked_dict, fc_df =
# rank_features_in_groups(
#     adata, "leiden", scale_values=False, cofactor=args.cofactor,
# )  # no scaling required because using adata_train and test which have already been normalized and winsorized -- StandardScaler still applied
# fc_df = adata.uns["leiden_feature_scores"]

# top5_leiden = top_common_features(fc_df)

# if args.include_all_views:

#     top5_leiden.to_csv(
#         os.path.join(args.output_dir, f"{args.cohort}_top5_features_across_clusters_leiden.tsv"),
#         sep="\t",
#     )

print("Neighbourhood enrichment analysis")

# sq.gr.co_occurrence(adata, cluster_key="leiden")  # if it works, it works -- didn't work, always NaNs

sq.gr.spatial_neighbors(adata)
sq.gr.nhood_enrichment(adata, cluster_key="leiden")


print("===> Create the neighbourhood features")

h5 = adata.copy()

sc.pp.neighbors(
    h5, use_rep="spatial", n_neighbors=10
)  # get spatial neighbour connectivities, we lose this when we make the new adata

data = ScModeDataloader(h5)

spatial_context = data.C.numpy()

spatial_context_names = [
    "neighbour_" + i
    for i in list(h5.var_names)
    + h5.uns["names_correlations"].tolist()
    + h5.uns["names_morphology"].tolist()
]

print("===> Creating new adata and ranking all features")

clustering = [i for i in h5.obs.columns if "leiden" in i]

all_features = np.concatenate(
    [h5.X, h5.obsm["correlations"], h5.obsm["morphology"], spatial_context], axis=1
)

names = np.concatenate(
    [
        h5.var_names,
        h5.uns["names_correlations"],
        h5.uns["names_morphology"],
        spatial_context_names,
    ]
)

all_features_df = pd.DataFrame(all_features, columns=names)


new_adata = AnnData(
    X=all_features_df,
    obs=h5.copy().obs,
    obsm=h5.copy().obsm,
    obsp=h5.copy().obsp,
    uns=h5.copy().uns,
)

for cl in clustering:
    print(f"Ranking features for clustering: {cl}")
    sc.tl.rank_genes_groups(new_adata, groupby=cl, key_added=f"{cl}_rank_gene_groups")

dfs = []

for cl in clustering:
    ranked_df = sc.get.rank_genes_groups_df(
        new_adata, group=None, key=f"{cl}_rank_gene_groups"
    )

    ranked_df["clustering"] = [cl] * ranked_df.shape[0]

    dfs.append(ranked_df)

full_ranked_df = pd.concat(dfs)

## get the top features across all the different clustering

dfs2 = {}

for cl in clustering:
    print(f"sorting ranked features for {cl}")
    fs = []
    features_df = full_ranked_df.copy().query("clustering==@cl")
    for group in features_df.group.unique():
        group_df = features_df.query("group==@group")
        top10 = group_df.names.tolist()[0:10]  # these are sorted by top

        for f in top10:
            fs.append(f)

    top_features = list(set(fs))

    new_df = pd.DataFrame({})

    for group in features_df.group.unique():
        group_df = features_df.query("group==@group")

        # print('df shape', group_df.shape[0])

        scores = (
            group_df.loc[group_df.names.isin(top_features), ["names", "scores"]]
            .set_index("names")
            .sort_index()
            .scores.tolist()
        )

        new_df[group] = scores

    # print(group, new_df.shape)

    new_df.index = np.sort(top_features)

    arr_features2, arr_list2 = arrange_features(new_df.index.to_list(), adata)

    new_df = new_df.reindex(arr_list2)

    new_df.columns = new_df.columns.map(int)

    new_df = new_df[np.sort(new_df.columns)]

    # new_df['clustering'] = [cl]*new_df.shape[0]

    dfs2[cl] = new_df

cmap = sns.diverging_palette(220, 20, as_cmap=True)

for n, cl in enumerate(clustering):
    # print(n)
    # bx = plt.subplot(6,1,n+1)
    sns.clustermap(
        dfs2[cl].fillna(0),
        row_cluster=False,
        center=0.00,
        cmap=cmap,
        vmin=-100,
        vmax=100,
        figsize=(25, 25),
        linewidth=2,
        linecolor="black",
    )

    # plt.title(f"rankings for {cl}")

    plt.savefig(f"{args.cohort}_cluster_rankings_for_{cl}.png")

print("old", new_adata.uns.keys())

new_uns = {str(k): v for k, v in new_adata.uns.items()}

print("new", new_uns.keys())

adata.uns = new_uns

if args.include_all_views:
    new_adata.obs.to_csv(
        os.path.join(args.output_dir, f"{args.cohort}_clusters.tsv"), sep="\t"
    )
    new_adata.write(os.path.join(args.output_dir, f"{args.cohort}_adata_new.h5ad"))
    full_ranked_df.to_csv(
        os.path.join(args.output_dir, f"{args.cohort}_clusters_ranked_features.tsv"),
        sep="\t",
    )

# if args.include_all_views:
#     adata.obs.to_csv(os.path.join(args.output_dir, f"{args.cohort}_clusters.tsv"), sep="\t")
#     adata.write(os.path.join(args.output_dir, f"{args.cohort}_adata_new.h5ad"))

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


# sc.pl.umap(adata[random_inds], color=['leiden'], show
