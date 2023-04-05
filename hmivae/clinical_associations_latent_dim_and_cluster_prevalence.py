### Clinical associations

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import statsmodels.api as sm
import tifffile
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

### Load data

cohort = "basel"

adata = sc.read_h5ad(
    f"../cluster_analysis/{cohort}/best_run_{cohort}_no_dna/results_diff_res/{cohort}_adata_new.h5ad"
)  # directory where adata was stored

patient_data = pd.read_csv(
    f"{cohort}/{cohort}_survival_patient_samples.tsv", sep="\t", index_col=0
)

clinical_variables = [
    "ERStatus",
    "grade",
    "PRStatus",
    "HER2Status",
    "Subtype",
    "clinical_type",
    "HR",
]  # changes for each cohort, here example is basel

patient_col = "PID"

cluster_col = "leiden"

### Visualize the data

plt.rcParams["figure.figsize"] = [10, 10]

for n, i in enumerate(clinical_variables):
    ax = plt.subplot(4, 2, n + 1)
    df = pd.DataFrame(patient_data[i].value_counts()).transpose()

    df.plot.bar(ax=ax)

    ax.set_xticklabels([i], rotation=0)
    plt.legend(bbox_to_anchor=[1.0, 1.1])

# Patient / Latent Variable associations

df = pd.DataFrame(
    columns=["Sample_name"]
    + [f"median_latent_dim_{n}" for n in range(adata.obsm["VAE"].shape[1])]
)

for n, sample in enumerate(adata.obs.Sample_name.unique()):
    sample_adata = adata.copy()[adata.obs.Sample_name.isin([sample]), :]

    df.loc[str(n)] = [sample] + np.median(sample_adata.obsm["VAE"], axis=0).tolist()

patient_latent = pd.merge(df, patient_data, on="Sample_name")

## first try

latent_dim_cols = [i for i in patient_latent.columns if "median" in i]
exception_variables = []

dfs = []

for cvar in clinical_variables:
    cvar_dfs = []

    for sub_cvar in patient_latent[cvar].unique():
        print(cvar, sub_cvar)
        sub_cvar_df = pd.DataFrame({})
        selected_df = patient_latent.copy()[
            ~patient_latent[cvar].isna()
        ]  # drop nan values for each var
        selected_df[cvar] = list(map(int, selected_df[cvar] == sub_cvar))

        X = selected_df[
            latent_dim_cols
        ].to_numpy()  # select columns corresponding to latent dims and convert to numpy
        X = sm.add_constant(X)  # add constant
        y = selected_df[
            cvar
        ].to_numpy()  # select the clinical variable column and convert to numpy -- no fillna(0) since all the nans should have been dropped
        try:
            log_reg = sm.Logit(y, X).fit()  # fit the Logistic Regression model

            sub_cvar_df["latent_dim"] = [c.split("_")[-1] for c in latent_dim_cols]

            sub_cvar_df["tvalues"] = log_reg.tvalues[1:]  # remove the constant

            sub_cvar_df["clinical_variable"] = [
                f"{cvar}:{sub_cvar}"
            ] * sub_cvar_df.shape[0]

            cvar_dfs.append(sub_cvar_df)
        except Exception as e:
            exception_variables.append((cvar, sub_cvar))
            print(f"{cvar}:{sub_cvar} had an exception occur: {e}")

    full_cvar_dfs = pd.concat(cvar_dfs)

    dfs.append(full_cvar_dfs)

## Second try, which features caused issues for which clinical variable

features_to_remove = []
# cvar_dfs2 = []
for cvar, sub_cvar in exception_variables:
    # print(cvar, sub_cvar)
    selected_df = patient_latent[
        ~patient_latent[cvar].isna()
    ].copy()  # drop nan values for each var
    selected_df[cvar] = list(map(int, selected_df[cvar] == sub_cvar))
    y = selected_df[cvar].to_numpy()
    X = selected_df[
        latent_dim_cols
    ].to_numpy()  # select columns corresponding to latent dims and convert to numpy

    perf_sep_features = []
    for i in range(X.shape[1]):
        X_1 = X.copy()[:, 0 : i + 1]
        X_1 = sm.add_constant(X_1)  # add constant
        try:
            log_reg = sm.Logit(y, X_1).fit()  # fit the Logistic Regression model
            print(
                f"Completed: tvalues for {cvar}:{sub_cvar}, features till {i} -> {log_reg.tvalues}"
            )
            # print(log_reg.summary())
        except Exception as e:
            print(f"{cvar}:{sub_cvar} for feature {i} has exception: {e}")
            perf_sep_features.append(i)

    # if len(perf_sep_features) == 0:
    #     sub_cvar_df = pd.DataFrame({})
    #     sub_cvar_df['latent_dim'] = [c.split('_')[-1] for c in latent_dim_cols]

    #     assert len(log_reg.tvalues) == X.shape[1]+1 #for constant -- check this is the last one

    #     sub_cvar_df['tvalues'] = log_reg.tvalues[1:] # remove the constant -- this should be the last one

    #     sub_cvar_df['clinical_variable'] = [f"{cvar}:{sub_cvar}"]*sub_cvar_df.shape[0]

    #     cvar_dfs2.append(sub_cvar_df) # this will often turn out to be empty since if it gave issues before, it should give issues now

    # else:

    features_to_remove.append((cvar, sub_cvar, perf_sep_features))

## final try, remove the features causing issues and store their t-value as NaN

sub_cvars = []

for cvar, sub_cvar, del_inds in features_to_remove:
    selected_df = patient_latent[
        ~patient_latent[cvar].isna()
    ].copy()  # drop nan values for each var
    selected_df[cvar] = list(map(int, selected_df[cvar] == sub_cvar))
    y = selected_df[cvar].to_numpy()
    X = selected_df[
        latent_dim_cols
    ].to_numpy()  # select columns corresponding to latent dims and convert to numpy
    del_inds = del_inds
    X = np.delete(X, del_inds, axis=1)
    print(X.shape)
    X = sm.add_constant(X)  # add constant
    try:
        log_reg = sm.Logit(y, X).fit()  # fit the Logistic Regression model
        print(
            f"Completed: tvalues for {cvar}:{sub_cvar}, features till {i} -> {log_reg.tvalues}"
        )

        sub_cvar_df = pd.DataFrame({})
        sub_cvar_df["latent_dim"] = [c.split("_")[-1] for c in latent_dim_cols]

        tvalues = log_reg.tvalues[1:].tolist()  # + [np.nan]

        for i in del_inds:
            if i > len(tvalues):
                tvalues = np.insert(tvalues, i - 1, np.nan)
            else:
                tvalues = np.insert(tvalues, i, np.nan)

        #         tvalues = np.insert(tvalues, del_inds.remove(19), np.nan)
        # assert len(log_reg.tvalues) == X.shape[1]+1 #for constant -- check this is the last one

        sub_cvar_df["tvalues"] = tvalues

        sub_cvar_df["clinical_variable"] = [f"{cvar}:{sub_cvar}"] * sub_cvar_df.shape[0]

        sub_cvars.append(sub_cvar_df)
    except Exception as e:
        print(f"{cvar}:{sub_cvar} for feature {i} has exception: {e}")

sub_cvar_df1 = pd.concat(sub_cvars)

full_clin_df = pd.concat(dfs).reset_index(drop=True)

final_full_clin_df = pd.concat([full_clin_df, sub_cvar_df1]).reset_index(drop=True)

final_full_clin_df = pd.pivot_table(
    final_full_clin_df,
    index="clinical_variable",
    values="tvalues",
    columns="latent_dim",
)  # df that's plotted


# Patient / Cluster associations
# First we need to define cluster prevalance within a patient. Doing this in two ways:
# 1. How we were doing it before -- proportion of cells in patient x that belong to cluster c
# 2. Cells of cluster c per mm^2 of tissue

clusters_patient = pd.merge(
    adata.obs.reset_index()[["Sample_name", "leiden", "cell_id"]],
    patient_data.reset_index(),
    on="Sample_name",
)

## Option 1: Proportion of cells in patient x that belong in cluster c

hi_or_low = clusters_patient[[patient_col, cluster_col]]

## Proportion of cells belonging to each cluster for each image / patient

hi_or_low = hi_or_low.groupby([patient_col, cluster_col]).size().unstack(fill_value=0)


hi_or_low = hi_or_low.div(hi_or_low.sum(axis=1), axis=0).fillna(0)


hi_low_cluster_variables = (
    pd.merge(
        hi_or_low.reset_index(),
        clusters_patient[clinical_variables + [patient_col]],
        on=patient_col,
    )
    .drop_duplicates()
    .reset_index(drop=True)
)

prop_cluster_cols = [
    i
    for i in hi_low_cluster_variables.columns
    if i in clusters_patient[cluster_col].unique()
]
exception_variables = []

dfs = []

for cvar in clinical_variables:
    cvar_dfs = []
    filtered_df = hi_low_cluster_variables[
        ~hi_low_cluster_variables[cvar].isna()
    ].copy()  # drop nan values for each var

    for sub_cvar in filtered_df[cvar].unique():
        print(cvar, sub_cvar)
        selected_df = filtered_df.copy()
        selected_df[cvar] = list(map(int, selected_df[cvar] == sub_cvar))
        sub_cvar_df = pd.DataFrame({})
        y = selected_df[
            cvar
        ].to_numpy()  # select the clinical variable column and convert to numpy -- no fillna(0) since all the nans should have been dropped
        X = selected_df[
            prop_cluster_cols
        ].to_numpy()  # select columns corresponding to latent dims and convert to numpy
        tvalues = {}
        for cluster in range(X.shape[1]):
            X1 = X[:, cluster]
            X1 = sm.add_constant(X1)
            try:
                log_reg = sm.Logit(y, X1).fit()  # fit the Logistic Regression model

                tvalues[cluster] = log_reg.tvalues[
                    1
                ]  # there will be 2 t values, first one belongs to the constant

            except Exception as e:
                exception_variables.append((cvar, sub_cvar, cluster, e))
                print(
                    f"{cvar}:{sub_cvar} had an exception occur for cluster {cluster}: {e}"
                )

        sub_cvar_df["cluster"] = list(tvalues.keys())

        sub_cvar_df["tvalues"] = list(tvalues.values())

        sub_cvar_df["clinical_variable"] = [f"{cvar}:{sub_cvar}"] * sub_cvar_df.shape[0]

        cvar_dfs.append(sub_cvar_df)

    full_cvar_dfs = pd.concat(cvar_dfs)

    dfs.append(full_cvar_dfs)

full_cluster_clin_df = pd.concat(dfs).reset_index(drop=True)

full_cluster_clin_df = pd.pivot_table(
    full_cluster_clin_df, index="clinical_variable", values="tvalues", columns="cluster"
)  # df that's plotted

## Option 2: Number of cells per mm^2 tissue
# We're going to do this per image for now -- mainly because sizes might differ between images that belong to the same patient

clinical_variables = clinical_variables + [
    "diseasestatus"
]  # for basel, since doing per image

cohort_dirs = {
    "basel": ["OMEnMasks/Basel_Zuri_masks", "_a0_full_maks.tiff"],
    "metabric": ["METABRIC_IMC/to_public_repository/cell_masks", "_cellmask.tiff"],
    "melanoma": [
        "full_data/protein/cpout/",
        "_ac_ilastik_s2_Probabilities_equalized_cellmask.tiff",
    ],
}  # directories with the masks

adata_df = adata.obs.reset_index()[["cell_id", "Sample_name", "leiden"]]
clusters = adata_df.leiden.unique().tolist()

sample_dfs = []

progress = Progress(
    TextColumn(f"[progress.description]Finding cluster prevalances in {cohort}."),
    BarColumn(),
    TaskProgressColumn(),
    TimeElapsedColumn(),
)
with progress:
    for sample in progress.track(adata.obs.Sample_name.unique()):
        s_df = pd.DataFrame({})
        s_cluster_prevs = {}
        mask = tifffile.imread(
            f"../../../data/{cohort_dirs[cohort][0]}/{sample}{cohort_dirs[cohort][1]}"
        )
        sample_df = adata_df.copy().query("Sample_name==@sample")
        for cluster in clusters:
            num_cells_in_sample = Counter(sample_df.leiden.tolist())
            num_cells_in_clusters = num_cells_in_sample[cluster]

            # print(num_cells_in_clusters)
            # print(mask.shape[0] , mask.shape[1])

            cluster_prevalance_per_mm2 = (
                num_cells_in_clusters / (mask.shape[0] * mask.shape[1])
            ) * 1e6  # scale, 1 pixel == 1 micron

            s_cluster_prevs[cluster] = cluster_prevalance_per_mm2

        s_df["cluster"] = list(s_cluster_prevs.keys())
        s_df["prevalance_per_mm2_scaled_by_1e6"] = list(s_cluster_prevs.values())
        s_df["Sample_name"] = [sample] * s_df.shape[0]

        sample_dfs.append(s_df)

full_cohort_df = pd.concat(sample_dfs)

full_cohort_df["cluster"] = full_cohort_df["cluster"].map(int)

full_cohort_df = pd.pivot_table(
    full_cohort_df,
    values="prevalance_per_mm2_scaled_by_1e6",
    index="Sample_name",
    columns="cluster",
)

clusters = full_cohort_df.columns.tolist()  # to make sure correct order later

cluster_per_tissue_patient = pd.merge(
    full_cohort_df, patient_data[clinical_variables + ["Sample_name"]], on="Sample_name"
)

# The below is still being run and tested but this is close to what I will be doing

cluster_cols = clusters
exception_variables = []

dfs = []

for cvar in clinical_variables:
    cvar_dfs = []

    for sub_cvar in cluster_per_tissue_patient[cvar].dropna().unique().tolist():
        print(cvar, sub_cvar)
        sub_cvar_df = pd.DataFrame({})
        selected_df = cluster_per_tissue_patient.copy()[
            ~cluster_per_tissue_patient[cvar].isna()
        ]  # drop nan values for each var
        selected_df[cvar] = list(map(int, selected_df[cvar] == sub_cvar))

        X = selected_df[
            cluster_cols
        ].to_numpy()  # select columns corresponding to latent dims and convert to numpy
        X = sm.add_constant(X)  # add constant
        y = selected_df[
            cvar
        ].to_numpy()  # select the clinical variable column and convert to numpy -- no fillna(0) since all the nans should have been dropped
        try:
            log_reg = sm.Logit(y, X).fit()  # fit the Logistic Regression model

            sub_cvar_df["cluster"] = [c for c in cluster_cols]

            sub_cvar_df["tvalues"] = log_reg.tvalues[1:]  # remove the constant

            sub_cvar_df["clinical_variable"] = [
                f"{cvar}:{sub_cvar}"
            ] * sub_cvar_df.shape[0]

            cvar_dfs.append(sub_cvar_df)
        except Exception as e:
            exception_variables.append((cvar, sub_cvar))
            print(f"{cvar}:{sub_cvar} had an exception occur: {e}")

    full_cvar_dfs = pd.concat(cvar_dfs)

    dfs.append(full_cvar_dfs)
