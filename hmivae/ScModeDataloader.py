import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import TensorDataset


def sparse_numpy_to_torch(adj_mat):
    """Construct sparse torch tensor
    Need to do csr -> coo
    then follow https://stackoverflow.com/questions/50665141/converting-a-scipy-coo-matrix-to-pytorch-sparse-tensor
    """
    adj_mat_coo = adj_mat.tocoo()

    values = adj_mat_coo.data
    indices = np.vstack((adj_mat_coo.row, adj_mat_coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = adj_mat_coo.shape

    return torch.sparse_coo_tensor(i, v, shape)


class ScModeDataloader(TensorDataset):
    def __init__(self, adata, scalers=None):
        """
        Need to get the following from adata:
        Y - NxP mean expression matrix
        S - Nx(pC2) correlation matrix
        M - Nx7 morphology matrix
        scalers: set of data scalers
        """
        self.adata = adata
        Y = adata.X
        S = adata.obsm["correlations"]
        M = adata.obsm["morphology"]

        if scalers is None:
            self.scalers = {}
            self.scalers["Y"] = StandardScaler().fit(Y)
            self.scalers["S"] = StandardScaler().fit(S)
            self.scalers["M"] = StandardScaler().fit(M)
        else:
            self.scalers = scalers

        Y = self.scalers["Y"].transform(Y)
        S = self.scalers["S"].transform(S)
        M = self.scalers["M"].transform(M)

        self.Y = torch.tensor(Y).float()
        self.S = torch.tensor(S).float()
        self.M = torch.tensor(M).float()
        self.C = self.get_spatial_context()

        self.samples_onehot = self.one_hot_encoding()

    def __len__(self):
        return self.Y.shape[0]

    def one_hot_encoding(self, test=False):
        """
        Creates a onehot encoding for samples.
        """
        onehotenc = OneHotEncoder()
        X = self.adata.obs[["Sample_name"]]
        onehot_X = onehotenc.fit_transform(X).toarray()

        df = pd.DataFrame(onehot_X, columns=onehotenc.categories_[0])

        df = df.reindex(columns=self.adata.obs.Sample_name.unique().tolist())

        return torch.tensor(df.to_numpy()).float()

    def get_spatial_context(self):
        adj_mat = sparse_numpy_to_torch(
            self.adata.obsp["connectivities"]
        )  # adjacency matrix
        concatenated_features = torch.cat((self.Y, self.S, self.M), 1)

        C = torch.smm(  # normalize for the number of adjacent cells
            adj_mat, concatenated_features
        ).to_dense()  # spatial context for each cell
        return C

    def __getitem__(self, idx):

        return (
            self.Y[idx, :],
            self.S[idx, :],
            self.M[idx, :],
            self.C[idx, :],
            self.samples_onehot[idx, :],
            idx,
        )
