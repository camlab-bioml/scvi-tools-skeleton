import os

import pandas as pd
import scanpy as sc
from torch.utils.data import TensorDataset


class HMIDataset(TensorDataset):
    def __init__(
        self,
        h5ad_dir,
    ):
        """
        Input is a directory with all h5ad files.
        h5ad_dir: Directory containing all h5ad files for each image in HMI dataset
        transform: Default is None. Any transformations to be applied to the h5ad files
        """
        self.h5ad_dir = h5ad_dir
        self.h5ad_names = pd.DataFrame({"Sample_names": os.listdir(h5ad_dir)})

    def __len__(self):
        return len(os.listdir(self.h5ad_dir))

    def __getitem__(self, idx):
        h5ad_path = os.path.join(self.h5ad_dir, self.h5ad_names.iloc[idx, 0])
        h5ad = sc.read_h5ad(h5ad_path)

        return h5ad
