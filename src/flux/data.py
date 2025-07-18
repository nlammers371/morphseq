import torch
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import pytorch_lightning as pl

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, z0, z1, exp_idx, emb_idx, Δt):
        # store CPU tensors (float32 + int64)
        self.z0, self.z1 = z0, z1
        self.exp_idx, self.emb_idx = exp_idx, emb_idx
        self.dt = Δt

    def __len__(self):
        return self.z0.shape[0]

    def __getitem__(self, i):
        dz = (self.z1[i] - self.z0[i]) / self.dt[i]   # finite diff target
        return {
            "z0": self.z0[i],
            "dz": dz,
            "exp": self.exp_idx[i],
            "emb": self.emb_idx[i],
        }
    


def load_embryo_df(root:Path, 
                   model_class: str= "legacy", 
                   model_name: str="20241107_ds_sweep01_optimum"):
    

    data_path = root / "models" / model_class / model_name
    if not data_path.exists():
        raise FileNotFoundError(f"Model path {data_path} does not exist.")
    
    # Load the embeddings
    embryo_df = pd.read_csv(data_path / "embryo_stats_df.csv", index_col=0)
    if embryo_df.columns[0].startswith("Unnamed"):
        embryo_df.set_index(embryo_df.columns[0], inplace=True)

    return embryo_df


def build_traing_data(df: pd.DataFrame, 
                      min_frames:int=5,
                      max_dt:int=3600,
                      use_pca:bool=False,
                      pca_components:int=10):
    
    """ 
    Builds training data from the embryo DataFrame.
    Args:
        df (pd.DataFrame): DataFrame containing embryo metadata.
        min_frames (int): Minimum number of frames for a valid embryo.
        max_dt (int): Maximum time difference in seconds.
        use_pca (bool): Whether to use PCA for dimensionality reduction.
        pca_components (int): Number of PCA components to keep if use_pca is True.
    Returns:
        PairDataset: A dataset containing pairs of embeddings and their differences.
    """
    
    # get z col names
    latent_cols = [df.columns[i] for i in range(1, len(df.columns)) if df.columns[i].startswith("z_mu_b")]
    if use_pca:
        # Apply PCA to the mu columns
        pca = PCA(n_components=pca_components)
        mu_data = embryo_df[latent_cols].values
        mu_data_pca = pca.fit_transform(mu_data)
        latent_cols = [f"pca_{i}" for i in range(pca_components)]
        embryo_df[latent_cols] = mu_data_pca

    # generate IDs
    _, em_id = np.unique(embryo_df["embryo_id"], return_inverse=True)
    _, ex_id = np.unique(embryo_df["experiment_date"].astype(str), return_inverse=True)
    embryo_df["em_id"] = em_id.astype(int)
    embryo_df["ex_id"] = ex_id.astype(int)

    # strip things down
    merge_df0 = embryo_df.loc[:, ["em_id", "ex_id", "experiment_time"] + latent_cols]
    counts = merge_df0["em_id"].value_counts()
    merge_df0 = merge_df0[merge_df0["em_id"].map(counts) >= min_frames]
    merge_df0["row_idx"] = merge_df0.groupby("em_id").cumcount() 
    merge_df0 = merge_df0.rename(columns={"experiment_time": "t"})

    # create pairs
    merge_df1 = merge_df0.copy()
    merge_df0["row_idx"] += 1

    merged_df = merge_df0.merge(merge_df1,
                                on=["em_id", "row_idx"],
                                how="inner",
                                suffixes=("_0", "_1")
                            )
    merged_df["dt"] = merged_df["t_1"] - merged_df["t_0"]
    merged_df = merged_df[(merged_df["dt"] > 0) & (merged_df["dt"] <= max_dt)]

    # create data arrays
    z0 = merged_df[[f"{col}_0" for col in latent_cols]].values.astype("float32")
    z1 = merged_df[[f"{col}_1" for col in latent_cols]].values.astype("float32")
    exp_idx = merged_df["ex_id_0"].values
    emb_idx = merged_df["em_id"].values
    dt = merged_df["dt"].values.astype("float32")

    return PairDataset(
        z0=torch.tensor(z0),
        z1=torch.tensor(z1),
        exp_idx=torch.tensor(exp_idx),
        emb_idx=torch.tensor(emb_idx),
        dt=torch.tensor(dt)
    )


# class NVFData(pl.LightningDataModule):
#     def __init__(self, *arrays, batch_size=8192):
#         super().__init__()
#         self.arrays, self.bs = arrays, batch_size
#     def setup(self, stage=None):
#         self.ds = PairDataset(*self.arrays)
#     def train_dataloader(self):
#         return DataLoader(self.ds, batch_size=self.bs, shuffle=True, pin_memory=True)

if __name__ == "__main__":
    root = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/")
    model_class = "legacy"
    model_name = "20241107_ds_sweep01_optimum"

    embryo_df = load_embryo_df(root, model_class, model_name)
    test = build_traing_data(df=embryo_df,)
    test[0]
    print("check")


