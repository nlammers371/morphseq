import os
import sys

code_root = "/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq"
sys.path.insert(0, code_root)

# import glob as glob
# from src.functions.dataset_utils import *
# import os
from src._Archive.vae import AutoModel
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.decomposition import PCA
import numpy as np
import torch
import skimage.io as io

###################
# Step 2: train an MLP model to predict nuisance embeddings from pca compression of biological components
###################
# initialize model
def train_nbio_mlp(pca_df, morph_df, mdl_config=None):

    # get colnames
    # bio_morph_cols = [col for col in ref_morph_df.columns if "z_mu_b" in col]
    nbio_morph_cols = [col for col in morph_df.columns if "z_mu_n" in col]
    # morph_cols = [col for col in ref_morph_df.columns if "z_mu" in col]
    
    pca_cols = [col for col in pca_df.columns if "PCA" in col]

    mlp = MLPRegressor(random_state=42, max_iter=20000, hidden_layer_sizes=mdl_config, tol=1e-4)

    # get data
    X = pca_df[pca_cols].values
    Y = morph_df[nbio_morph_cols].values

    # make test/train splits
    n_obs = X.shape[0]
    test_frac = 0.1
    n_test = int(np.ceil(test_frac * n_obs))

    tt_options = np.arange(n_obs)
    test_indices = np.random.choice(tt_options, n_test, replace=False)
    train_indices = tt_options[~np.isin(tt_options, test_indices)]

    X_train = X[train_indices]
    Y_train = Y[train_indices]
    X_test = X[test_indices]
    Y_test = Y[test_indices]

    mlp.fit(X_train, Y_train)

    train_score = mlp.score(X_train, Y_train)
    test_score = mlp.score(X_test, Y_test)

    return mlp, (train_score, test_score)



if __name__ == "__main__":

    root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"

    # set path to VAE model 
    train_name = "20241107_ds"
    model_name = "SeqVAE_z100_ne150_sweep_01_block01_iter030" 
    train_dir = os.path.join(root, "training_data", train_name, "")
    output_dir = os.path.join(train_dir, model_name) 


    # get path to model
    training_path = sorted(glob(os.path.join(output_dir, "*")))[-1]
    training_name = os.path.dirname(training_path)
    # read_path = os.path.join(training_path, "figures", "")

    # path to save data
    read_path = os.path.join(root, "results", "20250312", "morph_latent_space", "")

    # set write path
    im_write_path = os.path.join(read_path, "images", "")
    os.makedirs(im_write_path, exist_ok=True)

    # load in PCA embeddings and PCA model
    hf_pca_df = pd.read_csv(os.path.join(read_path, "hf_pca_morph_df.csv"))
    ref_pca_df = pd.read_csv(os.path.join(read_path, "ab_pca_ref_morph_df.csv"))
    spline_df = pd.read_csv(os.path.join(read_path, "spline_morph_df_full.csv"))

    # load in full latent space infop
    hf_morph_df = pd.read_csv(os.path.join(read_path, "hf_morph_df.csv"))
    ref_morph_df = pd.read_csv(os.path.join(read_path, "ab_ref_morph_df.csv"))

    # get colnames
    bio_morph_cols = [col for col in ref_morph_df.columns if "z_mu_b" in col]
    nbio_morph_cols = [col for col in ref_morph_df.columns if "z_mu_n" in col]
    morph_cols = [col for col in ref_morph_df.columns if "z_mu" in col]

    pca_cols = [col for col in hf_pca_df.columns if "PCA" in col]

    # Load the saved parameters
    data = np.load(os.path.join(read_path,'morph_pca_params.npz'))
    n_components = data['components_'].shape[0]

    # Create a new PCA instance (n_components must match)
    morph_pca = PCA(n_components=n_components)

    # Manually set the PCA attributes
    morph_pca.components_ = data['components_']
    morph_pca.explained_variance_ = data['explained_variance_']
    morph_pca.singular_values_ = data['singular_values_']
    morph_pca.mean_ = data['mean_']

    ###################
    # Procedure: 
    #   (1) train MLP to predict nbio from nio pca
    #   (2) get bio from pca using inverse_transform
    #   (3) get nbio from MLP
    #   (4) use concatenated vector to generate an image
     
    ###################
    # initialize model assessment
    trained_model = AutoModel.load_from_folder(os.path.join(output_dir, 'final_model'))

    np.random.seed(123)
    recon_options = np.arange(ref_morph_df.shape[0])
    batch_size = 8
    n_recon = 12*batch_size
    n_iters = int(n_recon/batch_size)
    recon_indices = np.random.choice(recon_options, n_recon)

    for n in tqdm(range(n_iters), "generating images..."):
  
        n_start = n*batch_size
        indices = recon_indices[n_start:n_start + batch_size]

        # generate reconstructed latent vectors
        pca_i = ref_pca_df.loc[indices, pca_cols].values
        # get bio prediction
        bio_cols = morph_pca.inverse_transform(pca_i).astype(float)
        # get nbio
        nbio_cols = mlp.predict(pca_i).astype(float)
        # combine
        full_morph_cols = np.hstack((nbio_cols, bio_cols))
        morph_input = torch.tensor(full_morph_cols).reshape(batch_size, 1, 1, -1).float()
        recon_x_out = trained_model.decoder(morph_input)["reconstruction"]

        # save images
        snip_vec = ref_morph_df.loc[indices, "snip_id"].to_numpy()
        stage_vec = ref_morph_df.loc[indices, "predicted_stage_hpf"].to_numpy()

        for b in range(recon_x_out.shape[0]):
            im_arr = np.asarray(np.squeeze(recon_x_out[b].detach()))
            snip = snip_vec[b]
            stage = np.round(stage_vec[b]).astype(int)
            im_name = f"{snip}_stage{stage:03}.jpg"
            io.imsave(os.path.join(im_write_path, im_name), im_arr)


    # i = 27 
    # get PCA 
    # pca_i = ref_pca_df.loc[i, pca_cols].values[None, :]
    # # get bio prediction
    # bio_cols = morph_pca.inverse_transform(pca_i).astype(float)
    # # get nbio
    # nbio_cols = mlp.predict(pca_i).astype(float)
    # # combine
    # full_morph_cols = np.hstack((nbio_cols, bio_cols))
    # morph_input = torch.tensor(full_morph_cols).reshape(1, 1, 1, -1).float()
    # recon_x_out = trained_model.decoder(morph_input)["reconstruction"]
    