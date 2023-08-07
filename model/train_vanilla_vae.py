from functions.pythae_utils import *
import os
from pythae.models import VAE, VAEConfig
from pythae.trainers import BaseTrainerConfig
from pythae.pipelines.training import TrainingPipeline
from pythae.models import AutoModel
import matplotlib.pyplot as plt
from pythae.samplers import NormalSampler

def train_vanilla_vae(train_dir, latent_dim=16, batch_size=16, n_epochs=100, input_dim=None):

    if input_dim == None:
        input_dim = (1, 576, 256)

    model_name = f'_z{n_latent:02}_' + f'bs{batch_size:03}_' + f'ne{n_epochs:03}'

    train_dataset = MyCustomDataset(
        root=os.path.join(train_dir, "train"),
        transform=data_transform,
    )

    eval_dataset = MyCustomDataset(
        root=os.path.join(train_dir, "eval"),
        transform=data_transform
    )

    config = BaseTrainerConfig(
        output_dir=os.path.join(train_dir, train_name + model_name),
        learning_rate=1e-3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_epochs=n_epochs,  # Change this to train the model a bit more
    )

    model_config = VAEConfig(
        input_dim=input_dim,
        latent_dim=latent_dim
    )

    model = VAE(
        model_config=model_config
    )

    pipeline = TrainingPipeline(
        training_config=config,
        model=model
    )

    pipeline(
        train_data=train_dataset, # here we use the custom train dataset
        eval_data=eval_dataset # here we use the custom eval dataset
    )



if __name__ == "__main__":
    # root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"
    root = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\"
    train_name = "20230804_vae_full"
    n_latent = 5
    batch_size = 8
    n_epochs = 25
    model_name = f'_z{n_latent:02}_' + f'bs{batch_size:03}_' + f'ne{n_epochs:03}'
    train_dir = os.path.join(root, "training_data", train_name)

    # train model
    train_vanilla_vae(train_dir, latent_dim=n_latent, batch_size=batch_size, n_epochs=n_epochs)

    last_training = sorted(os.listdir(os.path.join(train_dir, train_name + model_name)))[-1]
    trained_model = AutoModel.load_from_folder(
        os.path.join(train_dir, train_name + model_name, last_training, 'final_model'))

    # create normal sampler
    normal_samper = NormalSampler(
        model=trained_model
    )

    # sample
    gen_data = normal_samper.sample(
        num_samples=9
    )

    # show results with normal sampler
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(3, 3))

    for i in range(3):
        for j in range(3):
            axes[i][j].imshow(gen_data[i * 2 + j].cpu().squeeze(0), cmap='gray')
            axes[i][j].axis('off')
    plt.tight_layout(pad=0.)

    plt.show()