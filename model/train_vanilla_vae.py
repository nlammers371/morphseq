from functions.pythae_utils import *
import os
from pythae.models import VAE, VAEConfig
from pythae.trainers import BaseTrainerConfig
from pythae.models.nn.benchmarks.mnist import Encoder_Conv_VAE_MNIST, Decoder_Conv_AE_MNIST
from pythae.pipelines.training import TrainingPipeline
from pythae.models import AutoModel
import matplotlib.pyplot as plt
from pythae.samplers import NormalSampler

def train_vanilla_vae(train_dir, latent_dim=16, batch_size=16, n_epochs=100, learning_rate=1e-3,
                      conv_flag=True, input_dim=None, depth=4):

    # input_dim = (1, 128, 128)
    if input_dim == None:
        input_dim = (1, 576, 256)
        transform = data_transform
    else:
        transform = make_dynamic_rs_transform(input_dim[1:])

    if conv_flag:
        prefix = '_conv'
    else:
        prefix = ''

    model_name = prefix + f'_z{n_latent:02}_' + f'bs{batch_size:03}_' + f'ne{n_epochs:03}'
    output_dir = os.path.join(train_dir, train_name + model_name)

    train_dataset = MyCustomDataset(
        root=os.path.join(train_dir, "train"),
        transform=transform
    )

    eval_dataset = MyCustomDataset(
        root=os.path.join(train_dir, "eval"),
        transform=transform
    )

    config = BaseTrainerConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_epochs=n_epochs,  # Change this to train the model a bit more
    )

    model_config = VAEConfig(
        input_dim=input_dim,
        latent_dim=latent_dim
    )

    if conv_flag:
        encoder = Encoder_Conv_VAE_FLEX(model_config, n_conv_layers=depth) # these are custom classes I wrote for this use case
        decoder = Decoder_Conv_AE_FLEX(encoder)

        model = VAE(
            model_config=model_config,
            encoder=encoder,
            decoder=decoder
        )
    else:
        model = VAE(
            model_config=model_config
        )

    pipeline = TrainingPipeline(
        training_config=config,
        model=model
    )

    pipeline(
        train_data=train_dataset,  # here we use the custom train dataset
        eval_data=eval_dataset  # here we use the custom eval dataset
    )

    return output_dir


if __name__ == "__main__":

    # root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"
    root = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\"
    train_name = "20230804_vae_full"
    n_latent = 10
    batch_size = 32
    n_epochs = 100
    conv_flag = True
    if conv_flag:
        prefix = 'conv'
    else:
        prefix = ''

    model_name = f'_z{n_latent:02}_' + f'bs{batch_size:03}_' + f'ne{n_epochs:03}'
    train_dir = os.path.join(root, "training_data", train_name)

    # train model
    output_dir = train_vanilla_vae(train_dir, latent_dim=n_latent, batch_size=batch_size, n_epochs=n_epochs,
                                   learning_rate=1e-4, conv_flag=conv_flag) #, input_dim=(1, 256, 128))

    last_training = sorted(os.listdir(output_dir))[-1]
    trained_model = AutoModel.load_from_folder(
        os.path.join(output_dir, last_training, 'final_model'))

    # create normal sampler
    normal_samper = NormalSampler(
        model=trained_model
    )

    # sample
    gen_data = normal_samper.sample(
        num_samples=9
    )

    # show results with normal sampler
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(6, 6))

    for i in range(3):
        for j in range(3):
            axes[i][j].imshow(gen_data[i * 2 + j].cpu().squeeze(0), cmap='gray')
            axes[i][j].axis('off')
    plt.tight_layout(pad=0.)

    plt.show()