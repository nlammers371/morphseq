import sys
sys.path.append("/functions")
# print("hello")
# import functions
from functions.pythae_utils import *
import os
from pythae.models import VAE, VAEConfig
from pythae.trainers import BaseTrainerConfig
from pythae.pipelines.training import TrainingPipeline
# from pythae.models import AutoModel
# import matplotlib.pyplot as plt
# from pythae.samplers import NormalSampler
import argparse

def train_vanilla_vae(train_dir, n_latent=16, batch_size=16, n_epochs=100, learning_rate=1e-3,
                      input_dim=None, depth=5):

    # input_dim = (1, 128, 128)
    if input_dim == None:
        input_dim = (1, 576, 256)
        transform = data_transform
    else:
        transform = make_dynamic_rs_transform(input_dim[1:])

    model_name = f'z{n_latent:02}_' + f'bs{batch_size:03}_' + f'ne{n_epochs:03}_' + f'depth{depth:02}'
    output_dir = os.path.join(train_dir, model_name)

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
        latent_dim=n_latent
    )


    encoder = Encoder_Conv_VAE_FLEX(model_config, n_conv_layers=depth) # these are custom classes I wrote for this use case
    # if matched_decoder_flag:
    decoder = Decoder_Conv_AE_FLEX_Matched(encoder)
    # else:
    #     decoder = Decoder_Conv_AE_FLEX(encoder)

    model = VAE(
        model_config=model_config,
        encoder=encoder,
        decoder=decoder
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
    # from functions.pythae_utils import *

    # parser = argparse.ArgumentParser(description="Function to call VAE training in batch",
    #                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # # root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"
    # parser.add_argument("-rd", "--root", default="E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\", help="string path to folder contining 'training_data' subfolder")
    # parser.add_argument("-nz", "--n_latent", default=25, help="Integer specifying number of latent dimensions")
    # parser.add_argument("-ne", "--n_epochs", default=100, help="Integer specifying number of training epochs")
    # parser.add_argument("-bs", "--batch_size", default=32, help="Integer specifying number of images per batch")
    # parser.add_argument("-md", "--depth", default=5, help="Integer specifying number of convolutional layers")
    # parser.add_argument("-tp", "--train_folder", default="20230815_vae", help="Folder containing training data to use")
    # parser.add_argument("-lt", "--learning_rate", default=1e-4, help="float <<1 specifying learning rate for model training")
    #
    # args = vars(parser.parse_args())
    #
    # train_dir = os.path.join(args["root"], "training_data", args["train_folder"])

    root = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\"
    train_folder = "20230815_vae"
    train_dir = os.path.join(root, "training_data", train_folder)
    batch_size = 32
    n_epochs = 100
    z_dim_vec = [10, 25, 50]
    depth_vec = [3, 5, 6, 7]
    max_tries = 3
    # output_dir = train_vanilla_vae(train_dir, n_latent=10, batch_size=batch_size, n_epochs=n_epochs,
    #                                learning_rate=1e-4, depth=7)
    for z in z_dim_vec[1:]:
        for d in depth_vec:
            iter_flag = 0
            while iter_flag < max_tries:
                try:
                    print(f"Depth: {d}")
                    print(f"Latent dim: {z}")
                    # train model
                    output_dir = train_vanilla_vae(train_dir, n_latent=z, batch_size=batch_size, n_epochs=n_epochs,
                                               learning_rate=1e-4, depth=d)
                    iter_flag = max_tries
                except:
                    iter_flag += 1
                    print(iter_flag)

    # last_training = sorted(os.listdir(output_dir))[-1]
    # trained_model = AutoModel.load_from_folder(
    #     os.path.join(output_dir, last_training, 'final_model'))
    #
    # # create normal sampler
    # normal_samper = NormalSampler(
    #     model=trained_model
    # )
    #
    # # sample
    # gen_data = normal_samper.sample(
    #     num_samples=9
    # )
    #
    # # show results with normal sampler
    # fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(6, 6))
    #
    # for i in range(3):
    #     for j in range(3):
    #         axes[i][j].imshow(gen_data[i * 2 + j].cpu().squeeze(0), cmap='gray')
    #         axes[i][j].axis('off')
    # plt.tight_layout(pad=0.)
    #
    # plt.show()