import sys
sys.path.append("/functions")
# print("hello")
# import functions
import pandas as pd
from functions.pythae_utils import make_dynamic_rs_transform, data_transform, MyCustomDataset
import os
from pythae.models import VAE, VAEConfig, BetaTCVAE, BetaTCVAEConfig, MetricVAE, MetricVAEConfig
from functions.custom_networks import Encoder_Conv_VAE, Decoder_Conv_VAE
from pythae.trainers import BaseTrainerVerboseConfig
from pythae.pipelines.training import TrainingPipeline
from functions.ContrastiveLearningDataset import ContrastiveLearningDataset
from functions.view_generator import ContrastiveLearningViewGenerator
# from custom_classes.base_trainer_metric import BaseTrainerMetric

import argparse

def train_metric_vae(root, train_folder, train_suffix='', n_latent=50, batch_size=32, n_epochs=100,
                      learning_rate=1e-3, n_out_channels=16,
                      nt_xent_temperature=1.0, distance_metric="cosine",
                      input_dim=None, depth=5, contrastive_flag=False, orth_flag=False, class_ignorance_flag=False,
                      time_ignorance_flag=False):

    if input_dim == None:
        input_dim = (1, 576, 256)
        transform = data_transform
    # else:
    #     transform = make_dynamic_rs_transform(input_dim[1:])

    train_dir = os.path.join(root, "training_data", train_folder)
    metadata_path = os.path.join(root, "metadata", '')

    # read in metadata database
    # class_key = pd.read_csv(os.path.join(metadata_path, "class_key.csv"), index_col=0)

    model_name = f'z{n_latent:02}_' + f'bs{batch_size:03}_' + f'ne{n_epochs:03}_' + f'depth{depth:02}_' + f'out{n_out_channels:02}_' + train_suffix


    output_dir = os.path.join(train_dir, model_name)

    if contrastive_flag:
        # train_generator = ContrastiveLearningDataset(os.path.join(train_dir, "train"))
        # train_dataset = train_generator.get_dataset('custom', 2)
        train_dataset = MyCustomDataset(root=os.path.join(train_dir, "train"),
                                        transform=ContrastiveLearningViewGenerator(
                                                 ContrastiveLearningDataset.get_simclr_pipeline_transform(),#(96),
                                                 2),
                                        return_name=True
                                        )

        # eval_generator = ContrastiveLearningDataset(os.path.join(train_dir, "eval"))
        # eval_dataset = eval_generator.get_dataset('custom', 2)

        eval_dataset = MyCustomDataset(root=os.path.join(train_dir, "eval"),
                                        transform=ContrastiveLearningViewGenerator(
                                            ContrastiveLearningDataset.get_simclr_pipeline_transform(),  # (96),
                                            2),
                                        return_name=True
                                        )

    else:
        train_dataset = MyCustomDataset(
            root=os.path.join(train_dir, "train"),
            transform=transform,
            return_name=True
        )

        eval_dataset = MyCustomDataset(
            root=os.path.join(train_dir, "eval"),
            transform=transform,
            return_name=True
        )

    config = BaseTrainerVerboseConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_epochs=n_epochs,  # Change this to train the model a bit more
        steps_saving=10,
    )

    if contrastive_flag:
        model_config = MetricVAEConfig(
            input_dim=input_dim,
            latent_dim=n_latent,
            zn_frac=0.1,
            orth_flag=orth_flag,
            temperature=nt_xent_temperature,
            n_conv_layers=depth,
            n_out_channels=n_out_channels,
            distance_metric=distance_metric,
            class_key_path=os.path.join(metadata_path, "class_key.csv"),
            class_ignorance_flag=class_ignorance_flag,
            time_ignorance_flag=time_ignorance_flag
        )
    else:
        model_config = VAEConfig(
            input_dim=input_dim,
            latent_dim=n_latent
        )

    encoder = Encoder_Conv_VAE(model_config) # these are custom classes I wrote for this use case
    # if matched_decoder_flag:
    decoder = Decoder_Conv_VAE(encoder)
    # else:
    #     decoder = Decoder_Conv_AE_FLEX(encoder)

    model = MetricVAE(
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

    root = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\"
    train_folder = "20231106_ds"
    train_suffix = "class_ignorance_test_real"

    contrastive_flag = True
    class_ignorance_flag = True
    time_ignorance_flag = True
    orth_flag = True

    temperature_vec = [0.0001, 0.001, 100, 0.01]
    train_dir = os.path.join(root, "training_data", train_folder)
    batch_size = 64
    n_epochs = 5
    z_dim_vec = [100]

    depth_vec = [5]
    distance_metric = "euclidean"
    max_tries = 3

    input_dim = (1, 288, 128)

    for z in z_dim_vec:
        for d in depth_vec:
            for t in temperature_vec:
                iter_flag = 0

                output_dir = train_metric_vae(root, train_folder, train_suffix=train_suffix, n_latent=z, batch_size=batch_size, n_epochs=n_epochs,
                                              nt_xent_temperature=t, learning_rate=1e-4, depth=d, contrastive_flag=contrastive_flag,
                                              orth_flag=orth_flag, distance_metric=distance_metric,
                                              class_ignorance_flag=class_ignorance_flag, time_ignorance_flag=time_ignorance_flag,
                                              input_dim=input_dim
                                              )
                        # iter_flag = max_tries
                    # except:
                    #     iter_flag += 1




