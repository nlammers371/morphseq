from functions.pythae_utils import *
import os
from pythae.models import VAE, VAEConfig
from pythae.trainers import BaseTrainerConfig
from pythae.pipelines.training import TrainingPipeline
from pythae.models import AutoModel
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from pythae.samplers import NormalSampler


if __name__ == "__main__":

    root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"
    # root = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\"
    train_name = "20230807_vae_test"
    train_dir = os.path.join(root, "training_data", train_name)

    # n_latent = 5
    # batch_size = 32
    # n_epochs = 15
    # model_name = f'_z{n_latent:02}_' + f'bs{batch_size:03}_' + f'ne{n_epochs:03}'
    output_dir = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/training_data/20230807_vae_test/20230807_vae_test_z10_bs006_ne015"

    main_dims = (128, 128)
    data_transform = make_dynamic_rs_transform(main_dims)

    test_data = MyCustomDataset(
        root=os.path.join(train_dir, "test"),
        transform=data_transform
    )

    last_training = sorted(os.listdir(output_dir))[-1]
    trained_model = AutoModel.load_from_folder(
        os.path.join(output_dir, last_training, 'final_model'))

    ############
    # Question 1: how well does it reproduce test images?
    ############



    # test_data_loader = DataLoader(
    #         dataset=test_data,
    #         batch_size=1,
    #         num_workers=1,
    #         shuffle=True,
    #         sampler=None,
    #         collate_fn=None,
    #     )

    i_test = 701
    im_test = torch.reshape(np.asarray(test_data[i_test]).tolist()[0], (1, 1, main_dims[0], main_dims[1]))
    reconstructions = trained_model.reconstruct(im_test).detach().cpu()

    # show results with normal sampler
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

    axes[0].imshow(np.squeeze(im_test), cmap='gray')
    axes[0].axis('off')

    axes[1].imshow(np.squeeze(reconstructions), cmap='gray')
    axes[1].axis('off')

    plt.tight_layout(pad=0.)

    plt.show()