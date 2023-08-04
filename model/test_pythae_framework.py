from functions.pythae_utils import *
import os
from pythae.models import VAE, VAEConfig
from pythae.trainers import BaseTrainerConfig
from pythae.pipelines.training import TrainingPipeline
from pythae.models import AutoModel
import matplotlib.pyplot as plt


# root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"
root = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\"
train_name = "20230804_vae_test"
train_dir = os.path.join(root, "training_data", train_name)

train_dataset = MyCustomDataset(
    root=os.path.join(train_dir, "train"),
    transform=data_transform,
)

eval_dataset = MyCustomDataset(
    root=os.path.join(train_dir, "eval"),
    transform=data_transform
)

config = BaseTrainerConfig(
    output_dir='vae_test_' + train_name,
    learning_rate=1e-3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_epochs=5,  # Change this to train the model a bit more
)

model_config = VAEConfig(
    input_dim=(1, 576, 256),
    latent_dim=16
)

model = VAE(
    model_config=model_config
)

pipeline = TrainingPipeline(
    training_config=config,
    model=model
)

model = VAE(
    model_config=model_config
)

pipeline(
    train_data=train_dataset, # here we use the custom train dataset
    eval_data=eval_dataset # here we use the custom eval dataset
)

last_training = sorted(os.listdir('vae_test_' + train_name))[-1]
trained_model = AutoModel.load_from_folder(os.path.join('vae_test_' + train_name, last_training, 'final_model'))

from pythae.samplers import NormalSampler

# create normal sampler
normal_samper = NormalSampler(
    model=trained_model
)

# sample
gen_data = normal_samper.sample(
    num_samples=25
)

# show results with normal sampler
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))

for i in range(3):
    for j in range(3):
        axes[i][j].imshow(gen_data[i*2 +j].cpu().squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout(pad=0.)

plt.show()