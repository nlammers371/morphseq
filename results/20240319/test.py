import json
path = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/training_data/20240312_test/VAE_z100_ne250_triplet_loss_test_SELF_and_OTHER/VAE_training_2024-03-19_08-49-34/final_model/"
f = open(path + "model_config.json")
model_config = json.load(f)
print("check")