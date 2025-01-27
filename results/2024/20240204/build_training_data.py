from src.build.build05_split_training_images import make_image_snips

root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
label_var = "experiment_date"

# everything but the morphseq datasets and the lmx1b data
train_name = "20240204_ds_v2"
test_dates = ["20230830", "20230831", "20231207", "20231208"]
test_perturbations = ["lmx1b"]
make_image_snips(root, train_name, label_var="experiment_date", frac_to_use=1.0, rs_factor=0.5, test_dates=test_dates, test_perturbations=test_perturbations, overwrite_flag=True)

# more restrictive case
train_name = "20240204_ds_v1"
test_dates = ["20230830", "20230831", "20231207", "20231208", "20231206"]
test_perturbations = ["lmx1b", "tbxta"]
make_image_snips(root, train_name, label_var="experiment_date", frac_to_use=1.0, rs_factor=0.5, test_dates=test_dates, test_perturbations=test_perturbations, overwrite_flag=True)