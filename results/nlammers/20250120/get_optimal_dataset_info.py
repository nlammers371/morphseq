import pandas as pd

merged_df = pd.read_csv("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/20241216/sweep_analysis/paired_models_and_metrics_df.csv")
merged_df_avg = merged_df[merged_df["Perturbation"]=="avg_pert"]
splines_final_df = pd.read_csv("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/20241216/sweep_analysis/splines_final_df.csv")
prev_embryo_df = pd.read_csv("/net/trapnell/vol1/home/mdcolon/proj/fishcaster/data/embryo_morph_df.csv")
model_index = 74 #choosen best morphology space
path = merged_df_avg[merged_df_avg["model_index"] == model_index]["embryo_df_path_nohld"].iloc[0]
df_orig = pd.read_csv(path)