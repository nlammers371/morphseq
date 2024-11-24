library(monocle3)
library(hooke)
library(dplyr)

lmx_path_01 = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/sci-PLEX/processed_sci_data/20230830_lmx1b_projected_comb_filt_cds"
lmx_path_02 = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/sci-PLEX/processed_sci_data/20231207_lmx1b_projected_comb_cds"

lmx_cds01 = load_monocle_objects(lmx_path_01)
lmx_cds02 = load_monocle_objects(lmx_path_02)

colData(lmx_cds02)$hash_plate = "P01"
colData(lmx_cds02)$embryo <- gsub('P18', 'P01', colData(lmx_cds02)$embryo)

lmx_cds_master = monocle3:::combine_cds_for_maddy(list(lmx_cds01, lmx_cds02),  keep_reduced_dims=T, sample_col_name="sample_id")

lmx_master_path = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/sci-PLEX/processed_sci_data/lmx1b_combined"

save_monocle_objects(cds=lmx_cds_master, directory_path=lmx_master_path, comment='Combines the two lmx1b datasets from 20230830 and 20231207, respectively')

# colData(lmx_cds_master)$sample_id = colData(lmx_cds_master)$sample 

ccs_lmx1b = new_cell_count_set(lmx_cds_master, sample_group = "embryo", cell_group = "cell_type")
sparse_counts_lmx = counts(ccs_lmx1b)
matrix_counts_lmx = as.matrix(sparse_counts_lmx)

lmx_analysis_path = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/sci-PLEX/processed_sci_data/lmx1b_combined_analysis_v2"
dir.create(lmx_analysis_path)
write.csv(matrix_counts_lmx, file.path(lmx_analysis_path, "lmx_comb_cell_counts.csv"))

main_model_str = "~ "
nuisance_model_str = "~ sample_id"

ccm_lmx  = new_cell_count_model(ccs_lmx1b, main_model_formula_str=main_model_str, nuisance_model_formula_str=nuisance_model_str, num_threads=4)

# extract key model outputs
best_model = ccm_lmx@best
latent_matrix = best_model$latent

saveRDS(x=matrix_counts_lmx, file=file.path(lmx_analysis_path, "best_lmx_model.rds"))
write.csv(latent_matrix, file.path(lmx_analysis_path, "best_lmx_model_latent_counts.csv"))
