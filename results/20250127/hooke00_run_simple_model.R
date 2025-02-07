library(monocle3)
library(plotly)
library(BPCells)
library(dplyr)
library(hooke)
library(tidyr)

# temp dir to store BP cell files
temp_path <- "/net/trapnell/vol1/home/nlammers/tmp_files/nobackup/"

ccs_dir <- "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/sci-PLEX/ccs_data_test/"
dir.create(ccs_dir, showWarnings = FALSE, recursive = TRUE) 

# make output dir
model_dir <- "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/sci-PLEX/hooke_model_test/"
dir.create(model_dir, showWarnings = FALSE, recursive = TRUE) 

############
# load cds files
############
print("Loading cds objects...")
# hotfish2
hot_path <- "/net/trapnell/vol1/home/hklee206/sci_3lvl/240813_hotfish2_run3_novaseq/hotfish2/hotfish2_projected_cds_v2.2.0"
hot_cds = load_monocle_objects(hot_path, matrix_control = list(matrix_class="BPCells", matrix_path=temp_path))

print("Done.")

# load coarse cell type labels
ct_broad = read.csv("/net/trapnell/vol1/home/elizab9/projects/projects/CHEMFISH/resources/unique_ct_full.csv")

##################
# add broad cell type names to hotfish data
print("Adding broad cell type labels...")

hot_col_data <- as.data.frame(colData(hot_cds)) 
valid_indices <- which(!is.na(hot_col_data$cell_type))
hot_col_data <- hot_col_data[valid_indices, ]

# ct_broad_filt <- as.data.frame(ct_broad) %>% select("cell_type", "cell_type_broad") %>% drop_na(cell_type) %>% distinct("cell_type", .keep_all = TRUE)
ct_broad_filt <- as.data.frame(ct_broad) %>%
                        count(cell_type, cell_type_broad) %>%
                        group_by(cell_type) %>%
                        slice_max(n, with_ties = FALSE) %>%
                        ungroup() %>%
                        select(cell_type, cell_type_broad)

# Perform left join to add "cell_type_broad"
hot_col_data <- hot_col_data %>%
  left_join(ct_broad_filt, by = "cell_type")

# Convert back to DataFrame and assign back to CDS
hot_cds <- hot_cds[, valid_indices]
colData(hot_cds)$cell_type_broad <- hot_col_data$cell_type_broad

print("Making ccs structures...")
ccs <- new_cell_count_set(hot_cds, 
                         sample_group = "embryo_ID", 
                         cell_group = "cell_type_broad")

ccs <- ccs[, !is.na(ccs$timepoint)]

# save
all_counts <- as.data.frame(as.matrix(counts(ccs)))
write.csv(all_counts, file.path(ccs_dir, "mdl_counts_table.csv"))

# extract and save reduced version of metadata
all_col_data <- as.data.frame(colData(ccs))
write.csv(all_col_data, file.path(ccs_dir, "mdl_embryo_metadata.csv"))

# Run 3 different versions of WT dev regression
############
# no nuisance variables
mdl_string = "~ timepoint"

ccm <- new_cell_count_model(ccs,
                            main_model_formula_str = mdl_string,
                            covariance_type="diagonal", #"full",
                            num_threads=  4)

best_full_model <- ccm@best_full_model

# save key model info
m1_dir = file.path(model_dir, "hot_test")
dir.create(m1_dir, showWarnings = FALSE, recursive = TRUE) 

writeLines(model_string, con = "model_string.txt")

saveRDS(best_full_model, file = file.path(m1_dir, "best_full_model.rds"))
write.csv(best_full_model$latent, file = file.path(m1_dir, "latents.csv"))
write.csv(best_full_model$latent_pos, file = file.path(m1_dir, "zi.csv"))
write.csv(as.data.frame(as.matrix(best_full_model$model_par$Sigma)), file = file.path(m1_dir, "COV.csv"))
write.csv(as.data.frame(as.matrix(best_full_model$model_par$Omega)), file = file.path(m1_dir, "Omega.csv"))
write.csv(as.data.frame(best_full_model$model_par$B), file = file.path(m1_dir, "B.csv"))
write.csv(as.data.frame(best_full_model$model_par$Theta), file = file.path(m1_dir, "Theta.csv"))

# get predictions
pln_data <- PLNmodels::prepare_data(counts = all_counts,
                                    covariates = all_col_data %>% as.data.frame)
counts_pd <- as.data.frame(predict(mdl1, pln_data))

write.csv(mdl1_counts_pd, file.path(ccs_dir, "wt_naive_predictions.csv"))
print("Done.")                                                                        

###########

