library(monocle3)
library(plotly)
library(BPCells)
library(dplyr)
library(hooke)
library(tidyr)
library(splines)

# temp dir to store BP cell files
temp_path <- "/net/trapnell/vol1/home/nlammers/tmp_files/nobackup/"

ccs_dir <- "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/sci-PLEX/ccs_data_cell_type_broad/"
dir.create(ccs_dir, showWarnings = FALSE, recursive = TRUE) 

################
# make output dir
model_dir <- "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/sci-PLEX/HF_hooke_regressions/"
dir.create(model_dir, showWarnings = FALSE, recursive = TRUE) 

################
# iterate through list of cds files (bead experiments only)
seahub_root <- "/net/seahub_zfish/vol1/data/annotated/v2.2.0/"

print("Adding HF2 data...")
########
# new hotfish
# root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/"
hot_path <- "/net/trapnell/vol1/home/hklee206/sci_3lvl/240813_hotfish2_run3_novaseq/hotfish2/hotfish2_projected_cds_v2.2.0"
hot_cds = load_monocle_objects(hot_path, matrix_control = list(matrix_class="BPCells", matrix_path=temp_path))

hot_col_data <- as.data.frame(colData(hot_cds)) 
valid_indices <- which(!is.na(hot_col_data$cell_type))
hot_col_data <- hot_col_data[valid_indices, ]

# load coarse cell type labels
ct_broad = read.csv("/net/trapnell/vol1/home/elizab9/projects/projects/CHEMFISH/resources/unique_ct_full.csv")
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

# save
# all_counts <- as.data.frame(as.matrix(counts(ccs)))

# # extract and save reduced version of metadata
# all_col_data <- as.data.frame(colData(ccs))

print("Setting up Hooke regression(s)...")
# infer WT developmental kinetics
# start_time <- min(ccs$timepoint)
# stop_time <- max(ccs$timepoint)
# num_spline_breaks <- 5
# spline_names <- lapply(seq_len(num_spline_breaks-1), function(x) paste0("t_spline_", x))
# time_formula = build_interval_formula(ccs, num_breaks = num_spline_breaks, interval_start = start_time, interval_stop = stop_time)

temperature_list = sort(unique(colData(hot_cds)$temp))
# mdl_name_list <- c("bead_expt_linear", "bead_expt_inter")
# formula_list <- c(paste0(time_formula, " + expt"),
#                   paste0(time_formula, " * expt"))

# initialize the progress bar (style = 3 shows a percentage bar)
# pb <- txtProgressBar(min = 0, max = length(formula_list), style = 3)
######
# mdl 1
for (m in seq_along(temperature_list)) {

    # formula_string <- formula_list[m]
    mdl_name <- paste0(as.character(temperature_list[m]), "C")
    print(mdl_name)
    # filter for ctrl temps in hotfish
    cds_temp <- hot_cds[, colData(hot_cds)$temp == temperature_list[m]]


    print("Making ccs structures...")
    ccs_temp <- new_cell_count_set(cds_temp, 
                            sample_group = "embryo_ID", 
                            cell_group = "cell_type_broad")
# var_list <- paste(c(covariate_vars), collapse = ", ")

    # different model versions
    # model_frame <- model.frame(as.formula(formula_string), all_col_data)
    # unique_levels <- lapply(all_col_data[covariate_vars], function(x) sort(unique(x)))
    # unique_combinations <- expand.grid(unique_levels, stringsAsFactors = FALSE)

    # save key model info
    mdl_dir <- file.path(model_dir, mdl_name)
    dir.create(mdl_dir, showWarnings = FALSE, recursive = TRUE) 
    # writeLines(formula_string, file.path(mdl_dir, "model_string.txt"))

    # write.csv(all_col_data, file.path(mdl_dir, "mdl_embryo_metadata.csv"))
    # write.csv(all_counts, file.path(mdl_dir, "mdl_counts_table.csv"))

    # no nuisance variables
    ccm <- new_cell_count_model(ccs_temp,
                                main_model_formula_str = "~ timepoint",
                                covariance_type="full",
                                num_threads=4)

    # save model predictions for all variable combos
    print("Saving Hooke predictions...")


    print("Saving model data structures...")
    # extract model and covariates
    best_full_model <- ccm@best_full_model

    # saveRDS(ccm, file = file.path(mdl_dir, "best_full_model.rds"))
    saveRDS(best_full_model, file = file.path(mdl_dir, "best_full_model.rds"))
    write.csv(best_full_model$latent, file = file.path(mdl_dir, "latents.csv"))
    write.csv(best_full_model$latent_pos, file = file.path(mdl_dir, "latents_pos.csv"))
    write.csv(as.data.frame(as.matrix(best_full_model$model_par$Sigma)), file = file.path(mdl_dir, "COV.csv"))
    write.csv(as.data.frame(as.matrix(best_full_model$model_par$Omega)), file = file.path(mdl_dir, "Omega.csv"))
    write.csv(as.data.frame(best_full_model$model_par$B), file = file.path(mdl_dir, "B.csv"))
    write.csv(as.data.frame(best_full_model$model_par$Theta), file = file.path(mdl_dir, "Theta.csv"))

}

# # linear offsets for disociation and experiment
# wt_expt_batch_ccm = new_cell_count_model(ccs,
#                                    main_model_formula_str = paste0(time_formula, "+ dis_protocol + expt"), 
#                                    covariance_type = "diagonal", #"full",
#                                    num_threads=4)

# m2_dir = file.path(model_dir, "wt_lin")
# dir.create(m2_dir, showWarnings = FALSE, recursive = TRUE) 
# saveRDS(wt_expt_batch_ccm, file = file.path(m2_dir, "best_full_model.rds"))
# write.csv(wt_expt_batch_ccm$latent, file = file.path(m2_dir, "latents.csv"))
# write.csv(wt_expt_batch_ccm$latent_pos, file = file.path(m2_dir, "zi.csv"))
# write.csv(as.data.frame(wt_expt_batch_ccm$model_par$Sigma), file = file.path(m2_dir, "COV.csv"))
# write.csv(as.data.frame(wt_expt_batch_ccm$model_par$Omega), file = file.path(m2_dir, "Omega.csv"))
# write.csv(as.data.frame(wt_expt_batch_ccm$model_par$B), file = file.path(m2_dir, "B.csv"))
# write.csv(as.data.frame(wt_expt_batch_ccm$model_par$Theta), file = file.path(m2_dir, "Theta.csv"))

# # linear offsets for experiment and interaction terms for disociation 
# wt_expt_batch_inter_ccm = new_cell_count_model(ccs,
#                                    main_model_formula_str = paste0(time_formula, "*dis_protocol + expt"), 
#                                    covariance_type="diagonal", #"full",
#                                    num_threads=4)

# m3_dir = file.path(model_dir, "wt_inter")
# dir.create(m3_dir, showWarnings = FALSE, recursive = TRUE) 
# saveRDS(wt_expt_batch_inter_ccm, file = file.path(m3_dir, "best_full_model.rds"))
# write.csv(wt_expt_batch_inter_ccm$latent, file = file.path(m3_dir, "latents.csv"))
# write.csv(wt_expt_batch_inter_ccm$latent_pos, file = file.path(m3_dir, "zi.csv"))
# write.csv(as.data.frame(wt_expt_batch_inter_ccm$model_par$Sigma) file = file.path(m3_dir, "COV.csv"))
# write.csv(as.data.frame(wt_expt_batch_inter_ccm$model_par$Omega) file = file.path(m3_dir, "Omega.csv"))
# write.csv(as.data.frame(wt_expt_batch_inter_ccm$model_par$B) file = file.path(m3_dir, "B.csv"))
# write.csv(as.data.frame(wt_expt_batch_inter_ccm$model_par$Theta) file = file.path(m3_dir, "Theta.csv"))





print("Done.")                                                                        

###########

