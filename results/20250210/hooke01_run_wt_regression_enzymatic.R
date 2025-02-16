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
# iterate through list of cds files (bead experiments only)
seahub_root <- "/net/seahub_zfish/vol1/data/annotated/v2.2.0/"

cds_name_list <- c("CHEM1.0", "CHEM1.1","CHEM9", "HF", "LMX1B")

# dewsignate list of control labels
ctrl_labels <- c("EtOH", "DMSO", "ctrl-inj", "reference", "ctrl-uninj", "novehicle")

# load coarse cell type labels
ct_broad = read.csv("/net/trapnell/vol1/home/elizab9/projects/projects/CHEMFISH/resources/unique_ct_full.csv")
ct_broad_filt <- as.data.frame(ct_broad) %>%
                        count(cell_type, cell_type_broad) %>%
                        group_by(cell_type) %>%
                        slice_max(n, with_ties = FALSE) %>%
                        ungroup() %>%
                        select(cell_type, cell_type_broad)

################
# make output dir
model_dir <- "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/sci-PLEX/hooke_model_files/"
dir.create(model_dir, showWarnings = FALSE, recursive = TRUE) 
cds_list <- list()

print("Loading cds files...")
# Loop over the names in cds_name_list
for (i in seq_along(cds_name_list)) {
    cds_name <- cds_name_list[i]
    cds_path <- file.path(seahub_root, cds_name, paste0(cds_name, "_projected_cds_v2.2.0/"))
    
    cds <- load_monocle_objects(cds_path, matrix_control = list(matrix_class = "BPCells", matrix_path = temp_path))
    
    if (!("cell_type_broad" %in% colnames(colData(cds)))) {
        col_data <- as.data.frame(colData(cds))
        col_data <- col_data %>% 
                    left_join(ct_broad_filt, by = "cell_type")
        colData(cds)$cell_type_broad <- col_data$cell_type_broad
    }
    
    cds_filtered <- cds[, colData(cds)$perturbation %in% ctrl_labels]

    # Save the cds object in the list using its name as key
    cds_list[[cds_name]] <- cds_filtered
}

print("Done.")

print("Adding original reference...")
########################
# REF orig
ref_path <- "/net/seahub_zfish/vol1/data/reference_cds/v2.2.0/reference_cds_fixed_colData_updated_241203/"
ref_cds = load_monocle_objects(ref_path, matrix_control = list(matrix_class="BPCells", matrix_path=temp_path))

cds_list[["REF"]] <- ref_cds

#############
# make ccs structures
print("Making combined cds...")
master_cds <- combine_cds(cds_list, 
                            keep_all_genes=FALSE, 
                            keep_reduced_dims=TRUE)   # required to prevent error when running ccm


print("Making ccs structures...")
ccs <- new_cell_count_set(master_cds, 
                         sample_group = "embryo_ID", 
                         cell_group = "cell_type_broad")

ccs <- ccs[, !is.na(ccs$timepoint)]

# Create a new column 'perturbation_collapsed'
colData(ccs)$pert_collapsed <- ifelse(
  colData(ccs)$perturbation %in% c("ctrl-uninj", "reference", "novehicle"),
  "ctrl",
  colData(ccs)$perturbation
)

# save
all_counts <- as.data.frame(as.matrix(counts(ccs)))

# # extract and save reduced version of metadata
all_col_data <- as.data.frame(colData(ccs))


print("Setting up Hooke regression(s)...")
# infer WT developmental kinetics
start_time <- min(ccs$timepoint)
stop_time <- max(ccs$timepoint)
num_spline_breaks <- 5
spline_names <- lapply(seq_len(num_spline_breaks-1), function(x) paste0("t_spline_", x))
time_formula = build_interval_formula(ccs, num_breaks = num_spline_breaks, interval_start = start_time, interval_stop = stop_time)

mdl_name_list <- c("enz_expt_linear", "enz_expt_inter")
formula_list <- c(paste0(time_formula, " + expt"),
                  paste0(time_formula, " * expt"))

# initialize the progress bar (style = 3 shows a percentage bar)
# pb <- txtProgressBar(min = 0, max = length(formula_list), style = 3)
######
# mdl 1
for (m in seq_along(formula_list)) {

    # setTxtProgressBar(pb, m-1)

    formula_string <- formula_list[m]
    mdl_name <- mdl_name_list[m]

    # Convert string to a formula object
    fml <- as.formula(formula_string)

    # Check the length of the formula object.
    # For a one-sided formula, length(fml) == 2.
    if(length(fml) == 2) {
      covariate_vars <- all.vars(fml[[2]])
    } else {
      covariate_vars <- all.vars(fml[[3]])
    }
# var_list <- paste(c(covariate_vars), collapse = ", ")

    # different model versions
    # model_frame <- model.frame(as.formula(formula_string), all_col_data)
    unique_levels <- lapply(all_col_data[covariate_vars], function(x) sort(unique(x)))
    unique_combinations <- expand.grid(unique_levels, stringsAsFactors = FALSE)

    # save key model info
    mdl_dir <- file.path(model_dir, mdl_name)
    dir.create(mdl_dir, showWarnings = FALSE, recursive = TRUE) 
    writeLines(formula_string, file.path(mdl_dir, "model_string.txt"))

    write.csv(all_col_data, file.path(mdl_dir, "mdl_embryo_metadata.csv"))
    write.csv(all_counts, file.path(mdl_dir, "mdl_counts_table.csv"))

    # no nuisance variables
    ccm <- new_cell_count_model(ccs,
                                main_model_formula_str = formula_string,
                                covariance_type="diagonal", #"full",
                                num_threads=4)

    # generate and save lookup table of time splines

    # Create a fine grid of time values over the entire range.
    t_grid <- seq(start_time, stop_time, length.out = 1000)
    knots_str <- sub(".*knots\\s*=\\s*c\\(([^)]+)\\).*", "\\1", time_formula)
    interior_knots <- c(as.numeric(unlist(strsplit(knots_str, ","))))
    boundary_knots <- c(start_time, stop_time)
    B <- ns(t_grid, knots = interior_knots, Boundary.knots = boundary_knots)

    # Combine the time values and the basis matrix into a data frame.
    lookup_df <- data.frame(timepoint = t_grid, as.data.frame(B))
    col_names_vec <- c("timepoint", spline_names)
    colnames(lookup_df) = col_names_vec
    # Save the lookup table as a CSV file.
    write.csv(lookup_df, file = file.path(mdl_dir, "time_splines.csv") , row.names = FALSE)

    # save model predictions for all variable combos
    print("Saving Hooke predictions...")

    # Initialize an empty list to store the results, along with the combination values
    results_list <- vector("list", nrow(unique_combinations))
    # Loop over each combination and call estimate_abundances
    for(i in seq_len(nrow(unique_combinations))) {
      # Extract the current combination as a tibble (one-row data frame)
      newdata <- unique_combinations[i, , drop = FALSE]
      # colnames(newdata)[1] = "timepoint"
      # Call the estimate_abundances function; assume it returns a data frame or a list of values.
      est <- estimate_abundances(ccm, newdata)
      results_list[[i]] <- est
    }
    combined_results <- bind_rows(results_list)
    write.csv(combined_results, file = file.path(mdl_dir, "abundance_estimates.csv"))

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

