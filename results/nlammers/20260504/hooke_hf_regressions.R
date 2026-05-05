library(monocle3)
library(BPCells)
library(dplyr)
library(hooke)

# ---- Paths ------------------------------------------------------------------
temp_path     <- "/net/trapnell/vol1/home/nlammers/tmp_files/nobackup/"
model_dir     <- "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/sci-PLEX/20260504/HF_hooke_regressions/"
hot_path      <- "/net/trapnell/vol1/home/hklee206/sci_3lvl/240813_hotfish2_run3_novaseq/hotfish2/hotfish2_projected_cds_v2.2.0"
ct_broad_path <- "/net/trapnell/vol1/home/elizab9/projects/projects/CHEMFISH/resources/unique_ct_full.csv"

dir.create(model_dir, showWarnings = FALSE, recursive = TRUE)

# ---- Load CDS ---------------------------------------------------------------
print("Loading HF2 data...")
hot_cds <- load_monocle_objects(
    hot_path,
    matrix_control = list(matrix_class = "BPCells", matrix_path = temp_path)
)

# Drop cells with NA cell_type
keep    <- !is.na(colData(hot_cds)$cell_type)
hot_cds <- hot_cds[, keep]

# ---- Attach coarse cell-type labels -----------------------------------------
# For each fine cell_type, pick the most common cell_type_broad label
ct_broad <- read.csv(ct_broad_path)
ct_broad_filt <- as.data.frame(ct_broad) %>%
    count(cell_type, cell_type_broad) %>%
    group_by(cell_type) %>%
    slice_max(n, with_ties = FALSE) %>%
    ungroup() %>%
    select(cell_type, cell_type_broad)

# Map by name rather than by row order
colData(hot_cds)$cell_type_broad <- ct_broad_filt$cell_type_broad[
    match(colData(hot_cds)$cell_type, ct_broad_filt$cell_type)
]

# ---- Hooke regressions, one per temperature ---------------------------------
print("Setting up Hooke regression(s)...")
temperature_list <- sort(unique(colData(hot_cds)$temp))

for (m in seq_along(temperature_list)) {

    mdl_name <- paste0(as.character(temperature_list[m]), "C")
    print(mdl_name)

    cds_temp <- hot_cds[, colData(hot_cds)$temp == temperature_list[m]]

    print("Building cell count set...")
    ccs_temp <- new_cell_count_set(
        cds_temp,
        sample_group = "embryo_ID",
        cell_group   = "cell_type_broad"
    )

    mdl_dir <- file.path(model_dir, mdl_name)
    dir.create(mdl_dir, showWarnings = FALSE, recursive = TRUE)

    # Fit model (no nuisance variables)
    ccm <- new_cell_count_model(
        ccs_temp,
        main_model_formula_str = "~ timepoint",
        covariance_type        = "spherical",
        num_threads            = 4
    )

    print("Saving model outputs...")
    best_full_model <- ccm@best_full_model

    saveRDS(best_full_model, file = file.path(mdl_dir, "best_full_model.rds"))

    write.csv(best_full_model$latent,
              file = file.path(mdl_dir, "latents.csv"))
    write.csv(best_full_model$latent_pos,
              file = file.path(mdl_dir, "latents_pos.csv"))
    write.csv(as.data.frame(as.matrix(best_full_model$model_par$Sigma)),
              file = file.path(mdl_dir, "COV.csv"))
    write.csv(as.data.frame(as.matrix(best_full_model$model_par$Omega)),
              file = file.path(mdl_dir, "Omega.csv"))
    write.csv(as.data.frame(best_full_model$model_par$B),
              file = file.path(mdl_dir, "B.csv"))
    write.csv(as.data.frame(best_full_model$model_par$Theta),
              file = file.path(mdl_dir, "Theta.csv"))
}

print("Done.")