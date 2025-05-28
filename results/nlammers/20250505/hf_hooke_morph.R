rm(list = ls())

library(msigdbr)
library(monocle3)
library(hooke)
library(dplyr)
library(ggplot2)
library(tidyverse)


# load cds
hot_path <- "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/seq_data/cds/hotfish2_projected_cds_v2.2.0"
temp_path = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/seq_data/cds/temp_files"
hot_cds = load_monocle_objects(hot_path, matrix_control = list(matrix_class="BPCells", matrix_path=temp_path))

# load morph dataset
morph_df_path <- "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/results/20250312/morph_latent_space/hf_pca_morph_df_hooke.csv"
morph_df <- read_csv(morph_df_path) %>% 
                    as.data.frame() %>%
                    select("sample", "morph_dist_spline", "morph_branch_flag")

temperature_list <- c(34, 35)
hot_cds_sub <- hot_cds[, colData(hot_cds)$temp %in% temperature_list]

# generate ccs 
ccs <- new_cell_count_set(hot_cds_sub, 
                               sample_group = "embryo_ID", 
                               cell_group = "cell_type")

# join on morph info 
cd <- as.data.frame(colData(ccs))

# find the rows in meta_df that match each cell’s sample
ix <- match(cd$sample, morph_df$sample)
cd$morph_branch_flag <- as.factor(morph_df$morph_branch_flag[ix])
cd$morph_dist_spline <- morph_df$morph_dist_spline[ix]

colData(ccs) <- DataFrame(cd)

ccs = ccs[, !is.na(colData(ccs)$morph_branch_flag)]

# no nuisance variables
ccm <- new_cell_count_model(ccs,
                            main_model_formula_str = "~ morph_branch_flag",
                            #covariance_type="full",
                            num_threads=4)


branched = estimate_abundances(ccm, tibble::tibble(morph_branch_flag = TRUE))
normal = estimate_abundances(ccm, tibble::tibble(morph_branch_flag = FALSE))

contrast_tbl = compare_abundances(ccm, branched, normal)
contrast_tbl <- contrast_tbl %>% select(cell_group, delta_log_abund, delta_log_abund_se, delta_q_value)


write_csv(contrast_tbl, "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/analyses/crossmodal/hotfish/hotfish_morph_ccs_contrast.csv")
