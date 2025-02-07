library(monocle3)
library(plotly)
library(BPCells)
library(dplyr)
library(hooke)
library(tidyr)
library(PLNmodels)
library(splines)

# make output dir
model_dir <- "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/sci-PLEX/hooke_model_files/"
ccs_dir <- "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/sci-PLEX/ccs_data_cell_type_broad/"

ccs_meta <- read.csv(file.path(ccs_dir, "mdl_embryo_metadata.csv"), header = TRUE, row.names = 1)
ccs_counts <- read.csv(file.path(ccs_dir, "mdl_counts_table.csv"), header = TRUE, row.names = 1) 

# extract size factor
size_factor_vec <- setNames(ccs_meta[["Size_Factor"]], rownames(ccs_meta))

pln_data <- PLNmodels::prepare_data(counts = ccs_counts,
                                    covariates = ccs_meta %>% as.data.frame)
                                    # offset = size_factor_vec)

# get model predictions
m1_dir <- file.path(model_dir, "wt_naive")
mdl1 <- readRDS(file.path(m1_dir, "best_full_model.rds"))
mdl1_counts_pd <- as.data.frame(predict(mdl1, pln_data))
write.csv(mdl1_counts_pd, file.path(ccs_dir, "wt_naive_predictions.csv"))

m2_dir <- file.path(model_dir, "wt_lin")
mdl2 <- readRDS(file.path(m1_dir, "best_full_model.rds"))
mdl2_counts_pd <- as.data.frame(predict(mdl2, pln_data))
write.csv(mdl2_counts_pd, file.path(ccs_dir, "wt_lin_predictions.csv"))

m3_dir <- file.path(model_dir, "wt_inter")
mdl3 <- readRDS(file.path(m1_dir, "best_full_model.rds"))
mdl3_counts_pd <- as.data.frame(predict(mdl3, pln_data))
write.csv(mdl3_counts_pd, file.path(ccs_dir, "wt_inter_predictions.csv"))