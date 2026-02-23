library(ggplot2)
library(monocle3)
library(tidyverse)
library(htmlwidgets)

# source("~/OneDrive/UW/Trapnell/hooke_manuscript/supplement/filter_doublets/filter_doublet_utils.R")

setwd("~/OneDrive/UW/Trapnell/hooke_manuscript/tmp/")


cds = readRDS("partition_11_ref_only_cds.rds")


plot_cells_3d(cds, color_cells_by = "cell_type_sub")

cds = cluster_cells(cds)

colData(cds)$cluster_res = clusters(cds)

# how many clusters you have 
unique(clusters(cds))

plot_cells_3d(cds, color_cells_by = "cluster_res")

colData(cds)$timepoint = as.numeric(colData(cds)$timepoint)
plot_cells_3d(cds, color_cells_by = "timepoint")


# how to pseudobulk 


marker_test_res <- top_markers(cds, 
                               group_cells_by="cluster_res", 
                               reference_cells=1000, cores=8)


marker_test_res %>%
  filter(fraction_expressing >= 0.10, mean_expression >=0.10) %>%
  group_by(cell_group) %>%
  top_n(10, specificity)

plot_cells(cds, genes = c("qkib", "slc45a2"))


# how to fix confetti 
cell_types_to_na = colData(cds) %>% as.data.frame() %>% 
  group_by(cell_type_sub) %>% tally() %>% 
  arrange(-n) %>% 
  filter(n < 10) %>% 
  pull(cell_type_sub)

colData(cds)$cell_type_sub  = ifelse(colData(cds)$cell_type_sub %in% cell_types_to_na, NA_character_, colData(cds)$cell_type_sub)

cds = make_cds_nn_index(cds, "UMAP")
cds = fix_missing_cell_labels(cds, 
                              reduction_method = "UMAP",
                              from_column_name = "cell_type_sub",
                              to_column_name = "cell_type_sub",
                              k=10)




colData(cds)$cell_type_sub_ann_party = colData(cds) %>% as.data.frame %>% 
                                        mutate(cell_type_sub_ann_party = case_when(
                                          cluster_res == "1" ~ "cell_type_1", 
                                          cluster_res == "2" ~ "cell_type_2"
                                          
                                        )) %>% pull(cell_type_sub_ann_party)