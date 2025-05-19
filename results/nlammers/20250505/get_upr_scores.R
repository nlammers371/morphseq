rm(list = ls())

library(msigdbr)
library(monocle3)
library(hooke)
library(dplyr)
library(ggplot2)
library(tidyverse)

# define function
aggregate_gene_score = function(cds, gene_list=c("CCNB1", "CCNB2","CDK1"), name = "gene_list_score"){
  cds$Size_Factor = size_factors(cds)
  cds_gene_list_norm = cds[rowData(cds)$gene_short_name %in% gene_list,] 
  aggregate_marker_expression = counts(cds_gene_list_norm)
  aggregate_marker_expression = t(t(aggregate_marker_expression) / colData(cds_gene_list_norm)$Size_Factor)
  aggregate_marker_expression = Matrix::colSums(aggregate_marker_expression)
  aggregate_marker_expression = log(aggregate_marker_expression + 1)
  colData(cds)[[name]] = aggregate_marker_expression
  return(cds)
}

# pull gene set
gene_set_BP = msigdbr::msigdbr(species = "Danio rerio", subcategory = "GO:BP")
gene_set_hallmarks = msigdbr::msigdbr(species = "Danio rerio", category = "H")

gene_set_BP_df = gene_set_BP %>% 
  dplyr::distinct(gs_name, gene_short_name=gene_symbol) %>% 
  as.data.frame()

gene_set_hallmarks_df = gene_set_hallmarks %>% 
  dplyr::distinct(gs_name, gene_short_name=gene_symbol) %>% 
  as.data.frame()

upr_genes = gene_set_hallmarks_df %>%
  filter(grepl("UNFOLDED", gs_name)) %>%
  distinct(gene_short_name, .keep_all=T)


# load cds
hot_path <- "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/seq_data/cds/hotfish2_projected_cds_v2.2.0"
temp_path = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/seq_data/cds/temp_files"
hot_cds = load_monocle_objects(hot_path, matrix_control = list(matrix_class="BPCells", matrix_path=temp_path))

# calculate scores
hot_cds = aggregate_gene_score(hot_cds, upr_genes$gene_short_name, "upr_score")

# pull relevant cols
hotfish_upr_df = colData(hot_cds) %>%
                as.data.frame() %>%
                 select("cell", "Size_Factor", "embryo_ID", "timepoint",
                        "temp", "expt", "upr_score", "hash_plate", "hash_well",
                        "count_per_embryo", "mean_nn_time", "min_nn_dist", 
                        "max_nn_dist")


write_csv(hotfish_upr_df, "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/analyses/crossmodal/hotfish/hotfish_upr_scores.csv")

