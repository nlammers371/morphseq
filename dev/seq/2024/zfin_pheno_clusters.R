rm(list = ls())

library(tidyr)
library(dplyr)
library(monocle3)
#library(data.table)
library(googledrive)
library(future)
library(purrr)
library(ggplot2)
library(msigdbr)
library(ggtext)
library(Matrix)

# file path 
file_path = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/zfin/20240326/clean_zfin_single-mut_with-ids_phenotype_df.csv"
#drive_get("https://drive.google.com/file/d/1-05lHha1Z2dE7ufwC2AbZ_1A8GZHSriX") %>%
#  drive_download(overwrite = T)
built_dir =  "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/zfin/20240326/built_data/" 

sing_mut_df = read.csv(file_path, sep=",")   #data.table::fread("clean_zfin_single-mut_with-ids_phenotype_df.csv", sep = ",", data.table = F, stringsAsFactors = F)
stage_to_hpf_key = read.csv(paste0(built_dir, "stage_to_hpf_key.csv"), stringsAsFactors = F)

# remove genes that lead to phenotypes later than 72hpf 
sing_mut_df = sing_mut_df %>% 
  left_join(stage_to_hpf_key, by = "start_stage") %>% 
  filter(start_hpf <= 72) 


struct_df = sing_mut_df %>% 
  group_by(aff_struct_super_1) %>% 
  tally(name = "aff_count")

struct_df %>% 
  arrange(aff_count) %>% dim()

filt_structs = struct_df %>% 
  filter(aff_count > 4) %>% 
  filter(aff_struct_super_1 != "whole_organism") %>% # could raise this threshold
  pull(aff_struct_super_1)

sel_df = sing_mut_df %>%
  filter(aff_struct_super_1 %in% filt_structs & phen_tag == "abnormal") %>% 
  select(gene, aff_struct_super_1, val)

mat = reshape2::acast(sel_df, gene ~ aff_struct_super_1, fill = 0, value.var = "val")

meta_df = as.data.frame(rownames(mat))
colnames(meta_df) = c("gene")
meta_df = meta_df %>% 
  left_join(sing_mut_df %>% 
              group_by(gene) %>% 
              tally(name = "aff_structures"),
            by = "gene")

rownames(meta_df) = meta_df$gene

rowdat = as.data.frame(rownames(t(mat)))
colnames(rowdat) = c("gene_short_name")
rownames(rowdat) = rowdat$gene_short_name

phen_cds = new_cell_data_set(expression_data = t(mat), 
                             cell_metadata = meta_df, 
                             gene_metadata = rowdat)

phen_cds = preprocess_cds(phen_cds, method = "LSI") %>% 
  reduce_dimension(umap.n_neighbors = 5L, preprocess_method = "LSI", max_components=2) %>% 
  cluster_cells(resolution = 1e-3)
colData(phen_cds)$umap1 = reducedDim(x = phen_cds,
                                     type = "UMAP")[,1]
colData(phen_cds)$umap2 = reducedDim(x = phen_cds,
                                     type = "UMAP")[,2]
colData(phen_cds)$group = clusters(phen_cds)

plot_cells(phen_cds, cell_size = 1, 
           color_cells_by = "cluster", group_label_size = 4)


#####
# subset to TFs
gene_set_mf = msigdbr(species = "Danio rerio", subcategory = "GO:MF")

transcription_regulators = gene_set_mf %>%
  dplyr::select(gs_id, gene_symbol, gs_name) %>%
  dplyr::filter(grepl("Transcription", gs_name, ignore.case=TRUE)) %>%
  pull(gene_symbol) %>% unique %>% sort

sing_mut_TF_df = sing_mut_df %>% 
  filter(gene %in% transcription_regulators & aff_struct_super_1 %in% filt_structs) 

# use left join to merge colData(phen_cds) with sing_mut_TF_df
sing_mut_TF_df = sing_mut_TF_df %>% left_join(as.data.frame(colData(phen_cds)), by = "gene")

# generate condensed table
tf_df_short = sing_mut_TF_df %>% 
  select(gene, pub_id, gene_id, start_stage, end_stage, umap1, umap2, group) %>%
  distinct()

# make plot
ggplot(tf_df_short, aes(x=umap1, y=umap2)) + geom_point(aes(color=group))

# get top impacted structures by group
n_top = 5
top_structs = sing_mut_TF_df %>% 
  group_by(group, aff_struct_super_1) %>% 
  tally(name = "aff_count") %>% 
  arrange(group, desc(aff_count)) %>%
  dplyr::slice(1:n_top)

# make key of developmental stages
#stage_key = sing_mut_TF_df %>% 
#  select(start_stage) %>% 
#  distinct() %>% 
#  arrange(start_stage)

# save stage key to csv
# comment out block below

# join on UMAP info for full gene df
sing_mut_df = sing_mut_df %>% left_join(as.data.frame(colData(phen_cds)), by = "gene")

# save stuff!
if (!dir.exists(built_dir)) {
  dir.create(built_dir)
}

write.csv(top_structs, paste0(built_dir, "top_phenotypes_per_cluster.csv"), row.names = F)
write.csv(tf_df_short, paste0(built_dir, "tf_df_short.csv"), row.names = F)
write.csv(sing_mut_TF_df, paste0(built_dir, "sing_mut_TF_df.csv"), row.names = F)
write.csv(sing_mut_df, paste0(built_dir, "sing_mut_df.csv"), row.names = F)