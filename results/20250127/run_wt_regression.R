library(monocle3)
library(plotly)
library(BPCells)
library(dplyr)
library(hooke)
library(tidyr)
library(spline)

# temp dir to store BP cell files
temp_path <- "/net/trapnell/vol1/home/nlammers/tmp_files/nobackup/"

# make output dir
model_dir <- "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/sci-PLEX/hooke_model_files/"
dir.create(model_dir, showWarnings = FALSE, recursive = TRUE) 

############
# load cds files
############
print("Loading cds objects...")
# hotfish2
hot_path <- "/net/trapnell/vol1/home/hklee206/sci_3lvl/240813_hotfish2_run3_novaseq/hotfish2/hotfish2_projected_cds_v2.2.0"
hot_cds = load_monocle_objects(hot_path, matrix_control = list(matrix_class="BPCells", matrix_path=temp_path))

# REF orig
ref_path <- "/net/seahub_zfish/vol1/data/reference_cds/v2.2.0/reference_cds_fixed_colData_updated_241203/"
ref_cds = load_monocle_objects(ref_path, matrix_control = list(matrix_class="BPCells", matrix_path=temp_path))
ref_cds <- ref_cds[, !is.na(colData(ref_cds)$timepoint)]

# REF 1
ref1_path <- "/net/seahub_zfish/vol1/data/annotated/v2.2.0/REF1/REF1_projected_cds_v2.2.0"
ref1_cds = load_monocle_objects(ref1_path, matrix_control = list(matrix_class="BPCells", matrix_path=temp_path))

# REF 2
ref2_path <- "/net/seahub_zfish/vol1/data/annotated/v2.2.0/REF2/REF2_projected_cds_v2.2.0"
ref2_cds = load_monocle_objects(ref2_path, matrix_control = list(matrix_class="BPCells", matrix_path=temp_path))

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

# filter for ctrl temps in hotfish
hot_cds_28 = hot_cds[, colData(hot_cds)$temp == 28]


#############
# make ccs structures
master_cds <- combine_cds(list(hot_cds_28, ref_cds, ref1_cds, ref2_cds), 
                            keep_all_genes=FALSE, 
                            keep_reduced_dims=TRUE) # required to prevent error when running ccm

print("Making ccs structures...")
ccs <- new_cell_count_set(master_cds, 
                         sample_group = "embryo_ID", 
                         cell_group = "cell_type_broad")

ccs = ccs[, !is.na(ccs$timepoint)]

# infer WT developmental kinetics
start_time = 12
stop_time = 72
time_formula = build_interval_formula(ccs, num_breaks = 3, interval_start = start_time, interval_stop = stop_time)


# main_string = paste0(time_formula, batch_var_string)

wt_expt_naive_ccm = new_cell_count_model(ccs,
                                   main_model_formula_str = time_formula,
                                #    nuisance_model_formula_str = batch_var_string0, 
                                covariance_type="full",
                                   num_threads=4)#,
                                #    verbose = TRUE)

wt_expt_batch_ccm = new_cell_count_model(ccs,
                                   main_model_formula_str = paste0(time_formula, "+ dis_protocol + expt + fix_protocol + strain"), 
                                   covariance_type="full",
                                   num_threads=4)#

# saveRDS(wt_expt_naive_ccm, file.path(model_dir, "wt_time_naive_ccm.rds"))

# hot_ccs = new_cell_count_set(hot_cds_28, 
#                          sample_group = "embryo_ID", 
#                          cell_group = "cell_type_broad")

# ref_ccs = new_cell_count_set(ref_cds, 
#                          sample_group = "embryo_ID", 
#                          cell_group = "cell_type_broad")

# ref1_ccs = new_cell_count_set(ref1_cds, 
#                          sample_group = "embryo_ID", 
#                          cell_group = "cell_type_broad")

# ref2_ccs = new_cell_count_set(ref2_cds,  
#                          sample_group = "embryo_ID", 
#                          cell_group = "cell_type_broad") 

print("Done.")                                                                        

###########

