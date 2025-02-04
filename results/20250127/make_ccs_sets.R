library(monocle3)
library(hooke)


# temp dir to store BP cell files
temp_path <- "/net/trapnell/vol1/home/nlammers/tmp_files/nobackup/"
# make output dir
ccs_dir <- "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/sci-PLEX/ccs_data/"
dir.create(ccs_dir, showWarnings = FALSE, recursive = TRUE) 

##### Now do same for hotfish experiment
# root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/"
hot_path <- "/net/trapnell/vol1/home/hklee206/sci_3lvl/240813_hotfish2_run3_novaseq/hotfish2/hotfish2_projected_cds_v2.2.0"
hot_cds = load_monocle_objects(hot_path, matrix_control = list(matrix_class="BPCells", matrix_path=temp_path))

# generate cell count set
hot_ccs = new_cell_count_set(hot_cds, 
                         sample_group = "embryo_ID", 
                         keep_cds=FALSE,
                         cell_group = "cell_type")

# get simple counts matrix
hot_counts = as.matrix(counts(hot_ccs))
write.csv(hot_counts, file.path(ccs_dir, "hot2_counts_table.csv"))

# extract and save reduced version of metadata
hot_col_data <- colData(hot_ccs)
write.csv(hot_col_data, file.path(ccs_dir, "hot2_embryo_metadata.csv"))


# save_monocle_objects(
#   cds = hot_ccs, 
#   directory_path = file.path(ccs_dir, "hot2_ccs.rds")
# )
#saveRDS(hot_ccs, file = file.path(ccs_dir, "hot2_ccs.rds"))

########################
# REF orig
ref_path <- "/net/seahub_zfish/vol1/data/reference_cds/v2.2.0/reference_cds_fixed_colData_updated_241203/"
ref_cds = load_monocle_objects(ref_path, matrix_control = list(matrix_class="BPCells", matrix_path=temp_path))

# generate cell count set
ref_ccs = new_cell_count_set(ref_cds, 
                         sample_group = "embryo_ID",
                         keep_cds = FALSE, 
                         cell_group = "cell_type")

# get simple counts matrix
ref_counts = as.matrix(counts(ref_ccs))
write.csv(ref_counts, file.path(ccs_dir, "ref_orig_counts_table.csv"))

# extract and save reduced version of metadata
ref_col_data <- colData(ref_ccs)
write.csv(ref_col_data, file.path(ccs_dir, "ref_orig_embryo_metadata.csv"))

# save_monocle_objects(
#   cds = ref_ccs, 
#   directory_path = file.path(ccs_dir, "ref_orig_ccs.rds")
# )
#saveRDS(ref_ccs, file = file.path(ccs_dir, "ref_orig_ccs.rds"))

#######################
# REF 1
ref1_path <- "/net/seahub_zfish/vol1/data/annotated/v2.2.0/REF1/REF1_projected_cds_v2.2.0"
ref1_cds = load_monocle_objects(ref1_path, matrix_control = list(matrix_class="BPCells", matrix_path=temp_path))

# generate cell count set
ref1_ccs = new_cell_count_set(ref1_cds, 
                         sample_group = "embryo_ID", 
                         keep_cds = FALSE,
                         cell_group = "cell_type")

# get simple counts matrix
ref1_counts = as.matrix(counts(ref1_ccs))
write.csv(ref1_counts, file.path(ccs_dir, "ref1_counts_table.csv"))

# extract and save reduced version of metadata
ref1_col_data <- colData(ref1_ccs)
write.csv(ref1_col_data, file.path(ccs_dir, "ref1_embryo_metadata.csv"))

# Assuming you have these objects
# save_monocle_objects(
#   cds = ref1_ccs, 
#   directory_path = file.path(ccs_dir, "ref1_ccs.rds")
# )

# REF 2
ref2_path <- "/net/seahub_zfish/vol1/data/annotated/v2.2.0/REF2/REF2_projected_cds_v2.2.0"
ref2_cds = load_monocle_objects(ref2_path, matrix_control = list(matrix_class="BPCells", matrix_path=temp_path))

# generate cell count set
ref2_ccs = new_cell_count_set(ref2_cds, 
                         sample_group = "embryo_ID", 
                         cell_group = "cell_type")

# # Assuming you have these objects
# save_monocle_objects(
#   cds = ref2_ccs, 
#   directory_path = file.path(ccs_dir, "ref2_ccs.rds")
# )

# get simple counts matrix
ref2_counts = as.matrix(counts(ref2_ccs))
write.csv(ref2_counts, file.path(ccs_dir, "ref2_counts_table.csv"))

# extract and save reduced version of metadata
ref2_col_data <- colData(ref2_ccs)
write.csv(ref2_col_data, file.path(ccs_dir, "ref2_embryo_metadata.csv"))
# saveRDS(ccs, file = file.path(ccs_dir, "ref1_ccs.rds"))



                         