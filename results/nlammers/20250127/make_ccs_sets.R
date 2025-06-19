library(monocle3)
library(hooke)
library(dplyr)

# temp dir to store BP cell files
temp_path <- "/net/trapnell/vol1/home/nlammers/tmp_files/nobackup/"
# make output dir
ccs_dir <- "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/sci-PLEX/ccs_data_cell_type_broad/"
dir.create(ccs_dir, showWarnings = FALSE, recursive = TRUE) 

# load coarse cell type labels
ct_broad = read.csv("/net/trapnell/vol1/home/elizab9/projects/projects/CHEMFISH/resources/unique_ct_full.csv")

# export most recent hotfish experiment
# ct_broad_filt <- as.data.frame(ct_broad) %>% select("cell_type", "cell_type_broad") %>% drop_na(cell_type) %>% distinct("cell_type", .keep_all = TRUE)
ct_broad_filt <- as.data.frame(ct_broad) %>%
                        count(cell_type, cell_type_broad) %>%
                        group_by(cell_type) %>%
                        slice_max(n, with_ties = FALSE) %>%
                        ungroup() %>%
                        select(cell_type, cell_type_broad)
################
# iterate through list of cds files
seahub_root <- "/net/seahub_zfish/vol1/data/annotated/v2.2.0/"

cds_name_list <- c("GENE3")#c("CHEM1.0","CHEM1.1","CHEM9", "HF","REF1","REF2",
                    #"LMX1B", "GENE1", "GENE2", "GENE3")

for(c in seq_len(length(cds_name_list))) {

    cds_name = cds_name_list[c]
    cds_path = file.path(seahub_root, cds_name, paste0(cds_name, "_projected_cds_v2.2.0/"))

    cds = load_monocle_objects(cds_path, matrix_control = list(matrix_class="BPCells", matrix_path=temp_path))

    if (!("cell_type_broad" %in% colnames(colData(cds)))) {
        col_data <- as.data.frame(colData(cds)) 
        col_data <- col_data %>%
                        left_join(ct_broad_filt, by = "cell_type")
        colData(cds)$cell_type_broad <- col_data$cell_type_broad
    }
    ccs = new_cell_count_set(cds, 
                         sample_group = "embryo_ID",
                         keep_cds = FALSE, 
                         cell_group = "cell_type_broad")

    # get simple counts matrix
    ref_counts = as.matrix(counts(ccs))
    write.csv(ref_counts, file.path(ccs_dir, paste0(cds_name, "_counts_table.csv")))

    # extract and save reduced version of metadata
    ref_col_data <- colData(ccs)
    write.csv(ref_col_data, file.path(ccs_dir, paste0(cds_name, "_metadata.csv")))
}

########
# new hotfish
# root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/"
hot_path <- "/net/trapnell/vol1/home/hklee206/sci_3lvl/240813_hotfish2_run3_novaseq/hotfish2/hotfish2_projected_cds_v2.2.0"
hot_cds = load_monocle_objects(hot_path, matrix_control = list(matrix_class="BPCells", matrix_path=temp_path))

hot_col_data <- as.data.frame(colData(hot_cds)) 
valid_indices <- which(!is.na(hot_col_data$cell_type))
hot_col_data <- hot_col_data[valid_indices, ]



# Perform left join to add "cell_type_broad"
hot_col_data <- hot_col_data %>%
  left_join(ct_broad_filt, by = "cell_type")

# Convert back to DataFrame and assign back to CDS
hot_cds <- hot_cds[, valid_indices]
colData(hot_cds)$cell_type_broad <- hot_col_data$cell_type_broad

# generate cell count set
hot_ccs = new_cell_count_set(hot_cds, 
                         sample_group = "embryo_ID", 
                         keep_cds=FALSE,
                         cell_group = "cell_type_broad")

# get simple counts matrix
hot_counts = as.matrix(counts(hot_ccs))
write.csv(hot_counts, file.path(ccs_dir, "hot2_counts_table.csv"))

# extract and save reduced version of metadata
hot_col_data <- colData(hot_ccs)
write.csv(hot_col_data, file.path(ccs_dir, "hot2_embryo_metadata.csv"))

########################
# REF orig
ref_path <- "/net/seahub_zfish/vol1/data/reference_cds/v2.2.0/reference_cds_fixed_colData_updated_241203/"
ref_cds = load_monocle_objects(ref_path, matrix_control = list(matrix_class="BPCells", matrix_path=temp_path))

# generate cell count set
ref_ccs = new_cell_count_set(ref_cds, 
                         sample_group = "embryo_ID",
                         keep_cds = FALSE, 
                         cell_group = "cell_type_broad")

# get simple counts matrix
ref_counts = as.matrix(counts(ref_ccs))
write.csv(ref_counts, file.path(ccs_dir, "ref_orig_counts_table.csv"))

# extract and save reduced version of metadata
ref_col_data <- colData(ref_ccs)
write.csv(ref_col_data, file.path(ccs_dir, "ref_orig_embryo_metadata.csv"))
