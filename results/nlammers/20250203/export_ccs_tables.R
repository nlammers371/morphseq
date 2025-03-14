library(monocle3)
library(plotly)
library(BPCells)
library(dplyr)
library(hooke)
library(tidyr)

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