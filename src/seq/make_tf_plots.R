library(monocle3)
library(hooke)
library(dplyr)

ref_cds_path = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/sci-PLEX/processed_sci_data/reference_cds_v2.0.2/cds_object.rds"
#ref_cds = load_monocle_objects(ref_cds_path)
cds = readRDS(ref_cds_path)


plot_cells(cds, genes=c("tbx6"))
