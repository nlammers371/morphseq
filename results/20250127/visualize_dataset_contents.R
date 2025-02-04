library(plotly)
library(dplyr)

# load csv files containing embryo metadata
ccs_dir <- "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/sci-PLEX/ccs_data/"

hot_df= as_tibble(as.data.frame(read.csv(file.path(ccs_dir, "hot2_embryo_metadata.csv"))))
ref_df = as_tibble(as.data.frame(read.csv(file.path(ccs_dir, "ref_orig_embryo_metadata.csv"))))
ref1_df = as_tibble(as.data.frame(read.csv(file.path(ccs_dir, "ref1_embryo_metadata.csv"))))
ref2_df = as_tibble(as.data.frame(read.csv(file.path(ccs_dir, "ref2_embryo_metadata.csv"))))