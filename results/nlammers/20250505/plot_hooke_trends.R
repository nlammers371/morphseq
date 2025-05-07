rm(list = ls())

library(monocle3)
library(hooke)
library(dplyr)
library(ggplot2)
library(tidyverse)
library(ggstream)
library(stringr)
library(matrixStats)
library(RColorBrewer)

# ---- 1) Load data ----
# set path to hooke data
root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/analyses/crossmodal/hotfish/"
hooke_spline_path = paste0(root, "/hooke_time_trends.csv")
hooke_ct_path = paste0(root, "/unique_ct_full_germ_layer_nl.csv")
n_cell_path = paste0(root, "/n_cell_table.csv")

# set path to monocle data
hooke_spline_df = as.data.frame(read.csv(hooke_spline_path))
ct_df = as.data.frame(read.csv(hooke_ct_path))
n_cell_df = as.data.frame(read.csv(n_cell_path))

make_key <- function(x) {
  x %>%
    str_remove_all("[-?+/ \\(\\),]") %>%  # remove spaces, ( ) and commas
    str_to_lower()                    # normalize case
}

# dedupe ct table
ct_df2 <- ct_df %>%
  mutate(join_key = make_key(cell_type_broad)) %>%
  distinct(join_key, .keep_all=TRUE) %>%
  select(-cell_type_broad)

ct_df2 <- ct_df2 %>%
  mutate(
    germ.layer = replace_na(germ.layer, "other")
  )

# ---- 2) Pivot longer & join metadata ----
hooke_counts_long <- hooke_spline_df %>%
  pivot_longer(
    cols      = -stage_hpf,
    names_to  = "cell_type_broad",
    values_to = "log_count"
  ) 

hooke_counts_long$cell_type_broad  <- hooke_counts_long$cell_type_broad <- gsub("\\.", " ", hooke_counts_long$cell_type_broad)

hooke_counts_long <- hooke_counts_long%>%
  mutate(join_key = make_key(cell_type_broad))

hooke_counts_long <- hooke_counts_long %>%
  left_join(ct_df2, by = "join_key") %>%
  select(-join_key) 

# Condense to tissue level
hooke_counts_tissue <- hooke_counts_long %>%
                        group_by(stage_hpf, tissue, germ.layer) %>%
                        summarise(log_count = logSumExp(log_count), .groups   = "drop"  ) %>%
                        filter(!is.na(tissue)) %>%
                        mutate(gl_tissue = interaction(germ.layer, tissue, sep=":")) %>%
                        arrange(stage_hpf, gl_tissue) %>%
                        filter((stage_hpf>= 0) & (stage_hpf <= 48) & (tissue != "periderm")) %>% 
                        mutate(counts = exp(log_count)) 

desired_levels <- hooke_counts_tissue %>%
  distinct(gl_tissue, germ.layer) %>%
  arrange(germ.layer, gl_tissue) %>%
  pull(gl_tissue)

# 2) Refactor gl_tissue in your main df
hooke_counts_tissue <- hooke_counts_tissue %>%
  mutate(
    gl_tissue = factor(gl_tissue, levels = desired_levels)
  )

# 2) Build a lookup of tissue → germ.layer (if you don’t already have it)
tissue_meta <- hooke_counts_tissue %>%
  distinct(gl_tissue, germ.layer)

# 2) Define one base palette per germ.layer
base_pals <- list(
  ectoderm  = brewer.pal(8, "Blues"),
  mesoderm  = brewer.pal(8, "Greens"),
  endoderm  = brewer.pal(8, "Reds"),
  other     = brewer.pal(8, "Greys")
)

# 3) Now your fill palette vector must be named exactly by those same levels:
tissue_shades <- tissue_meta %>%
  # refactor gl_tissue to the right levels
  mutate(gl_tissue = factor(gl_tissue, levels = desired_levels)) %>%
  group_by(germ.layer) %>%
  mutate(shade = colorRampPalette(base_pals[[unique(germ.layer)]])(n())) %>%
  ungroup() %>%
  # now explicitly reference the columns in `.`
  { setNames(.$shade, .$gl_tissue) }

# 1) Turn your published log10 totals back into linear counts:
n_cell_df2 <- n_cell_df %>%
  arrange(stage_hpf) %>%
  mutate(total_cells = 10^n_cells_log)   # now raw counts


total_cells_df <- hooke_counts_tissue %>%
  group_by(stage_hpf) %>%
  summarise(total_cells_base = sum(counts), .groups   = "drop" ) 


# 2) Interpolate those counts onto your hooke stages:
hooke_counts_tissue2 <- hooke_counts_tissue %>%
  mutate(
    total_cells_true = approx(
      x    = n_cell_df2$stage_hpf,
      y    = n_cell_df2$total_cells,
      xout = stage_hpf,
      rule = 2
    )$y
  ) %>%
  left_join(total_cells_df, by = "stage_hpf") # %>%
  #mutate(
   # total_cells_model = exp(total_cells_base),           # linear from your logSumExp
    #counts_model       = exp(log_count),                 # per‑tissue modeled counts
    #counts_norm        = 100 *counts_model 
   # / total_cells_model 
   # * total_cells_true          # rescale to “true” envelope
 # )

# 3) Plot the linear rescaled counts, with an optional log‑y axis:
ggplot(hooke_counts_tissue2, 
       aes(x = stage_hpf,
           y = counts_norm,
           group = gl_tissue,
           fill  = gl_tissue)) +
  geom_stream(#type       = "mirror",
              offset = "zero",
              extra_span = 0.2,
              bw         = 0.75,
              sorting    = "none") +
  scale_fill_manual(values = tissue_shades) +
  coord_trans(y = "log10") +
  # if you really want a log scale on the y‑axis:
  # scale_y_log10(labels = scales::comma_format()) +
  theme_void() +
  theme(legend.position = "none")