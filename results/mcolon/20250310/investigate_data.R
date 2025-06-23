library("zscapetools")
library("Matrix")
library("dplyr")
library("splines")
library("purrr")
# library("ggplot2")


#load monocle object containing mseq transcriptomic 
cds = load_monocle_objects("/net/seahub_zfish/vol1/data/annotated/v2.2.0/LMX1B/LMX1B_projected_cds_v2.2.0")
subset_cds <- cds[, colData(cds)$perturbation %in% c("lmx1ba;lmx1bb", "ctrl-inj")]
length(unique(colData(subset_cds)$embryo_ID))


# look at morphseq/results/mcolon/20250310/lm1b_injctrl_poster_inv.ipynb for how it was generate
morph_severity_key_path <- "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20250310/data/mseq_lmx1b_snipid_to_distfromspline.csv"
morph_severity_key      <- read.csv(morph_severity_key_path, stringsAsFactors = FALSE)
# morph_severity_key      <- morph_severity_key %>% rename(embryo_ = sample)


# Read the morphseq metadata file into a dataframe
morphseq_metadata_key_path <- "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20250310/data/morphseq_metadata.csv"
morphseq_metadata <- read.csv(morphseq_metadata_key_path, stringsAsFactors = FALSE)
## subset to values of morphseq_metadata we hace confident morph measurements of 
morphseq_metadata <- morphseq_metadata[morphseq_metadata$snip_id %in% morph_severity_key$snip_id, ]
## Change the column name from 'sample' to 'embryo_ID' colData(subset_cds)$embryo_ID <------> morphseq_metadata$sample  these have the same values (the key) 
morphseq_metadata <- morphseq_metadata %>% rename(embryo_ID = sample)


# Subset the subset_cds again based on the embryo_ID column in morphseq_metadata
subset_cds_mseq <- subset_cds[, colData(subset_cds)$embryo_ID %in% morphseq_metadata$embryo_ID]




# Count the number of unique embryo_IDs for each perturbation
subset_cds_mseq_df <- as.data.frame(colData(subset_cds_mseq))
perturbation_counts <- subset_cds_mseq_df %>%
  group_by(perturbation) %>%
  summarise(unique_embryo_IDs = n_distinct(embryo_ID))
print(perturbation_counts)


# Merge the two data frames by 'embryo_ID'
morphseq_metadata <- morphseq_metadata %>%
  left_join(morph_severity_key %>% select(snip_id, hypotenuse), by = "snip_id") %>%
  rename(dist_from_spline = hypotenuse)


# add a new column to the dataset by for a give row in embryo_ID if add the corresponding embryo_ID value of dist_from_spline
# essentuially the rows have a given embryo_ID value and i need to assigne a distfromspline value to the value, using a new column called dist_from_spline
subset_cds_mseq$dist_from_spline <- morphseq_metadata$dist_from_spline[
  match(subset_cds_mseq$embryo_ID, morphseq_metadata$embryo_ID)
]

cds_mseq <- subset_cds_mseq



# # to start hooke we need a cell_count_set which is embryo X cell_count number
# unique_cell_type <- unique(colData(cds_mseq)$cell_type)[1:10]
# cds_mseq_subset <- cds_mseq[, colData(cds_mseq)$cell_type %in% unique_cell_type]
# ccs_mseq = new_cell_count_set(cds_mseq_subset, 
#                          sample_group = "embryo_ID", 
#                          cell_group = "cell_type")


ccs_mseq = new_cell_count_set(cds_mseq, 
                         sample_group = "embryo_ID", 
                         cell_group = "cell_type")

# colData(ccs_mseq)$dist_from_spline

# for each timepoint in timepoint column and for each unique val in $perturbation get assign it as severe if it is above that subsets median thhreshold 


library(dplyr)

# Convert colData to a dataframe for easier manipulation
coldata_df <- as.data.frame(colData(ccs_mseq))

# Group by timepoint and perturbation, compute the median, and assign severity
coldata_df <- coldata_df %>%
  group_by(timepoint, perturbation) %>%
  mutate(
    group_median = median(dist_from_spline, na.rm = TRUE),
    m_severity = ifelse(dist_from_spline > group_median, "severe", "not_severe")
  ) %>%
  ungroup()

# Assign the new m_severity column back to colData
colData(ccs_mseq)$m_severity <- coldata_df$m_severity




# First, compute the per-group median:
group_thresholds <- coldata_df %>%
  group_by(timepoint, perturbation) %>%
  summarize(m_severe_threshold = median(dist_from_spline, na.rm = TRUE), .groups = "drop")

library(ggplot2)

# Plot the violins with dashed black lines at each median threshold, localized to each violin
p <- ggplot(coldata_df, aes(x = factor(timepoint), y = dist_from_spline, fill = perturbation)) +
  geom_violin(position = position_dodge(width = 0.7)) +
  geom_segment(
    data = group_thresholds,
    aes(
      x = as.numeric(factor(timepoint)) - 0.25, 
      xend = as.numeric(factor(timepoint)) + 0.25,
      y = m_severe_threshold, 
      yend = m_severe_threshold,
      group = perturbation
    ),
    color = "black",
    linetype = "dashed",  # Changed from "solid" to "dashed"
    size = 0.8,
    position = position_dodge(width = 0.7)
  ) +
  guides(
    fill = guide_legend(title = "Perturbation")
  ) +
  theme_classic() +
  labs(
    x = "Timepoint",
    y = "Distance from spline"
  )

# Save the plot
ggsave(
  filename = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20250325/dist_from_spline_by_timepoint_perturbation_msevere.pdf",
  plot = p,
  width = 6,
  height = 4
)

# median_ctrlinj_dist <- median(colData(ccs_mseq)$dist_from_spline[colData(ccs_mseq)$perturbation == "ctrl-inj"], na.rm = TRUE)


# # Create the plot
# p <- ggplot(df, aes(x = factor(timepoint), y = dist_from_spline, fill = perturbation)) +
#   geom_violin() +
#   theme_classic()

# # Save the plot to a Pdf
# ggsave(
#   filename = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20250326/dist_from_spline_by_timepoint_perturbation_noline.pdf",
#   plot = p,
#   width = 6,
#   height = 4
# )


# # get the median control distance by subsetting for "ctrl-inj"

# p <- ggplot(df, aes(x = factor(timepoint), y = dist_from_spline, fill = perturbation)) +
#   geom_violin() +
#   geom_hline(aes(yintercept = median_ctrlinj_dist, linetype = "Median dist for ctrl-inj"), color = "red") +
#   scale_linetype_manual(name = "Reference Line", values = c("Median dist for ctrl-inj" = "dashed")) +
#   theme_classic()

# # Save the plot to a Pdf
# ggsave(
#   filename = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20250326/dist_from_spline_by_timepoint_perturbation_ctrlline.pdf",
#   plot = p,
#   width = 6,
#   height = 4
# )





reg_formula_1= "~ ns( timepoint , knots= c(48) ) + ns( timepoint , knots= c(48) ):perturbation + m_severity:perturbation + offset(log(Offset))"


ccm_mseq  = new_cell_count_model(ccs_mseq, main_model_formula_str =reg_formula_1, num_threads = 6)

# The two levels for perturbation
perturbations <- c("ctrl-inj", "lmx1ba;lmx1bb")

# The two levels for severity
m_severities  <- c("not_severe", "severe")

timepoints    <- unique(colData(ccs_mseq)$timepoint)

# Create all combinations of these factors
comparisons <- tidyr::expand_grid(
  timepoint     = timepoints,
  perturbation  = perturbations,
  m_severity    = m_severities
)

# Now use that tibble to estimate abundances from your model:
cond_estimates <- estimate_abundances(ccm_mseq, comparisons)

# Check the first few rows
head(cond_estimates)


# --------------------------
# 1) Compare ctrl-inj vs lmx1ba;lmx1bb for all timepoints and severities
# --------------------------

library(dplyr)
library(tidyr)

# We assume ccm_mseq is your cell_count_model
# and cond_estimates is the tibble from estimate_abundances(ccm_mseq, ...)

# Identify the relevant timepoints and severities
timepoints    <- unique(cond_estimates$timepoint)
m_severities  <- unique(cond_estimates$m_severity)

comparison_list <- list()

# Loop over each combination of timepoint and m_severity
for(tpt in timepoints) {
  for(msv in m_severities) {
    cond_x <- cond_estimates %>%
      filter(timepoint == tpt,
             perturbation == "ctrl-inj",
             m_severity == msv)
    
    cond_y <- cond_estimates %>%
      filter(timepoint == tpt,
             perturbation == "lmx1ba;lmx1bb",
             m_severity == msv)
    
    # Skip if either side is empty
    if(nrow(cond_x) == 0 || nrow(cond_y) == 0) {
      next
    }
    
    # Compare the two sets of abundances
    cmp_tbl <- compare_abundances(ccm_mseq, cond_x, cond_y)
    
    # Add extra columns to record timepoint & severity
    cmp_tbl <- cmp_tbl %>%
      mutate(
        timepoint = tpt,
        m_severity = msv,
        x_perturbation = "ctrl-inj",
        y_perturbation = "lmx1ba;lmx1bb"
      )
    
    # Store in a list
    comparison_list[[paste0("t",tpt,"_",msv)]] <- cmp_tbl
   }
}

# Combine into one data frame
all_comparisons <- dplyr::bind_rows(comparison_list)


# --------------------------
# 2) Write out comparison + cond_estimates if desired
# --------------------------

# Write out the original cond_estimates
write.csv(
  cond_estimates,
  "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20250325/data/cond_estimates.csv",
  row.names = FALSE
)

# Write out our new comparison table
write.csv(
  all_comparisons,
  "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20250325/data/cond_comparisons.csv",
  row.names = FALSE
)

# --------------------------
# 3) Example: plot contrast at timepoint=48 for not_severe vs. severe
# --------------------------

# Suppose you want to compare “not_severe” vs. “severe” at timepoint=48
# (Regardless of perturbation, or filtered for a single perturbation—your choice.)



# Create a single PDF file to contain all plots
pdf("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20250325/lmx1b_all_timepoint_severe_vs_not.pdf", width = 8, height = 6)
# Get unique timepoints
unique_timepoints <- unique(cond_estimates$timepoint)[0]
for(tp in unique_timepoints) {
  # Filter the estimates for the current timepoint for each severity
  cond_not <- cond_estimates %>%
    filter(timepoint == tp, m_severity == "not_severe", perturbation == "lmx1ba;lmx1bb")
  cond_sev <- cond_estimates %>%
    filter(timepoint == tp, m_severity == "severe", perturbation == "lmx1ba;lmx1bb")
  
  # Skip this timepoint if either condition is missing
  if(nrow(cond_not) == 0 || nrow(cond_sev) == 0) {
    message(paste("Skipping timepoint", tp, "- insufficient data"))
    next
  }
  
  # Compare the abundances
  cmp_contrast <- compare_abundances(ccm_mseq, cond_not, cond_sev)

  # Option 1: If plot_contrast returns a ggplot object
  p <- plot_contrast(ccm_mseq, cmp_contrast, q_value_thresh = 0.05) +
    labs(title = paste("Difference: not_severe vs. severe (lmx1ba;lmx1bb) at", tp, "hpf"))
  print(p)
  
  # Option 2: If plot_contrast is a base R plotting function (uncomment this block instead)
  # plot_contrast(ccm_mseq, cmp_contrast, q_value_thresh = 0.05)
  # title(main = paste("Difference: not_severe vs. severe (lmx1ba;lmx1bb) at", tp, "hpf"))
}
#Close the PDF device
dev.off()



library(dplyr)
library(tidyr)
library(ggplot2)
library(ggrepel)  # For text labels that don't overlap

library(dplyr)
library(tidyr)
library(ggplot2)
library(ggrepel)  # For text labels that don't overlap

# Get unique timepoints
unique_timepoints <- unique(cond_estimates$timepoint)

# Create a single PDF file to contain all scatter plots
pdf("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20250325/lmx1b_abundance_scatter_plots.pdf", 
    width = 8, height = 7)

for(tp in unique_timepoints) {
  # Filter the estimates for the current timepoint for each severity
  cond_not <- cond_estimates %>%
    filter(timepoint == tp, m_severity == "not_severe", perturbation == "lmx1ba;lmx1bb")
  cond_sev <- cond_estimates %>%
    filter(timepoint == tp, m_severity == "severe", perturbation == "lmx1ba;lmx1bb")
  
  # Skip this timepoint if either condition is missing
  if(nrow(cond_not) == 0 || nrow(cond_sev) == 0) {
    cat("Skipping timepoint", tp, "- insufficient data\n")
    next
  }
  
  # Compare the abundances
  cmp_contrast <- compare_abundances(ccm_mseq, cond_not, cond_sev)
  
  # Identify top 5 cell groups with highest and lowest delta log abundances
  top_high <- cmp_contrast %>%
    arrange(desc(delta_log_abund)) %>%
    slice_head(n = 5) %>%
    pull(cell_group)
    
  top_low <- cmp_contrast %>%
    arrange(delta_log_abund) %>%
    slice_head(n = 5) %>%
    pull(cell_group)
    
  # Combine them for labeling
  to_label <- c(top_high, top_low)
  
  # Create a column for labeling
  cmp_contrast <- cmp_contrast %>%
    mutate(label = ifelse(cell_group %in% to_label, cell_group, ""))
    
  # Find the range for both axes to make them equal
  max_val <- max(c(cmp_contrast$log_abund_x, cmp_contrast$log_abund_y), na.rm = TRUE)
  min_val <- min(c(cmp_contrast$log_abund_x, cmp_contrast$log_abund_y), na.rm = TRUE)
  
  # Add a small buffer
  buffer <- (max_val - min_val) * 0.05
  plot_min <- min_val - buffer
  plot_max <- max_val + buffer
  
  # Create the scatter plot
  p <- ggplot(cmp_contrast, aes(x = log_abund_x, y = log_abund_y)) +
    # Points with normal coloring
    geom_point(data = filter(cmp_contrast, abs(delta_log_abund) <= 1), 
               color = "grey50", alpha = 0.7) +
    # Points with significant changes highlighted in red
    geom_point(data = filter(cmp_contrast, abs(delta_log_abund) > 1), 
               color = "red", alpha = 0.7) +
    # Add labels only for significant points
    geom_text_repel(
      data = filter(cmp_contrast, abs(delta_log_abund) > 1),
      aes(label = cell_group),
      box.padding = 0.5,
      max.overlaps = 20,
      size = 3
    ) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "blue") +
    coord_fixed(xlim = c(plot_min, plot_max), ylim = c(plot_min, plot_max)) +
    labs(
      title = paste("Cell Type Abundance Comparison at", tp, "hpf"),
      subtitle = "not_severe (x-axis) vs. severe (y-axis) in lmx1ba;lmx1bb",
      x = "Log Abundance (not_severe)",
      y = "Log Abundance (severe)"
    ) +
    theme_bw() +
    theme(
      plot.title = element_text(size = 12, face = "bold")
    )
  
  print(p)
}

# Close the PDF device
dev.off()

cat("All lmx1b abundance scatter plots saved to a single PDF file\n")





pdf("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20250325/ctrl_inj_abundance_scatter_plots.pdf", 
    width = 8, height = 7)

for(tp in unique_timepoints) {
  # Filter the estimates for the current timepoint for each severity
  cond_not <- cond_estimates %>%
    filter(timepoint == tp, m_severity == "not_severe", perturbation == "ctrl-inj")
  cond_sev <- cond_estimates %>%
    filter(timepoint == tp, m_severity == "severe", perturbation == "ctrl-inj")
  
  # Skip this timepoint if either condition is missing
  if(nrow(cond_not) == 0 || nrow(cond_sev) == 0) {
    cat("Skipping timepoint", tp, "- insufficient data\n")
    next
  }
  
  # Compare the abundances
  cmp_contrast <- compare_abundances(ccm_mseq, cond_not, cond_sev)
  
  # Identify top 5 cell groups with highest and lowest delta log abundances
  top_high <- cmp_contrast %>%
    arrange(desc(delta_log_abund)) %>%
    slice_head(n = 5) %>%
    pull(cell_group)
    
  top_low <- cmp_contrast %>%
    arrange(delta_log_abund) %>%
    slice_head(n = 5) %>%
    pull(cell_group)
    
  # Combine them for labeling
  to_label <- c(top_high, top_low)
  
  # Create a column for labeling
  cmp_contrast <- cmp_contrast %>%
    mutate(label = ifelse(cell_group %in% to_label, cell_group, ""))
    
  # Find the range for both axes to make them equal
  max_val <- max(c(cmp_contrast$log_abund_x, cmp_contrast$log_abund_y), na.rm = TRUE)
  min_val <- min(c(cmp_contrast$log_abund_x, cmp_contrast$log_abund_y), na.rm = TRUE)
  
  # Add a small buffer
  buffer <- (max_val - min_val) * 0.05
  plot_min <- min_val - buffer
  plot_max <- max_val + buffer
  
  # Create the scatter plot
  p <- ggplot(cmp_contrast, aes(x = log_abund_x, y = log_abund_y)) +
    # Points with normal coloring
    geom_point(data = filter(cmp_contrast, abs(delta_log_abund) <= 1), 
               color = "grey50", alpha = 0.7) +
    # Points with significant changes highlighted in red
    geom_point(data = filter(cmp_contrast, abs(delta_log_abund) > 1), 
               color = "red", alpha = 0.7) +
    # Add labels only for significant points
    geom_text_repel(
      data = filter(cmp_contrast, abs(delta_log_abund) > 1),
      aes(label = cell_group),
      box.padding = 0.5,
      max.overlaps = 20,
      size = 3
    ) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "blue") +
    coord_fixed(xlim = c(plot_min, plot_max), ylim = c(plot_min, plot_max)) +
    labs(
      title = paste("Cell Type Abundance Comparison at", tp, "hpf"),
      subtitle = "not_severe (x-axis) vs. severe (y-axis) in ctrl-inj",
      x = "Log Abundance (not_severe)",
      y = "Log Abundance (severe)"
    ) +
    theme_bw() +
    theme(
      plot.title = element_text(size = 12, face = "bold")
    )
  
  print(p)
}

# Close the PDF device
dev.off()

source("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20250310/venn_diagram_script.r")

# Subset the data into three lists by category
lmx1b_only_list <- combined_df[combined_df$category == "lmx1b_only", ]$cell_group
ctrl_only_list <- combined_df[combined_df$category == "ctrl_only", ]$cell_group
both_list <- combined_df[combined_df$category == "both", ]$cell_group

# Print the lists
print(lmx1b_only_list)
print(ctrl_only_list)
print(both_list)

severe_embryo_ID <- colData(ccs_mseq[ colData(ccs_mseq)$m_severity == "severe",])$sample

cds_mseq_severe <- cds_mseq[,colData(cds_mseq)$embryo_ID %in% severe_embryo_ID]
cds_mseq_severe <- cds_mseq_severe[, colData(cds_mseq_severe)$cell_type %in% combined_df$cell_group]


ccm_mseq_severe = fit_mt_models(cds_mseq_severe, 
                             sample_group = "embryo_ID", 
                             cell_group = "cell_type", 
                             perturbation_col = "perturbation", 
                             ctrl_ids = c("ctrl-inj"),
                             mt_ids = c("lmx1ba;lmx1bb"),
                             num_threads=6)

perturb_ccm_lmx1b <- ccm_mseq_severe$perturb_ccm[[1]]

library(purrr)

# Load necessary library
library(ggplot2)

p <- plot_cell_type_perturb_kinetics(perturb_ccm_lmx1b, 
                                newdata = tibble("expt"= "LMX1B"), 
                                raw_counts = F, 
                                nrow = 5) + 
                                xlab("time")


# Define the filename and save the plot as a PDF
ggsave(filename = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20250325/plots/cell_type_severe_lmx1bvs_ctrl_perturb_kinetics.pdf", 
       plot = p, 
       device = "pdf", 
       width = 20, 
       height = 12)

# Print confirmation message
message("Plot saved as cell_type_perturb_kinetics.pdf")

# 1) Identify the embryo_IDs for lmx1b

colData(ccs_mseq[ colData(ccs_mseq)$perturbation == "lmx1ba;lmx1bb",])$perturbation

lmx1b_embryo_ID <- colData(ccs_mseq[ ,colData(ccs_mseq)$perturbation == "lmx1ba;lmx1bb"])$sample

colData(cds_mseq)$m_severity <-
  colData(ccs_mseq)$m_severity[
    match(colData(cds_mseq)$embryo_ID, colData(ccs_mseq)$sample)
  ]


# 2) Subset the CDS object to include only those embryo_IDs
cds_mseq_lmx1b <- cds_mseq[,colData(cds_mseq)$embryo_ID %in% lmx1b_embryo_ID]
cds_mseq_lmx1b <- cds_mseq_lmx1b[, colData(cds_mseq_lmx1b)$cell_type %in% combined_df$cell_group]

# 3) Create a new column by concatenating 'perturbation' and 'm_severity'
colData(cds_mseq_lmx1b)$perturbation_severity <-
  paste(colData(cds_mseq_lmx1b)$perturbation,
        colData(cds_mseq_lmx1b)$m_severity,
        sep = "_")

# 4) Get the unique values of the new column
unique_vals <- unique(colData(cds_mseq_lmx1b)$perturbation_severity)
unique_vals
ccm_mseq_lmx1b = fit_mt_models(cds_mseq_lmx1b, 
                             sample_group = "embryo_ID", 
                             cell_group = "cell_type", 
                             perturbation_col = "perturbation_severity", 
                             ctrl_ids = c("lmx1ba;lmx1bb_not_severe"),
                             mt_ids = c("lmx1ba;lmx1bb_severe"),
                             num_threads=6)

perturb_ccm_lmx1b_only <- ccm_mseq_lmx1b$perturb_ccm[[1]]

library(purrr)

# Load necessary library
library(ggplot2)

p <- plot_cell_type_perturb_kinetics(perturb_ccm_lmx1b_only, 
                                newdata = tibble("expt"= c("LMX1B") ), 
                                raw_counts = F, 
                                nrow = 5) + 
                                xlab("time")


# Define the filename and save the plot as a PDF
ggsave(filename = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20250325/plots/cell_type_severe_lmx1bvs_nonseverelmx1b_perturb_kinetics_exptLMX1B.pdf", 
       plot = p, 
       device = "pdf", 
       width = 20, 
       height = 12)

p <- plot_cell_type_perturb_kinetics(perturb_ccm_lmx1b_only, 
                                newdata = tibble("expt"= c("LMX1Bearly") ), 
                                raw_counts = F, 
                                nrow = 5) + 
                                xlab("time")


# Define the filename and save the plot as a PDF
ggsave(filename = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20250325/plots/cell_type_severe_lmx1bvs_nonseverelmx1b_perturb_kinetics_exptLMX1Bearly.pdf", 
       plot = p, 
       device = "pdf", 
       width = 20, 
       height = 12)







wt_expt_ccm = new_cell_count_model(wt_ccs, 
                                   main_model_formula_str = "ns(timepoint, df=3)", 
                                   nuisance_model_formula_str = "~ expt")


comparisons <- tidyr::expand_grid(
  timepoint     = timepoints,
  perturbation  = perturbations,
  m_severity    = m_severities
)

# Now use that tibble to estimate abundances from your model:
cond_estimates <- estimate_abundances(ccm_mseq, comparisons)

colnames(colData(ccs_mseq))
colData(ccs_mseq)$perturbation + colData(ccs_mseq)$m_severity #add these like they are python strings

batches = data.frame(batch = unique(colData(wt_ccs)$expt))                                   
batches = batches %>% mutate(tp_preds = purrr::map(.f = function(batch) {
 estimate_abundances_over_interval(wt_expt_ccm,
                                            start_time,
                                            stop_time,
                                            knockout=FALSE,
                                            interval_col="timepoint",
                                            interval_step=2,
                                            expt = batch)
}, .x=batch))

wt_timepoint_pred_df = batches %>% select(tp_preds) %>% tidyr::unnest(tp_preds)

ggplot(wt_timepoint_pred_df, aes(x = timepoint)) +
  geom_line(aes(y = exp(log_abund) + exp(log_abund_detection_thresh), color=expt)) +
  facet_wrap(~cell_group, scales="free_y", nrow = 2) + monocle3:::monocle_theme_opts() + 
  ggtitle("wild-type kinetics by expt")

library(future)
library(corrplot)
library(factoextra)
plan(multisession, workers = 6)


ccs_mseq_pln_data <- PLNmodels::prepare_data(counts = counts(ccs_mseq) + 1,
                                          covariates = colData(ccs_mseq) %>% as.data.frame,
                                          offset = monocle3::size_factors(ccs_mseq))


reg_formula_2= "~ ns( timepoint , knots= c(48) ) + ns( timepoint , knots= c(48) ):perturbation + offset(log(Offset))"



PCA_models <- PLNmodels::PLNPCA(
  # Abundance ~  ns(timepoint , knots= c(48) ) + ns(timepoin),t , knots= c(48) ):expt + ns( timepoint , knots= c(48) ):perturbation + offset(log(Offset)
  Abundance ~  ns(timepoint , knots= c(48) ) + ns(timepoint, knots= c(48) ):expt + offset(log(Offset)),
  data  = ccs_mseq_pln_data, 
  ranks = 1:7
)                                        


# Save the plot to a PDF file
ggsave(
  filename = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20250310/plots/PCA_models_selection.png",
  plot = plot(PCA_models, reverse = TRUE),
  device =  "png",
  width = 8,
  height = 6
)

myPCA_ICL <- getModel(PCA_models, 5)

# Save the plot to a PDF file
ggsave(
  filename = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20250310/plots/PCA_models_mseverity_plot.png",
  plot = plot(myPCA_ICL, ind_cols = ccs_mseq_pln_data$m_severity),
  device = "png",
  width = 8,
  height = 6
)

message("PCA models plot saved successfully.")

# myPCA_ICL <- getBestModel(PCA_models, "ICL") 




ggsave(
  filename = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20250310/plots/PCA_ICL_structure_models_plot_perturbation.png",
  plot = plot(myPCA_ICL, ind_cols = ccs_mseq_pln_data$perturbation),
  device = "png",
  width = 8,
  height = 6
)

## Variables





# Ensure row names align before merging
pc_scores <- as.data.frame(myPCA_ICL$scores)
pc_scores$sample <- rownames(pc_scores)

# Merge with ccs_mseq_pln_data based on sample names
ccs_mseq_pln_data <- ccs_mseq_pln_data %>%
  select(-starts_with("PC")) %>%
  left_join(pc_scores, by = c("sample" = "sample"))

# Save the updated data to a CSV file
write.csv(ccs_mseq_pln_data, "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20250310/data/ccs_mseq_pln_data_with_PCs.csv", row.names = FALSE)

cat("Updated ccs_mseq_pln_data with PCs saved successfully.\n")