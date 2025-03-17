
library(dplyr)
library(ggplot2)
library(VennDiagram)
library(grid)

# Get unique timepoints
unique_timepoints <- unique(cond_estimates$timepoint)

# Create a PDF for the Venn diagrams
pdf("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20250310/affected_populations_venn.pdf", 
    width = 8, height = 7)

# To store results for CSV export
all_results <- list()

# Store all affected cell populations across timepoints
all_lmx1b_affected <- c()
all_ctrlinj_affected <- c()

# For each timepoint, create a Venn diagram
for(tp in unique_timepoints) {
  # Get cell types affected in lmx1b
  cond_not_lmx1b <- cond_estimates %>%
    filter(timepoint == tp, m_severity == "not_severe", perturbation == "lmx1ba;lmx1bb")
  
  cond_sev_lmx1b <- cond_estimates %>%
    filter(timepoint == tp, m_severity == "severe", perturbation == "lmx1ba;lmx1bb")
  
  # Skip if data is missing
  if(nrow(cond_not_lmx1b) == 0 || nrow(cond_sev_lmx1b) == 0) {
    next
  }
  
  # Compare abundances for lmx1b
  cmp_lmx1b <- compare_abundances(ccm_mseq, cond_not_lmx1b, cond_sev_lmx1b)
  
  # Get affected cell types in lmx1b (|delta_log_abund| > 1)
  lmx1b_affected <- cmp_lmx1b %>%
    filter(abs(delta_log_abund) > 1) %>%
    pull(cell_group)
  
  # Get cell types affected in ctrl-inj
  cond_not_ctrl <- cond_estimates %>%
    filter(timepoint == tp, m_severity == "not_severe", perturbation == "ctrl-inj")
  
  cond_sev_ctrl <- cond_estimates %>%
    filter(timepoint == tp, m_severity == "severe", perturbation == "ctrl-inj")
  
  # Skip if data is missing
  if(nrow(cond_not_ctrl) == 0 || nrow(cond_sev_ctrl) == 0) {
    next
  }
  
  # Compare abundances for ctrl-inj
  cmp_ctrl <- compare_abundances(ccm_mseq, cond_not_ctrl, cond_sev_ctrl)
  
  # Get affected cell types in ctrl-inj (|delta_log_abund| > 1)
  ctrl_affected <- cmp_ctrl %>%
    filter(abs(delta_log_abund) > 1) %>%
    pull(cell_group)
  
  # Add to the all timepoints lists
  all_lmx1b_affected <- c(all_lmx1b_affected, lmx1b_affected)
  all_ctrlinj_affected <- c(all_ctrlinj_affected, ctrl_affected)
  
  # Create Venn diagram for this timepoint
  venn_list <- list(
    "lmx1ba;lmx1bb" = lmx1b_affected,
    "ctrl-inj" = ctrl_affected
  )
  
  # Calculate numbers for the Venn diagram
  n_lmx1b <- length(lmx1b_affected)
  n_ctrl <- length(ctrl_affected)
  n_both <- length(intersect(lmx1b_affected, ctrl_affected))
  
  # Create the Venn diagram
  venn <- venn.diagram(
    x = venn_list,
    filename = NULL,
    category.names = c("lmx1ba;lmx1bb", "ctrl-inj"),
    main = paste("Affected Cell Types at", tp, "hpf"),
    sub = paste("Cell types with |log fold change| > 1"),
    main.cex = 1.2,
    sub.cex = 0.8,
    fill = c("#B19CD9", "#93C6E7"),  # Soft purple and soft blue
    alpha = 0.5,
    cex = 1,
    cat.cex = 0.8,
    cat.pos = c(0, 0),
    cat.dist = c(0.05, 0.05),
    # Add the labels inside the Venn diagram
    euler.d = TRUE,
    scaled = TRUE
  )
  
  # Print the Venn diagram to the PDF
  grid.newpage()
  grid.draw(venn)
  
  # Add cell type names directly to the diagram
  grid.newpage()
  
  # Base drawing
  pushViewport(viewport(width = 0.8, height = 0.8))
  
  # Draw Venn diagram as background
  grid.draw(venn)
  
  # Add text for lmx1b-only cell types
  if(length(lmx1b_only) > 0) {
    grid.text(
      paste(lmx1b_only, collapse = "\n"),
      x = 0.25, y = 0.5,
      gp = gpar(fontsize = min(8, 30/max(1, length(lmx1b_only)))),
      just = "center"
    )
  }
  
  # Add text for ctrl-only cell types
  if(length(ctrl_only) > 0) {
    grid.text(
      paste(ctrl_only, collapse = "\n"),
      x = 0.75, y = 0.5,
      gp = gpar(fontsize = min(8, 30/max(1, length(ctrl_only)))),
      just = "center"
    )
  }
  
  # Add text for both cell types
  if(length(both) > 0) {
    grid.text(
      paste(both, collapse = "\n"),
      x = 0.5, y = 0.5,
      gp = gpar(fontsize = min(8, 30/max(1, length(both)))),
      just = "center"
    )
  }
  
  popViewport()
  
  # Print the cell types in each section
  grid.newpage()
  
  # Create a text representation of the Venn results
  lmx1b_only <- setdiff(lmx1b_affected, ctrl_affected)
  ctrl_only <- setdiff(ctrl_affected, lmx1b_affected)
  both <- intersect(lmx1b_affected, ctrl_affected)
  
  # Add to results list for CSV export
  all_results[[paste0("timepoint_", tp)]] <- list(
    timepoint = tp,
    lmx1b_only = lmx1b_only,
    ctrl_only = ctrl_only,
    both = both
  )
  
  text_content <- paste0(
    "Timepoint: ", tp, " hpf\n\n",
    "lmx1ba;lmx1bb only (", length(lmx1b_only), " cell types):\n",
    paste(lmx1b_only, collapse = ", "), "\n\n",
    "ctrl-inj only (", length(ctrl_only), " cell types):\n",
    paste(ctrl_only, collapse = ", "), "\n\n",
    "Both (", length(both), " cell types):\n",
    paste(both, collapse = ", ")
  )
  
  grid.text(text_content, x = 0.5, y = 0.5, just = "center", gp = gpar(fontsize = 10))
}

# Create a Venn diagram for all timepoints
# Remove duplicates
all_lmx1b_affected <- unique(all_lmx1b_affected)
all_ctrlinj_affected <- unique(all_ctrlinj_affected)

# Create Venn diagram for all timepoints
venn_list_all <- list(
  "lmx1ba;lmx1bb" = all_lmx1b_affected,
  "ctrl-inj" = all_ctrlinj_affected
)

# Calculate numbers for the Venn diagram
n_lmx1b_all <- length(all_lmx1b_affected)
n_ctrl_all <- length(all_ctrlinj_affected)
n_both_all <- length(intersect(all_lmx1b_affected, all_ctrlinj_affected))

# Create the Venn diagram
venn_all <- venn.diagram(
  x = venn_list_all,
  filename = NULL,
  category.names = c("lmx1ba;lmx1bb", "ctrl-inj"),
  main = "Affected Cell Types Across All Timepoints",
  sub = "Cell types with |log fold change| > 1",
  main.cex = 1.2,
  sub.cex = 0.8,
  fill = c("#B19CD9", "#93C6E7"),  # Soft purple and soft blue
  alpha = 0.5,
  cex = 1,
  cat.cex = 0.8,
  cat.pos = c(0, 0),
  cat.dist = c(0.05, 0.05),
  # Add the labels inside the Venn diagram
  euler.d = TRUE,
  scaled = TRUE
)

# Print the Venn diagram to the PDF
grid.newpage()
grid.draw(venn_all)

# Add cell type names directly to the diagram for all timepoints
grid.newpage()

# Base drawing
pushViewport(viewport(width = 0.8, height = 0.8))

# Draw Venn diagram as background
grid.draw(venn_all)

# Adjust font size based on number of cell types
font_size_lmx1b <- max(4, min(8, 100/max(1, length(lmx1b_only_all))))
font_size_ctrl <- max(4, min(8, 100/max(1, length(ctrl_only_all))))
font_size_both <- max(4, min(8, 100/max(1, length(both_all))))

# Add text for lmx1b-only cell types
if(length(lmx1b_only_all) > 0) {
  grid.text(
    paste(lmx1b_only_all, collapse = ", "),
    x = 0.25, y = 0.5,
    gp = gpar(fontsize = font_size_lmx1b),
    just = "center"
  )
}

# Add text for ctrl-only cell types
if(length(ctrl_only_all) > 0) {
  grid.text(
    paste(ctrl_only_all, collapse = ", "),
    x = 0.75, y = 0.5,
    gp = gpar(fontsize = font_size_ctrl),
    just = "center"
  )
}

# Add text for both cell types
if(length(both_all) > 0) {
  grid.text(
    paste(both_all, collapse = ", "),
    x = 0.5, y = 0.5,
    gp = gpar(fontsize = font_size_both),
    just = "center"
  )
}

popViewport()

# Print the cell types in each section
grid.newpage()

# Create a text representation of the Venn results
lmx1b_only_all <- setdiff(all_lmx1b_affected, all_ctrlinj_affected)
ctrl_only_all <- setdiff(all_ctrlinj_affected, all_lmx1b_affected)
both_all <- intersect(all_lmx1b_affected, all_ctrlinj_affected)

# Add to results list for CSV export
all_results[["all_timepoints"]] <- list(
  timepoint = "all",
  lmx1b_only = lmx1b_only_all,
  ctrl_only = ctrl_only_all,
  both = both_all
)

text_content_all <- paste0(
  "All Timepoints Combined\n\n",
  "lmx1ba;lmx1bb only (", length(lmx1b_only_all), " cell types):\n",
  paste(lmx1b_only_all, collapse = ", "), "\n\n",
  "ctrl-inj only (", length(ctrl_only_all), " cell types):\n",
  paste(ctrl_only_all, collapse = ", "), "\n\n",
  "Both (", length(both_all), " cell types):\n",
  paste(both_all, collapse = ", ")
)

grid.text(text_content_all, x = 0.5, y = 0.5, just = "center", gp = gpar(fontsize = 10))

# Close the PDF
dev.off()

# Create the directory if it doesn't exist
dir.create("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20250310/data/lmx1b_vs_ctrlinj_timepoints/", 
           showWarnings = FALSE, recursive = TRUE)

# Export results to CSV files
for (tp_name in names(all_results)) {
  result <- all_results[[tp_name]]
  
  # Create data frames for each category
  lmx1b_only_df <- data.frame(
    timepoint = result$timepoint,
    cell_group = result$lmx1b_only,
    category = "lmx1b_only"
  )
  
  ctrl_only_df <- data.frame(
    timepoint = result$timepoint,
    cell_group = result$ctrl_only,
    category = "ctrl_only"
  )
  
  both_df <- data.frame(
    timepoint = result$timepoint,
    cell_group = result$both,
    category = "both"
  )
  
  # Combine into one data frame
  combined_df <- rbind(lmx1b_only_df, ctrl_only_df, both_df)
  
  # Create filename
  if (tp_name == "all_timepoints") {
    filename <- "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20250310/data/lmx1b_vs_ctrlinj_timepoints/affected_populations_all_timepoints.csv"
  } else {
    filename <- paste0("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20250310/data/lmx1b_vs_ctrlinj_timepoints/affected_populations_", result$timepoint, ".csv")
  }
  
  # Write to CSV
  write.csv(combined_df, filename, row.names = FALSE)
}

cat("Venn diagrams saved to affected_populations_venn.pdf\n")
cat("CSV files with affected populations saved to the data/lmx1b_vs_ctrlinj_timepoints/ directory\n")






wt_expt_ccm = new_cell_count_model(wt_ccs, 
                                   main_model_formula_str = "ns(timepoint, df=3)", 
                                   nuisance_model_formula_str = "~ expt")

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