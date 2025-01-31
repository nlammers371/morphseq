library(monocle3)
library(plotly)
library(BPCells)
library(dplyr)

# root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/"
root = "~/projects/data/morphseq/"

# set root to cds folder
cds_root <- "morphseq/hotfish/20240813/hotfish2/hotfish2_projected_cds_v2.2.0"

data_path <- path.dir(root, "hotfish/20240813/hotfish2/built_data/")
dir.create(data_path, showWarnings = FALSE)

# make folder for figues
fig_path <- "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/hotfish/figures/20240813_exploratory/"
dir.create(fig_path, showWarnings = FALSE)
# print("Made it (11)")
# load data
monocle_objects <- load_monocle_objects(cds_root)
hot_cds <- readRDS(paste(cds_root, "cds_object.rds", sep = "/"))

# first, look at distribution of min_nn_dist to check for outliers
# Create the histogram
p <- plot_ly(x = colData(hot_cds)$min_nn_dist, type = "histogram") %>%
  layout(
    title = "minimum nn distance",
    xaxis = list(title = "X"),
    yaxis = list(title = "Frequency")
  )

# flag outliers
outlier_thresh <- quantile(colData(hot_cds)$min_nn_dist, probs = c(0.9995))
outlier_flag_vec <- as.logical(colData(hot_cds)$min_nn_dist > outlier_thresh)

# Add a vertical line at a specific x-value (e.g., x = 1)
p <- p %>%
  add_trace(
    x = c(outlier_thresh, outlier_thresh),  # X-coordinates of the line (same value for vertical line)
    y = c(0, max(table(cut(colData(hot_cds)$min_nn_dist, breaks = 30)))),  # Y-coordinates (cover the y-range)
    type = "scatter",
    mode = "lines",
    line = list(color = "black", dash = "dash"),  # Custom line style
    name = "Threshold"
  )

# save
htmlwidgets::saveWidget(as_widget(p), file.path(fig_path, "min_nn_dist_hist.html"), selfcontained = TRUE)

# visualize putative outliers in UMAP space
umap_coords <- reducedDims(hot_cds)$UMAP # Get the UMAP coordinates (3D)

# Extract cell metadata for color-coding
cell_metadata <- as.data.frame(colData(hot_cds))

# Define custom colors for TRUE and FALSE
boolean_colors <- c("TRUE" = "red", "FALSE" = "blue")

# Create a 3D scatter plot
u_outlier <- plot_ly(
  x = ~umap_coords[, 1],  # UMAP dimension 1
  y = ~umap_coords[, 2],  # UMAP dimension 2
  z = ~umap_coords[, 3],  # UMAP dimension 3
  type = "scatter3d",
  mode = "markers",
  marker = list(size = 3, color = as.numeric(outlier_flag_vec),  # Use the boolean vector for colors
    colorscale = list(
      list(0, boolean_colors["FALSE"]),  # Color for FALSE
      list(1, boolean_colors["TRUE"])   # Color for TRUE
    ),
    colorbar = list(title = "Boolean")
  ),
  text = ~paste("Cell:", rownames(cell_metadata))  # Add hover text
) %>%
  layout(
    title = "3D UMAP Plot",
    scene = list(
      xaxis = list(title = "UMAP1"),
      yaxis = list(title = "UMAP2"),
      zaxis = list(title = "UMAP3")
    )
  )

u_outlier # display
# save
htmlwidgets::saveWidget(as_widget(u_outlier), file.path(fig_path, "umap_3d_outliers.html"), selfcontained = TRUE)

# Create 3D UMAP colored according to cell type
umap_filtered <- umap_coords[!outlier_flag_vec, ]
cell_type_vec <- colData(hot_cds)$cell_type
cell_type_filtered <- cell_type_vec[!outlier_flag_vec]

u_cell <- plot_ly(
  x = umap_filtered[, 1],  # UMAP Dimension 1
  y = umap_filtered[, 2],  # UMAP Dimension 2
  z = umap_filtered[, 3],  # UMAP Dimension 3
  type = "scatter3d",
  mode = "markers",
  marker = list(
    size = 3       # Color scale
  ),
  color = as.factor(cell_type_filtered),
  colors = "Set2",
  showlegend = FALSE,
  text = paste("Cell Type:", cell_type_filtered),  # Hover text showing cell type
  hoverinfo = "text"
  ) %>%
  layout(
    title = "3D UMAP Embedding Colored by Cell Type",
    scene = list(
      xaxis = list(title = "UMAP 1"),
      yaxis = list(title = "UMAP 2"),
      zaxis = list(title = "UMAP 3")
    )
  )

u_cell # display
htmlwidgets::saveWidget(as_widget(u_cell), file.path(fig_path, "umap_filtered_3d_celltype.html"), selfcontained = TRUE)

##################
# Now, let's start doing pseudo-stage analysis
col_df <- colData(hot_cds) %>% as.data.frame()
col_df <- col_df[!outlier_flag_vec, ] %>%
              filter(count_per_embryo > 150)
col_df$mean_nn_time <- as.numeric(col_df$mean_nn_time)

write.csv(col_df, file = file.path(data_path, "col_data.csv"), row.names = FALSE)

# # first group by 
# emb_stage_df <- col_df %>% 
#                 group_by(embryo_ID, timepoint, temp) %>%
#                 summarise(avg_nn_time=mean(mean_nn_time), std_nn_time=sd(mean_nn_time, na.rm = TRUE), n_cells=n()) %>%
#                 ungroup() %>% 
#                 mutate(linear_stage_prediction= 6 + (timepoint - 6)*(0.055*temp-0.57), se_nn_time=std_nn_time/sqrt(n_cells))
# print("Made it (126)")
# # Create a scatter plot with error bars
# emb_stage_sctr <- plot_ly(
#   data = emb_stage_df,
#   x = ~linear_stage_prediction,             # X-axis: linear stage prediction
#   y = ~avg_nn_time,                         # Y-axis: avg_nn_time
#   type = "scatter",
#   mode = "markers",                         # Only markers
#   # error_y = list(
#   #   type = "data",
#   #   array = ~se_nn_time,                   # Error bar values
#   #   visible = TRUE,
#   #   color="black"
#   # ),
#   marker = list(
#     size = 7
#   ),
#   color = ~as.factor(temp),                            # Color points by "temp"
#   colors = "Set2"                           # Discrete colormap
# ) %>%
#   add_trace(
#     x = c(10, max(emb_stage_df$linear_stage_prediction)),  # Diagonal line x-coordinates
#     y = c(10, max(emb_stage_df$linear_stage_prediction)),  # Diagonal line y-coordinates
#     type = "scatter",
#     mode = "lines",
#     line = list(dash = "dash", color = "black", width = 2),  # Dashed line style
#     showlegend = FALSE,
#     inherit = FALSE                                     # Hide legend for line
#   ) %>%
#   layout(
#     title = "2D Scatterplot of Avg NN Time vs. Timepoint",
#     xaxis = list(title = "Linear Stage Prediction"),
#     yaxis = list(title = "Average NN Time"),
#     showlegend = TRUE                       # Show legend for colors
#   )

# # Display the plot
# emb_stage_sctr
# htmlwidgets::saveWidget(as_widget(emb_stage_sctr), file.path(fig_path, "embryo_transcriptional_vs_predicted_stage.html"), selfcontained = TRUE)

# print("Made it (166)")
# ###########
# # Now let's look at the breakdown by tissue
# tissue_stage_df <- col_df %>% 
#                 group_by(embryo_ID, timepoint, temp, tissue) %>%
#                 summarise(avg_nn_time=mean(mean_nn_time), n_cells=n(), .groups = "keep") %>%
#                 ungroup() %>% 
#                 group_by(timepoint, temp, tissue, .groups = "keep") %>%
#                 summarise(q1_nn_time=quantile(avg_nn_time, 0.25), 
#                           q2_nn_time=quantile(avg_nn_time, 0.75),
#                           med_nn_time=median(avg_nn_time), 
#                           cells_per_embryo=mean(n_cells), 
#                           .groups = "drop_last") %>%
#                 ungroup() %>%
#                 mutate(linear_stage_prediction= 6 + (timepoint - 6)*(0.055*temp-0.57)) 

# # Step 1: Determine sorting order based on timepoint = 24
# ordering <- df %>%
#   filter(timepoint == 24) %>%
#   arrange(avg_nn_time) %>%
#   pull(tissue)

# # Step 2: Reorder tissue for all rows based on the derived ordering
# df <- df %>%
#   mutate(tissue = factor(tissue, levels = ordering))

# # Get the unique order of tissues
# unique_tissues <- levels(df$tissue)

# # Step 3: Dynamically assign Y positions for the rectangles
# df <- df %>%
#   group_by(timepoint, tissue) %>%
#   mutate(
#     y_min = as.numeric(tissue) - 0.4,
#     y_max = as.numeric(tissue) + 0.4
#   ) %>%
#   ungroup()

# # make plot for each temperature cohort
# # Step 4: Create scatter plot
# p <- plot_ly()

# # Step 5: Add scatter points and rectangles for each timepoint group
# for (time in unique(df$timepoint)) {
#   # Subset data for the current timepoint
#   subset_df <- df %>% filter(timepoint == time)
  
#   # Add scatter points
#   p <- p %>%
#     add_trace(
#       data = subset_df,
#       x = ~avg_nn_time,
#       y = ~tissue,
#       type = "scatter",
#       mode = "markers",
#       marker = list(size = 10),
#       name = paste("Timepoint", time)
#     )
  
#   # Add translucent rectangles for quartiles
#   shapes <- lapply(1:nrow(subset_df), function(i) {
#     list(
#       type = "rect",
#       x0 = subset_df$Q1[i],
#       x1 = subset_df$Q3[i],
#       y0 = subset_df$y_min[i],
#       y1 = subset_df$y_max[i],
#       fillcolor = "blue",
#       opacity = 0.2,
#       line = list(width = 0)
#     )
#   })
  
#   p <- p %>% layout(shapes = shapes)
# }

# # Step 6: Finalize layout
# p <- p %>% layout(
#   title = "Scatter Plot with Quartile Ranges and Ordered Tissue",
#   xaxis = list(title = "Avg NN Time"),
#   yaxis = list(title = "Tissue", categoryorder = "array", categoryarray = unique_tissues),
#   showlegend = TRUE
# )

# # Display the plot
# p
#     data = tissue_stage_df %>% filter(temp==34),
#     x = ~avg_nn_time,             # X-axis: linear stage prediction
#     y = ~as.factor(tissue),                         # Y-axis: avg_nn_time
#     type = "scatter",
#     mode = "markers") %>%
#     layout(
#     shapes = list(
#       list(
#         type = "rect",
#         x0 = x_quartiles[1],  # Lower quartile
#         x1 = x_quartiles[2],  # Upper quartile
#         y0 = -Inf,            # Extend rectangle vertically
#         y1 = Inf,
#         fillcolor = "blue",   # Rectangle color
#         opacity = 0.2,        # Transparency
#         line = list(width = 0)  # No border
#       )
#     )
#   )
# ###########
# # Now let's look at the breakdown by tissue
# print("Made it (273)")
# tissue_stage_df <- col_df %>% 
#                 group_by(embryo_ID, timepoint, temp, tissue) %>%
#                 summarise(avg_nn_time=mean(mean_nn_time), n_cells=n(), .groups = "keep") %>%
#                 ungroup() %>% 
#                 group_by(timepoint, temp, tissue, .groups = "keep") %>%
#                 summarise(q1_nn_time=quantile(avg_nn_time, 0.25), 
#                           q2_nn_time=quantile(avg_nn_time, 0.75),
#                           med_nn_time=median(avg_nn_time), 
#                           cells_per_embryo=mean(n_cells), 
#                           .groups = "drop_last") %>%
#                 ungroup() %>%
#                 mutate(linear_stage_prediction= 6 + (timepoint - 6)*(0.055*temp-0.57)) 

# # Step 1: Determine sorting order based on timepoint = 24
# ordering <- df %>%
#   filter(timepoint == 24) %>%
#   arrange(avg_nn_time) %>%
#   pull(tissue)

# # Step 2: Reorder tissue for all rows based on the derived ordering
# df <- df %>%
#   mutate(tissue = factor(tissue, levels = ordering))

# # Get the unique order of tissues
# unique_tissues <- levels(df$tissue)

# # Step 3: Dynamically assign Y positions for the rectangles
# df <- df %>%
#   group_by(timepoint, tissue) %>%
#   mutate(
#     y_min = as.numeric(tissue) - 0.4,
#     y_max = as.numeric(tissue) + 0.4
#   ) %>%
#   ungroup()

# # make plot for each temperature cohort
# # Step 4: Create scatter plot
# p <- plot_ly()

# # Step 5: Add scatter points and rectangles for each timepoint group
# for (time in unique(df$timepoint)) {
#   # Subset data for the current timepoint
#   subset_df <- df %>% filter(timepoint == time)
  
#   # Add scatter points
#   p <- p %>%
#     add_trace(
#       data = subset_df,
#       x = ~avg_nn_time,
#       y = ~tissue,
#       type = "scatter",
#       mode = "markers",
#       marker = list(size = 10),
#       name = paste("Timepoint", time)
#     )
  
#   # Add translucent rectangles for quartiles
#   shapes <- lapply(1:nrow(subset_df), function(i) {
#     list(
#       type = "rect",
#       x0 = subset_df$Q1[i],
#       x1 = subset_df$Q3[i],
#       y0 = subset_df$y_min[i],
#       y1 = subset_df$y_max[i],
#       fillcolor = "blue",
#       opacity = 0.2,
#       line = list(width = 0)
#     )
#   })
  
#   p <- p %>% layout(shapes = shapes)
# }

# # Step 6: Finalize layout
# p <- p %>% layout(
#   title = "Scatter Plot with Quartile Ranges and Ordered Tissue",
#   xaxis = list(title = "Avg NN Time"),
#   yaxis = list(title = "Tissue", categoryorder = "array", categoryarray = unique_tissues),
#   showlegend = TRUE
# )

# p

# browser()

# tissue_stage_34C <- plot_ly(
#   data = tissue_stage_df %>% filter(temp==34),
#   x = ~avg_nn_time,             # X-axis: linear stage prediction
#   y = ~as.factor(tissue),                         # Y-axis: avg_nn_time
#   type = "scatter",
#   mode = "markers",                         # Only markers
#   error_x = list(
#     type = "data",
#     array = ~std_nn_time,                   # Error bar values
#     visible = TRUE,
#     color="black"
#   ),
#   marker = list(
#     size = 7
#   ),
#   color = ~as.factor(timepoint),                            # Color points by "temp"
#   colors = rev("RdYlBu")                         # Discrete colormap
# ) %>%
#   layout(
#     title = "Transcriptional Stage by Tissue",
#     xaxis = list(title = "Average NN Time"),
#     yaxis = list(title = "Tissue"),
#     showlegend = TRUE                       # Show legend for colors
#   )