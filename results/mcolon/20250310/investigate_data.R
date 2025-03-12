
library("monocle3")
library("Matrix")
library("hooke")


ref_cds = load_monocle_objects("/net/trapnell/vol1/home/duran/seahub/all_reference_v2.1.0/reference_cds_v2.1.0", matrix_control = list(matrix_class="BPCells", matrix_path="/net/gs/vol1/home/mdcolon/nobackups/mdcolon/BPCells/"))

DEG_by_assembly_group <- hooke:::aggregated_expr_data(ref_cds, group_cells_by = "assembly_group")
write.csv(DEG_by_assembly_group, "./data/ref_cds/DEG_by_assembly.csv", row.names = FALSE)

counts  <- as(counts(ref_cds), 'dgCMatrix')
Matrix::writeMM( t(counts), "./data/ref_cds/gene_count.mtx" )
write.csv(data.frame(pData(ref_cds)), "./data/ref_cds/df_cell.csv")
write.csv(data.frame(fData(ref_cds)), "./data/ref_cds/df_gene.csv")





ref_cds = load_monocle_objects("/net/trapnell/vol1/home/duran/seahub/all_reference_v2.1.0/partition_CNS_progenitor_ref_cds_v2.1.0", matrix_control = list(matrix_class="BPCells", matrix_path="/net/gs/vol1/home/mdcolon/nobackups/mdcolon/BPCells"))
counts  <- as(counts(ref_cds), 'dgCMatrix')

ref_cds <- cluster_cells(ref_cds, resolution= .00020, verbose=TRUE) 

ref_cds$clusters <-  clusters(ref_cds)
plot <- plot_cells(ref_cds, color_cells_by = "clusters")
ggsave("/net/gs/vol1/home/mdcolon/proj/zfish_spatial/results/progenitors_plot.png", plot = plot)

ref_agr_exp_data <- hooke:::aggregated_expr_data(ref_cds, group_cells_by = "clusters")

write.csv(ref_agr_exp_data, "./data/ref_cds/CNS_progenitor/DEG_37clusters.csv", row.names = FALSE)

counts  <- as(counts(ref_cds), 'dgCMatrix')
Matrix::writeMM(t(counts), "./data/ref_cds/CNS_progenitor/gene_count.mtx")
write.csv(data.frame(pData(ref_cds)), "./data/ref_cds/CNS_progenitor/df_cell.csv")
write.csv(data.frame(fData(ref_cds)), "./data/ref_cds/CNS_progenitor/df_gene.csv")



library(ggplot2)

loc = read.csv("/net/gs/vol1/home/mdcolon/proj/zfish_spatial/data/ref_cds/CNS_progenitor/df_cell_wloc.csv")
ref_cds = load_monocle_objects("/net/trapnell/vol1/home/duran/seahub/all_reference_v2.1.0/partition_CNS_progenitor_ref_cds_v2.1.0")

loc = loc[-(1)]

for (colname in colnames(loc)) {
  # Add the column from loc to colData of ref_cds
  print(colname)
  ref_cds@colData[[colname]] <- as.character(loc[[colname]])

}



# Start recording plots to a PDF file
pdf("/net/gs/vol1/home/mdcolon/proj/zfish_spatial/data/ref_cds/CNS_progenitor/plots_of_location.pdf", width=120)


# Plot the initial cells
plot_cells(ref_cds, color_cells_by = "cell_type", label_cell_group=FALSE)

plot_cells(ref_cds, color_cells_by = "sub_clusters", label_cell_group=FALSE)

# Loop through each column in loc and plot cells colored by each column
for (colname in colnames(loc)) {
  # Plot cells colored by the current column
  print(colname)
  p <- plot_cells(ref_cds, color_cells_by = colname, label_cell_group=FALSE) +
    ggtitle(paste("Location:", colname)) 

  print(p)

}

# End recording plots to the PDF file
dev.off()

ggsave("/net/gs/vol1/home/mdcolon/proj/zfish_spatial/results/progenitors_test_plot.png", plot = plot_cells(ref_cds, color_cells_by = "cell_type"))


pdf("/net/gs/vol1/home/mdcolon/proj/zfish_spatial/data/ref_cds/CNS_progenitor/plots_celltypes.pdf", width=10)
# Plot the initial cells
plot_cells(ref_cds, color_cells_by = "cell_type", label_cell_group=FALSE)

dev.off()





library("monocle3", lib.loc="/net/gs/vol3/software/modules-sw/R/4.3.2/Linux/Ubuntu22.04/x86_64/lib/R/library")