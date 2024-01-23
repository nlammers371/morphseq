subcluster_cds = function(cds,
                          recursive_subcluster = FALSE,
                          partition_name = NULL,
                          num_dim = NULL,
                          max_components = 3,
                          resolution_fun = NULL,
                          max_num_cells = NULL,
                          min_res=5e-6,
                          max_res=1e-5,
                          cluster_k=20) {
  
  message("Clustering all cells")
  
  if (is.null(max_num_cells)){
    max_num_cells = ncol(cds)
  }
  
  if (is.null(resolution_fun)){
    resolution_fun = function(num_cells) {
      resolution = approxfun(c(0, log10(max_num_cells)), c(min_res, max_res))(log10(num_cells))
      reflected_resolution = (max_res - resolution) + min_res
      return (reflected_resolution)
    }
    #resolution_fun(min_res)
  }
  
  partition_resolution = resolution_fun(ncol(cds))
  message(paste("Clustering", ncol(cds), "cells at resolution =", partition_resolution))
  cds = monocle3::cluster_cells(cds, resolution = partition_resolution, k=cluster_k)
  colData(cds)$cluster = monocle3::clusters(cds)
  colData(cds)$res = partition_resolution
  partitions = unique(monocle3::partitions(cds))
  
  # if (length(partitions) > 1) {
  #   message(paste0("This could be split up further into more partitions. Num partitions = ", length(partitions)))
  # }
  
  # if we want to recursively subcluster
  if (length(partitions) > 1 & recursive_subcluster) {
    
    partition_res = lapply(partitions, function(partition) {
      if (is.null(partition_name)) {
        next_partition_name = partition
      } else {
        next_partition_name = paste0(partition_name, "_",  partition)
      }
      print(next_partition_name)
      message(paste("Constructing sub-UMAP for partition", next_partition_name))
      cds = cds[, monocle3::partitions(cds) == partition]
      
      RhpcBLASctl::blas_set_num_threads(num_threads)
      RhpcBLASctl::omp_set_num_threads(num_threads)
      
      cds = suppressMessages(suppressWarnings(preprocess_cds(cds))) %>%
        align_cds(residual_model_formula_str = "~log.n.umi") %>%
        suppressMessages(suppressWarnings(reduce_dimension(max_components = max_components,
                                                           preprocess_method="Aligned",
                                                           umap.fast_sgd=TRUE,
                                                           cores=num_threads)))
      
      RhpcBLASctl::blas_set_num_threads(1)
      RhpcBLASctl::omp_set_num_threads(1)
      
      cds = subcluster_cds(cds,
                           partition_name = next_partition_name,
                           recursive_subcluster = recursive_subcluster,
                           num_dim = num_dim,
                           max_components = max_components,
                           resolution_fun = resolution_fun,
                           max_num_cells = max_num_cells,
                           min_res=min_res,
                           max_res=max_res,
                           cluster_k=cluster_k)
      
      # otherwise sometimes the matrix columns dim don't match when
      # trying to keep the reduced dims in the combine
      reducedDims(cds)$PCA = NULL
      reducedDims(cds)$Aligned = NULL
      cds
    })
    # undebug(combine_cds)
    
    cds = combine_cds(partition_res, keep_reduced_dims = T)
    
  } else {
    
    # save the umap coordinates
    partion_umap_coords = reducedDims(cds)[["UMAP"]]
    num_components = dim(partion_umap_coords)[[2]]
    for (i in 1:num_components) {
      name = paste0("partition_umap", num_components, "d_", i)
      colData(cds)[[name]] = reducedDims(cds)[["UMAP"]][,i]
    }
    
    # if it doesn't have an error, then it hasn't gone through combine cds
    has_partitions = tryCatch(monocle3::partitions(cds),
                              error = function(e) {NULL})
    
    if (!is.null(has_partitions)) {
      if (is.null(partition_name)) {
        colData(cds)$partition = monocle3::partitions(cds)
        colData(cds)$cluster = monocle3::clusters(cds)
        colData(cds)$cell_state = monocle3::clusters(cds)
      } else {
        colData(cds)$partition = as.character(partition_name) #
        # colData(cds)$partition = paste0(partition_name, "_",  monocle3::partitions(cds))
        colData(cds)$cluster = monocle3::clusters(cds)
        colData(cds)$cell_state = paste0(partition_name, "-", monocle3::clusters(cds))
      }
    }
  }
  
  return(cds)
}

is_ref = function(cds) {
  
  colData(cds)$is_ref = ifelse(grepl("ctrl", colData(cds)$gene_target), T, F)
  cds = cds[, colData(cds)$is_ref == T]
  return(cds)
  
}


# how to save a plot 
# fig = plot_cells_3d(cds, color_cells_by = "cell_type_sub")
# saveWidget(fig, file = paste0("partition_", i, "_ref_cts.html"))


add_umap_coords = function(cds, prefix = "") {
  colData(cds)[[paste0(prefix, "umap3d_1")]] = reducedDims(cds)[["UMAP"]][,1]
  colData(cds)[[paste0(prefix, "umap3d_2")]] = reducedDims(cds)[["UMAP"]][,2]
  colData(cds)[[paste0(prefix, "umap3d_3")]] = reducedDims(cds)[["UMAP"]][,3]
  return(cds)
}



fix_coldata = function(cds) {
  
  colData(cds)$cell_type_sub_before = colData(cds)$cell_type_sub
  colData(cds)$cell_type_sub = colData(cds)$cell_type_sub_new_revised 
  
  colnames_to_keep = c("cell", "expt", "embryo", "Oligo", "Size_Factor", "n.umi", 
                       "hash_plate", "hash_well", "pcr_plate", 
                       "perc_mitochondrial_umis", "top_to_second_best_ratio", 
                       "hash_umis", "drug", "drug_target", "gene_target", 
                       "temp", "cell_type_sub", "cell_type_sub_before", "tissue", 
                       "germ_layer", "major_group", "partition", "cell_state", 
                       "umap3d_1", "umap3d_2", "umap3d_3", 
                       "subumap3d_1", "subumap3d_2", "subumap3d_3",
                       "partition_umap3d_1","partition_umap3d_2","partition_umap3d_3")
  
  new_coldata = colData(cds) %>% as.data.frame() %>% 
    select(all_of(colnames_to_keep)) %>% 
    mutate(temp = "28C", 
           perturbation = ifelse(is.na(drug), gene_target, as.character(drug)), 
           project = case_when(
             grepl("CHEM", expt) ~ "CHEM", 
             grepl("GAP", expt) ~ "GAP",
             grepl("HF", expt) ~ "HF"))
  
  new_colnames = colnames(new_coldata)
  old_colnames = colnames(colData(cds))
  colnames_to_remove = setdiff(old_colnames, new_colnames)
  
  for (colname in colnames_to_remove) {
    colData(cds)[[colname]] = NULL
  }
  
  return(cds)
  
}


fix_coldata = function(cds) {
  
  colData(cds)$cell_type_sub_before = colData(cds)$cell_type_sub
  colData(cds)$cell_type_sub = colData(cds)$cell_type_sub_new_revised 
  
  colnames_to_keep = c("cell", "expt", "embryo", "Oligo", "Size_Factor", "n.umi", 
                       "log.n.umi", "hash_plate", "hash_well", "pcr_plate", 
                       "perc_mitochondrial_umis", "top_to_second_best_ratio", 
                       "timepoint", "hash_umis", "drug", "drug_target", "gene_target", 
                       "temp", "cell_type_sub", "cell_type_sub_before", "tissue", 
                       "germ_layer", "major_group", "partition", "cell_state", 
                       "umap3d_1", "umap3d_2", "umap3d_3", 
                       "subumap3d_1", "subumap3d_2", "subumap3d_3",
                       "partition_umap3d_1","partition_umap3d_2","partition_umap3d_3")
  
  new_coldata = colData(cds) %>% as.data.frame() %>% 
    select(any_of(colnames_to_keep)) %>% 
    mutate(temp = ifelse(is.na(temp), "28C", as.character(temp)), 
           perturbation = ifelse(is.na(drug), gene_target, as.character(drug)), 
           project = case_when(
             grepl("CHEM", expt) ~ "CHEM", 
             grepl("GAP", expt) ~ "GAP",
             grepl("HF", expt) ~ "HF"))
  
  new_colnames = colnames(new_coldata)
  old_colnames = colnames(colData(cds))
  colnames_to_remove = setdiff(old_colnames, new_colnames)
  
  for (colname in colnames_to_remove) {
    colData(cds)[[colname]] = NULL
  }
  
  return(cds)
  
}


