
plot_cell_type_perturb_kinetics_new = function(perturbation_ccm,
                                           cell_groups=NULL,
                                           start_time=NULL,
                                           stop_time=NULL,
                                           interval_step=1,
                                           log_abund_detection_thresh=-3,
                                           delta_log_abund_loss_thresh=0,
                                           interval_col="timepoint",
                                           q_val=0.01,
                                           log_scale=TRUE,
                                           control_ccm=perturbation_ccm,
                                           control_start_time=start_time,
                                           control_stop_time=control_stop_time,
                                           group_nodes_by = "cell_type",
                                           ...){
  
  colData(perturbation_ccm$ccs)[,interval_col] = as.numeric(colData(perturbation_ccm$ccs)[,interval_col])

  if (is.null(start_time))
    start_time = min(colData(perturbation_ccm$ccs)[,interval_col])
  if (is.null(stop_time))
    stop_time = max(colData(perturbation_ccm$ccs)[,interval_col])

  wt_timepoint_pred_df = hooke:::estimate_abundances_over_interval(perturbation_ccm, start_time, stop_time, knockout=FALSE, interval_col=interval_col, interval_step=interval_step, ...)
  ko_timepoint_pred_df = hooke:::estimate_abundances_over_interval(perturbation_ccm, start_time, stop_time, knockout=TRUE, interval_col=interval_col, interval_step=interval_step, ...)

  timepoints = seq(start_time, stop_time, interval_step)

  perturb_vs_wt_nodes = tibble(t1=timepoints) %>%
    mutate(comp_abund = purrr::map(.f = platt:::compare_ko_to_wt_at_timepoint,
                                   .x = t1,
                                   perturbation_ccm=perturbation_ccm,
                                   interval_col=interval_col,
                                   wt_pred_df = wt_timepoint_pred_df,
                                   ko_pred_df = ko_timepoint_pred_df)) %>% tidyr::unnest(comp_abund)

  extant_wt_tbl = get_extant_cell_types(perturbation_ccm,
                                        start_time,
                                        stop_time,
                                        log_abund_detection_thresh=log_abund_detection_thresh,
                                        knockout=FALSE,
                                        ...)

  sel_ccs_counts = normalized_counts(perturbation_ccm$ccs, norm_method="size_only", pseudocount=0)
  sel_ccs_counts_long = tibble::rownames_to_column(as.matrix(sel_ccs_counts) %>% as.data.frame, var="cell_group") %>%
    pivot_longer(!cell_group, names_to="embryo", values_to="num_cells")

  cell_group_metadata = collect_psg_node_metadata(perturbation_ccm$ccs,
                                                          group_nodes_by=group_nodes_by,
                                                          color_nodes_by=group_nodes_by,
                                                          label_nodes_by=group_nodes_by) %>%
    select(id, cell_group = group_nodes_by)

  sel_ccs_counts_long = left_join(sel_ccs_counts_long,
                                  cell_group_metadata,
                                  by=c("cell_group"="id"))

  sel_ccs_counts_long = left_join(sel_ccs_counts_long,
                                  colData(perturbation_ccm$ccs) %>% as.data.frame %>% select(sample, !!sym(interval_col), knockout, expt, gene_target),
                                  by=c("embryo"="sample"))

  if (is.null(cell_groups) == FALSE){
    sel_ccs_counts_long = sel_ccs_counts_long %>% filter(cell_group %in% cell_groups)
    perturb_vs_wt_nodes = perturb_vs_wt_nodes %>% filter(cell_group %in% cell_groups)
  }

  perturb_vs_wt_nodes = left_join(perturb_vs_wt_nodes,
                                  extant_wt_tbl %>% select(cell_group, !!sym(interval_col), present_above_thresh), by=c("cell_group" = "cell_group", "t1" = "timepoint"))
  
  if (is.null(cell_groups)){
    cell_groups = unique(as.character(perturb_vs_wt_nodes$cell_group)) %>% sort()
  }
  perturb_vs_wt_nodes$cell_group = factor(as.character(perturb_vs_wt_nodes$cell_group), levels=cell_groups)
  sel_ccs_counts_long$cell_group = factor(as.character(sel_ccs_counts_long$cell_group), levels=cell_groups)

  perturb_vs_wt_nodes$t1 = as.numeric(perturb_vs_wt_nodes$t1)
  sel_ccs_counts_long$timepoint = as.numeric(sel_ccs_counts_long$timepoint)
  
  kinetic_plot = ggplot(perturb_vs_wt_nodes, aes(x = !!sym(paste(interval_col, "_x", sep=""))))+ 
    geom_point(aes(x = !!sym(paste(interval_col, "_x", sep="")), 
                   y =1.0*(exp(log_abund_y) + exp(log_abund_detection_thresh))), 
               shape=8, 
               data=perturb_vs_wt_nodes %>% filter(present_above_thresh & delta_log_abund > 0 & delta_q_value < q_val)) +
    geom_point(aes(x = !!sym(paste(interval_col, "_x", sep="")), y =1.0*(exp(log_abund_y) + exp(log_abund_detection_thresh))), 
               shape=8, data=perturb_vs_wt_nodes %>% filter(present_above_thresh & delta_log_abund < 0 & delta_q_value < q_val)) +
    geom_jitter(data=sel_ccs_counts_long,
                aes(x = timepoint, y = num_cells+exp(log_abund_detection_thresh), shape=expt),
                height=0,
                width=2, color = "gray", size=0.5) +
    geom_line(aes(y = exp(log_abund_x) + exp(log_abund_detection_thresh), linetype = "Wild-type")) +
    geom_line(aes(y = exp(log_abund_y) + exp(log_abund_detection_thresh), linetype = "Knockout")) +
    ggh4x::stat_difference(aes(ymin = exp(log_abund_x)+exp(log_abund_detection_thresh), 
                               ymax = exp(log_abund_y) +exp(log_abund_detection_thresh)), alpha=0.5) + 
    scale_fill_manual(values = c("orangered3", "royalblue3", "lightgray")) + 
    facet_wrap(~cell_group, scales="free_y", nrow=2) + monocle3:::monocle_theme_opts() + 
    geom_hline(yintercept=exp(log_abund_detection_thresh), color="lightgrey") + xlab("timepoint") + 
    facet_wrap(~cell_group, scales="free_y", nrow=1)
  
  if (log_scale)
    kinetic_plot = kinetic_plot + scale_y_log10()

  return(kinetic_plot)
}
