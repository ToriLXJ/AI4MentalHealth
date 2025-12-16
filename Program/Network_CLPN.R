################ Packages #############################
library(glmnet)
library(doParallel)
library(foreach)
library(igraph)
library(qgraph)
library(bootnet)
library(ggplot2)
library(dplyr)
library(RColorBrewer)
library(tidyr)
library(ggpubr)
library(glmnet)

################ CLPN Core Estimator Function #############################
clpn_estimator <- function(data, t1_vars_base, t2_vars_base) {
  
  t1_cols <- paste0(t1_vars_base, "_T1")
  t2_cols <- paste0(t2_vars_base, "_T2")
  
  if (!all(c(t1_cols, t2_cols) %in% colnames(data))) {
    return(NULL) 
  }
  
  n_vars <- length(t1_vars_base)
  adjMat_full <- matrix(0, n_vars, n_vars)
  rownames(adjMat_full) <- t1_vars_base
  colnames(adjMat_full) <- t2_vars_base
  
  # Perform LASSO regression for each T2 variable
  for (i in 1:n_vars) {
    t2_var <- t2_cols[i]
    
    x_data <- as.matrix(data[, t1_cols])
    y_data <- data[[t2_var]]
    
    complete_cases <- complete.cases(x_data) & !is.na(y_data)
    x_complete <- x_data[complete_cases, ]
    y_complete <- y_data[complete_cases]
    
    # Check sample size (N > 10*P heuristic)
    if (sum(complete_cases) < 10 * ncol(x_complete)) {
      next
    }
    
    # LASSO Regression
    tryCatch({
      lasso_model <- cv.glmnet(
        x = x_complete,
        y = y_complete,
        family = "gaussian",
        alpha = 1,
        standardize = TRUE,
        nfolds = 5
      )
      
      # Extract coefficients (excluding intercept)
      coefs <- as.numeric(coef(lasso_model, s = lasso_model$lambda.min))[-1]
      adjMat_full[, i] <- coefs
    }, error = function(e) {
      warning(paste("LASSO failed for variable:", t2_var, "Error:", e$message))
    })
  }
  
  return(adjMat_full)
}

################ CLPN Main Analysis Function #############################

analyze_clpn_network <- function(
    t1_data,
    t2_data,
    output_dir,
    lognames,
    var_mapping, # Retained for structure, but not used in the final version
    communities_times,
    community_colors,
    threshold_value = 0.05,
    iter = 1000
) {
  
  ###### 1. Data Preparation and Variable Definition ######
  
  t1_vars_base <- names(t1_data)
  t2_vars_base <- names(t2_data)
  num_t1 <- length(t1_vars_base)
  
  # Explicitly create clpn_data
  clpn_data <- cbind(
    t1_data %>% setNames(paste0(names(.), "_T1")),
    t2_data %>% setNames(paste0(names(.), "_T2"))
  )
  
  ###### 2. Estimate Original CLPN Matrix ######
  cat("--- 2. Estimating LASSO Cross-Lagged Regression Matrix ---\n")
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Unified call to clpn_estimator
  set.seed(123)
  adjMat <- clpn_estimator(
    data = clpn_data, 
    t1_vars_base = t1_vars_base, 
    t2_vars_base = t2_vars_base
  )
  
  if (is.null(adjMat)) {
    stop("Error: Original network estimation failed. Check data for sufficiency.")
  }
  
  cat(paste("LASSO estimation complete. Matrix dimensions:", dim(adjMat), "\n"))
  write.csv(adjMat, file.path(output_dir, "cross_lagged_effects_matrix_full.csv"), row.names = TRUE)
  
  ###### 3. Network Plotting (Full and Thresholded) ######
  
  full_names <- unlist(lognames[t1_vars_base])
  # Assumes 'communities' is available in the calling environment
  item <- rep(communities, times = communities_times) 
  
  cat("--- 3. Plotting Cross-Lagged Network ###### \n")
  
  # 3a. Full Network (including diagonals/autocorrelations if they exist)
  net_clpn_full <- qgraph(
    adjMat,
    labels = t1_vars_base,  
    nodeNames = full_names,  
    groups = item,  
    color = community_colors,
    layout = "spring",  
    posCol = "darkgreen",  
    negCol = "red",  
    legend.cex = 0.5,
    vsize = 5,
    border.width = 1,
    edge.width = 0.5,
    maximum = max(abs(adjMat)),
    minimum = min(abs(adjMat[adjMat != 0])),
    asize = 3,
    label.cex = 0.8,  
    title ="Cross-Lagged Network (T1 -> T2), Full Estimates"
  )
  
  pdf_file_full <- file.path(output_dir, "plot_CLPN_Network_full.pdf")
  pdf(pdf_file_full, width = 12, height = 10); 
  plot(net_clpn_full); 
  dev.off()
  cat(paste0("Full CLPN Network saved to: ", pdf_file_full, "\n"))
  
  # 3b. Thresholded Network (for centrality calculation and final visualization)
  adjMat_noAR <- adjMat
  diag(adjMat_noAR) <- 0 # Remove self-loops for cross-lagged focus
  
  net_clpn <- qgraph(
    adjMat_noAR,
    labels = t1_vars_base,  
    nodeNames = full_names,  
    groups = item,  
    color = community_colors,
    layout = "spring",  
    posCol = "darkgreen",  
    negCol = "red",  
    threshold = threshold_value,
    label.cex = 0.8,  
    title = paste("Cross-Lagged Network (T1 -> T2)\nThreshold =", threshold_value)
  )
  
  pdf_file_final <- file.path(output_dir, "CLPN_Network.pdf")
  pdf(pdf_file_final, width = 12, height = 10); plot(net_clpn); dev.off()
  cat(paste0("Thresholded CLPN Network saved to: ", pdf_file_final, "\n"))
  
  ###### 4. Centrality Calculation and Plotting ######
  
  cat("--- 4. Calculating and Plotting Centrality Measures ---\n")
  
  centrality_res <- centrality_auto(adjMat)$node.centrality
  
  centrality_measures <- centrality_res %>%
    select(OutExpectedInfluence, InExpectedInfluence) %>%
    setNames(names(.))
  
  rownames(centrality_measures) <- t1_vars_base
  
  bridge_results <- bridge(
    net_clpn,
    communities = item,
    useCommunities = "all",
    directed = NULL,
    nodes = paste0(lognames, " (", node_names, ")"),
    normalize = FALSE
  )
  
  bridge_expected_influence <- bridge_results[["Bridge Expected Influence (1-step)"]]
  names(bridge_expected_influence) <- t1_vars_base
  
  all_centrality <- data.frame(
    OutExpectedInfluence = centrality_measures$OutExpectedInfluence,
    InExpectedInfluence = centrality_measures$InExpectedInfluence,
    BridgeExpectedInfluence = bridge_expected_influence
  )
  rownames(all_centrality) <- t1_vars_base
  
  all_centrality_z <- as.data.frame(scale(all_centrality))
  all_centrality_z$Variable <- rownames(all_centrality_z)
  
  node_names <- names(lognames)
  all_centrality_z$VariableLabel <- paste0(lognames, " (", node_names, ")")
  
  # Sort in descending order of the size of BridgeExpectedInfluence
  bridge_order <- order(all_centrality$BridgeExpectedInfluence, decreasing = TRUE)
  all_centrality_z <- all_centrality_z[bridge_order, ]
  
  # Ensure that the VariableLabel is arranged in the order of BridgeExpectedInfluence
  all_centrality_z$VariableLabel <- factor(
    all_centrality_z$VariableLabel,
    levels = all_centrality_z$VariableLabel
  )
  
  # Assign a position number (from 1 to 19, from top to bottom) to each variable
  all_centrality_z$Position <- seq_len(nrow(all_centrality_z))
  
  all_centrality_long <- all_centrality_z %>%
    pivot_longer(
      cols = c(OutExpectedInfluence, InExpectedInfluence, BridgeExpectedInfluence),
      names_to = "Measure",
      values_to = "ZScore"
    ) %>%
    mutate(
      Measure = factor(Measure, 
                       levels = c("OutExpectedInfluence", "InExpectedInfluence", "BridgeExpectedInfluence"),
                       labels = c("Out-Expected Influence", "In-Expected Influence", "Bridge Expected Influence"))
    ) %>%
    arrange(Measure, Position)
  

  my_theme <- theme_minimal(base_size = 16) +
    theme(
      axis.text.y = element_text(size = 14, hjust = 1, margin = margin(r = 5), color = "black"),
      axis.text.x = element_text(size = 14, color = "black"),
      axis.title.x = element_text(margin = margin(t = 10), size = 15, color = "black"),
      axis.title.y = element_blank(),
      
      strip.background = element_rect(fill = "gray95", color = "lightgray", linewidth = 1),
      strip.text = element_text(size = 14, face = "plain", color = "black"),
      panel.spacing = unit(0.5, "lines"),
      
      panel.border = element_rect(color = "lightgray", fill = NA, linewidth = 0.8),
      
      panel.grid.major.x = element_line(color = "gray90", linewidth = 0.3),
      panel.grid.minor.x = element_blank(),
      panel.grid.major.y = element_line(color = "gray90", linewidth = 0.3),
      
      legend.position = "none",
      
      plot.margin = margin(20, 20, 20, 20)
    )
  
  centrality_plot <- ggplot(all_centrality_long, 
                            aes(x = ZScore, y = reorder(VariableLabel, -Position))) +

    geom_vline(xintercept = 0, linetype = "dashed", color = "gray70", linewidth = 0.3) +
    geom_point(aes(shape = Measure), size = 2.5, color = "black", fill = "white") +
    
    # Within each plane, connect the points in sequence from top to bottom along the Y-axis direction.
    geom_path(aes(group = Measure), color = "black", linewidth = 0.8, linetype = "solid") +
    
    facet_grid(. ~ Measure, scales = "free_x") +
    
    labs(
      x = "Cross-Lagged Centrality (z-scored)",
      y = NULL
    ) +

    my_theme +

    scale_shape_manual(
      values = c("Out-Expected Influence" = 16,
                 "In-Expected Influence" = 17,
                 "Bridge Expected Influence" = 15)
    )
  
  pdf_file <- file.path(output_dir, "plot_centrality_comparison.pdf")
  pdf(pdf_file, width = 12, height = 8)
  print(centrality_plot)
  dev.off()
  
  cat("Centrality comparison plot saved to::", pdf_file, "\n")

  print(centrality_plot)
  
  ###### 5. Network Stability Analysis (Bootnet) ######
  
  cat("--- 5. CLPN Network Stability Analysis ---\n")
  
  # Wrapper to pass base variable names to clpn_estimator
  clpn_estimator_wrapper <- function(data) {
    clpn_estimator(data, t1_vars_base, t2_vars_base)
  }
  
  # 5a. Nonparametric Bootstrap (Edge and Centrality Confidence Intervals)
  set.seed(79)
  # estimateNetwork uses the wrapper to get the original network object
  net_boot <- estimateNetwork(clpn_data, fun = clpn_estimator_wrapper, labels = t1_vars_base, directed = TRUE)
  
  set.seed(7)
  boot_nonpara <- bootnet(net_boot, type = "nonparametric", nBoots = iter, communities = item,
                          directed = TRUE,
                          statistics = c("edge", "outExpectedInfluence", "inExpectedInfluence", "bridgeExpectedInfluence"),
                          ncores = detectCores() - 1)
  
  # Plot edge CIs
  plot_boot_edge <- plot(boot_nonpara, order = "sample", labels=FALSE, meanColor = NA) + 
    xlab("Edge Weight") + 
    theme(plot.title = element_text(hjust = 0.1, size = 12, face = "bold"),
          strip.text = element_blank(),
          axis.text.x = element_text(size = 10))
  
  pdf_file_edge <- file.path(output_dir, "plot_boot_edge.pdf")
  pdf(pdf_file_edge, width = 12, height = 8); plot(plot_boot_edge); dev.off()
  cat(paste("Edge confidence interval plot saved to:", pdf_file_edge, "\n"))
  print(plot_boot_edge)
  
  # Plot centrality difference (nonparametric)
  plot_centrality_clpn <- annotate_figure(ggarrange(
    plot(boot_nonpara, "outExpectedInfluence", plot = "difference", order = "sample") + theme(plot.title=element_text(hjust=0.5,size=5,face="bold")),
    plot(boot_nonpara, "inExpectedInfluence", plot = "difference", order = "sample") + theme(plot.title=element_text(hjust=0.5,size=5,face="bold")),
    plot(boot_nonpara, "bridgeExpectedInfluence", plot = "difference", order = "sample") + theme(plot.title=element_text(hjust=0.5,size=5,face="bold")),
    ncol=3, nrow=1, common.legend=FALSE)
  )
  
  pdf_file_diff <- file.path(output_dir, "plot_boot_centrality.pdf")
  pdf(pdf_file_diff, width = 16, height = 8); plot(plot_centrality_clpn); dev.off()
  
  cat(paste("Centrality difference plot saved to:", pdf_file_diff, "\n"))
  print(plot_centrality_clpn)
  
  # 5b. Case-Dropping Bootstrap (CS-Coefficient and Centrality Stability)
  set.seed(79)
  boot_case <- bootnet(net_boot, type = "case", nBoots = iter, directed = TRUE, communities = item,
                       statistics = c("bridgeExpectedInfluence", "edge", "outExpectedInfluence", "inExpectedInfluence"),
                       ncores = detectCores() - 1)
  
  # Plot centrality stability
  plot_boot_case <- plot(boot_case, statistics = c("bridgeExpectedInfluence", "edge","outExpectedInfluence", "inExpectedInfluence")) + 
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 12), 
          axis.text = element_text(size = 8), 
          axis.title = element_text(size = 8))
  
  pdf_file_case <- file.path(output_dir, "plot_boot_case.pdf")
  pdf(pdf_file_case, width = 12, height = 8); plot(plot_boot_case); dev.off()
  
  cat(paste("Case-dropping stability plot saved to:", pdf_file_case, "\n"))
  print(plot_boot_case)
  
  # Calculate CS coefficients
  cs_results <- corStability(boot_case)
  cat("Correlation Stability (CS) Coefficients:\n")
  print(cs_results)
  
  cat("--- 6. Performing Structural Analysis for Excel Output ---\n")
  ###### 6a. Overall Connectivity Summary ######
  num_nonzero_edges <- sum(abs(adjMat_noAR) > 1e-10)
  num_positive_edges <- sum(adjMat_noAR > 0)
  num_negative_edges <- sum(adjMat_noAR < 0)
  percentage_positive <- ifelse(num_nonzero_edges > 0, num_positive_edges / num_nonzero_edges * 100, 0)
  
  overall_summary_df <- data.frame(
    Measure = c("Total Non-Zero Edges", "Positive Edges", "Negative Edges", "Percentage Positive"),
    Value = c(num_nonzero_edges, num_positive_edges, num_negative_edges, round(percentage_positive, 2))
  )
  
  ###### 6b. Connectivity per Node ######
  
  # Initialize the data frame for node connectivity summary
  node_connectivity_df <- data.frame(Node = t1_vars_base) %>%
    mutate(
      Community = item,
      Outgoing_NonZero = apply(adjMat_noAR, 1, function(x) sum(abs(x) > 1e-10)),
      Outgoing_Positive = apply(adjMat_noAR, 1, function(x) sum(x > 0)),
      Outgoing_Negative = apply(adjMat_noAR, 1, function(x) sum(x < 0)),
      Incoming_NonZero = apply(adjMat_noAR, 2, function(x) sum(abs(x) > 1e-10)),
      Incoming_Positive = apply(adjMat_noAR, 2, function(x) sum(x > 0))
    )
  
  ###### 6c. Inter-Community Connectivity ######
  
  community_names <- unique(item)
  inter_community_edges_df <- data.frame(
    Source_Community = character(),
    Target_Community = character(),
    NonZero_Edges = numeric()
  )
  
  for (source_comm in community_names) {
    for (target_comm in community_names) {
      source_indices <- which(item == source_comm)
      target_indices <- which(item == target_comm) 
      
      if (length(source_indices) > 0 && length(target_indices) > 0) {
        sub_matrix <- adjMat_noAR[source_indices, target_indices, drop = FALSE]
        nonzero_count <- sum(abs(sub_matrix) > 1e-10)
        
        inter_community_edges_df <- rbind(inter_community_edges_df, data.frame(
          Source_Community = source_comm,
          Target_Community = target_comm,
          NonZero_Edges = nonzero_count
        ))
      }
    }
  }
  
  # Pivot for a matrix-like output in Excel
  inter_community_pivot_df <- inter_community_edges_df %>% 
    pivot_wider(names_from = Target_Community, values_from = NonZero_Edges)
  
  ###### 6d. Strongest Edges ######
  
  adjMat_abs <- abs(adjMat_noAR)
  sorted_indices <- order(adjMat_abs, decreasing = TRUE)
  nonzero_edge_indices <- sorted_indices[adjMat_abs[sorted_indices] > 1e-10]
  num_strongest_to_show <- min(30, length(nonzero_edge_indices)) # Show top 30 or all non-zero
  strongest_edge_indices <- nonzero_edge_indices[seq_len(num_strongest_to_show)]
  pos_indices <- arrayInd(strongest_edge_indices, dim(adjMat_noAR), useNames = TRUE)
  
  strongest_edges_df <- data.frame(
    Rank = 1:num_strongest_to_show,
    Node_Out = t1_vars_base[pos_indices[, 1]],
    Node_In = t2_vars_base[pos_indices[, 2]],
    Beta_Coefficient = adjMat_noAR[strongest_edge_indices],
    Domain_Out = item[pos_indices[, 1]],
    Domain_In = item[pos_indices[, 2]]
  )
  
  
  cat("Summary of CLPN and strongest edges complete.\n")
  write.csv(strongest_edges_df, file.path(output_dir, "summary_clpn_strongestEdge.csv"), row.names = TRUE)
  
  ###### 7. Save All Results to Excel ######
  
  return(list(
    Overall_Summary = overall_summary_df,
    Centrality_ZScores = all_centrality_z,
    Node_Connectivity = node_connectivity_df,
    Community_Connectivity = inter_community_pivot_df,
    Strongest_Edges = strongest_edges_df,
    CS_Coefficients = cs_results
  ))
}