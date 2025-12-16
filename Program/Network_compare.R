# Network_compare.R
# ==============================================================================
# 功能：使用 NetworkComparisonTest (NCT) 包比较 T1 和 T2 的 MGM 网络。
# ==============================================================================

# 确保加载了必要的包
library(NetworkComparisonTest)
library(mgm)

# --- 1. 定义 MGM 估计器函数 ---
# 这是 NCT 要求的回调函数，用于在每次置换时估计网络。
# 参数名称必须与 estimatorArgs 中的名称保持一致。
mgm_estimator <- function(data) {
  
  type_vec <- ifelse(
    sapply(data, function(x) length(unique(x)) <= 5), "c", "g"
  )
  level_vec <- sapply(data, function(x) length(unique(x)))
  level_vec[type_vec == "g"] <- 1
  
  dfmat <- as.matrix(data)
  
  mgm_model <- mgm(
    data = dfmat,
    type = type_vec,   # 使用全局变量
    level = level_vec, # 使用全局变量
    lambdaSel = "EBIC",
    lambdaGam = 0.25,
    scale = TRUE,
    verbosity = 0
  )
 
  result <- list(
    graph = mgm_model$pairwise$wadj,  # 权重邻接矩阵
    intercepts = mgm_model$intercepts,  # 截距项
    signs = mgm_model$pairwise$signs,  # 边的符号
    type = type_vec,
    level = level_vec
  )
  
  return(result)
}

# --- 2. 定义 NCT 比较核心函数 ---
#' 使用 NCT::NCT() 比较 T1 和 T2 网络
#' 
#' @param D1 The original data matrix for T1.
#' @param D2 The original data matrix for T2.
#' @param type_vec A vector of variable types used for MGM modeling.
#' @param level_vec A vector of variable levels used for MGM modeling.
#' @param output_dir The directory to save comparison results.
#' @param n_iter The number of iterations (permutations, it).
#' @return A list containing the network comparison results.
compare_two_networks_nct <- function(D1, D2, type_vec, level_vec, output_dir, n_iter = 2) {
  
  cat("Starting network comparison using NetworkComparisonTest::NCT()...\n")
  
  # 核心 NCT 函数调用
  comparison_results <- NCT(
    data1 = D1,
    data2 = D2,
    
    # 核心参数设置
    it = n_iter,                # 置换次数 (Permutations)
    paired = TRUE,              # 假设为配对样本 (同一组测量两次)
    weighted = TRUE,            # 使用边权重进行比较
    
    # 测试项目设置 (根据您的意图，只测试全局强度和边差异的最大值)
    test.edges = FALSE,         # ⚠️ 设为 FALSE 可加速测试；若要测试单边差异，设为 TRUE
    test.centrality = FALSE,    # ⚠️ 设为 FALSE 可加速测试；若要测试中心性，设为 TRUE
    # centrality = c("strength"), # 即使 test.centrality=FALSE, 也保留此参数以防万一
    
    # 估计器设置 (用于指定 MGM 模型)
    estimator = mgm_estimator,
    # estimatorArgs = list(
    #   type = type_vec,
    #   level = level_vec,
    #   lambdaSel = "EBIC",
    #   lambdaGam = 0.25,
    #   scale = TRUE
    # ),
    
    verbose = TRUE
  )
  
  cat("NCT comparison finished.\n")
  
  # --- 3. 保存和输出 ---
  
  # 保存 R 对象
  comparison_file <- file.path(output_dir, "NCT_comparison_T1_T2.rds")
  saveRDS(comparison_results, comparison_file)
  cat("NCT results saved:", comparison_file, "\n")

  
  glstrinv_pval <- comparison_results$glstrinv.pval
  nwinv_pval <- comparison_results$nwinv.pval
  
  cat("\n--- Global Invariance Test Results ---\n")
  cat("Global Strength Invariance p-value (glstrinv.pval):", glstrinv_pval, "\n")
  cat("Network Structure Invariance p-value (nwinv.pval):", nwinv_pval, "\n")
  
  # Conclusion based on Global Strength Invariance
  if (glstrinv_pval < 0.05) {
    cat("Conclusion: Networks show **significant** difference in global strength (p =", glstrinv_pval, ") ⚠️\n")
  } else {
    cat("Conclusion: Networks show **no significant** difference in global strength (p =", glstrinv_pval, ")\n")
  }
  
  if (comparison_results$test.edges) {
    pdf_file <- file.path(output_dir, "NCT_Edge_Differences.pdf")
    pdf(pdf_file, width = 10, height = 10)
    plot(comparison_results, what = "edge")
    dev.off()
    cat(paste0("NCT difference plot saved to: ", pdf_file, "\n"))
  } else {
    cat("test.edges = FALSE, skipping edge difference plot.\n")
  }
  
  # 4. Save All Results to Excel File
  excel_file <- file.path(output_dir, "NCT_Summary_Results.xlsx")
  wb <- openxlsx::createWorkbook()
  
  # a) Global Metrics (Workbook Sheet 1)
  global_df <- data.frame(
    Metric = c("Global_Strength_p", "Network_Structure_p"),
    P_Value = c(glstrinv_pval, nwinv_pval),
    Threshold = 0.05,
    Significant_Difference = ifelse(c(glstrinv_pval, nwinv_pval) < 0.05, "Yes", "No")
  )
  openxlsx::addWorksheet(wb, "Global_Invariance")
  openxlsx::writeData(wb, "Global_Invariance", global_df)
  
  # b) Edge Differences (Workbook Sheet 2)
  if (comparison_results$test.edges) {
    edge_df <- as.data.frame(comparison_results$einv.pvals)
    colnames(edge_df) <- c("Node1", "Node2", "p_value", "p_corrected")
    openxlsx::addWorksheet(wb, "Edge_Differences")
    openxlsx::writeData(wb, "Edge_Differences", edge_df)
  }
  
  # c) Centrality Differences (Workbook Sheet 3)
  if (comparison_results$test.centrality) {
    centrality_df <- as.data.frame(comparison_results$diffcen.pval)
    centrality_df$Node <- rownames(centrality_df)
    openxlsx::addWorksheet(wb, "Centrality_Differences")
    openxlsx::writeData(wb, "Centrality_Differences", centrality_df)
  }
  
  openxlsx::saveWorkbook(wb, excel_file, overwrite = TRUE)
  cat(paste0("All results saved to Excel: ", excel_file, "\n"))
  
  return(comparison_results)
}