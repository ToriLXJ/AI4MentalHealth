################## packages ############################
library(glmnet)
library(qgraph)
library(bootnet)
library(NetworkComparisonTest)
library(EGAnet)
library(mgm)
library(networktools)
library(ggplot2)
library(dplyr)
library(tidyr)
library(doParallel)
library(igraph)
library(RColorBrewer)
library(here)

################## Config ###############################
set.seed(123)
input_dir <- "INS_prediction/Input/df_all_ins.csv"
output_base_dir <- "INS_prediction/Output/network_analysis"
communities <- c('Loneliness', 'Insomnia', 'Somatic symptoms')
var_extract <- c('loneliness', 'insomnia', 'somatic')
var_mapping = list(
  loneliness = "LON",
  insomnia = "INS", 
  somatic = "SOM"
)

community_colors <- brewer.pal(3, "Set2")
communities_times <- c(6, 7, 6)  # 每个社区的变量数量

lognames <- c(
  "LON1" = "Feelings of loneliness",
  "LON2" = "Lack of companionship",
  "LON3" = "Sense of isolation",
  "LON4" = "Social connection deficit",
  "LON5" = "No one to confide in",
  "LON6" = "Social isolation",
  "INS1" = "Hard to fall asleep",
  "INS2" = "Wake up at night and hard to sleep",
  "INS3" = "Difficulty in getting up early",
  "INS4" = "Poor sleep quality",
  "INS5" = "Daytime fatigue",
  "INS6" = "Sleep impacted life",
  "INS7" = "Worried about the sleep",
  "SOM1" = "Headache",
  "SOM2" = "Nausea",
  "SOM3" = "Fatigue",
  "SOM4" = "Muscle pain",
  "SOM5" = "Stomach ache",
  "SOM6" = "Weakness"
)

################## Data prepare ###############################
cat("Start the data cleaning process...\n")

source(here("INS_prediction", "Program", "Network_cleandata.R"))

data_prepared <- prepare_data(
  input_dir = input_dir,
  main_vars = var_extract,
  con_vars = c("gender_T1"),
  time_points = c("T1", "T2"),
  var_mapping = var_mapping,
  na_handling = "median"
)

t1_data <- data_prepared$complete_t1
t2_data <- data_prepared$complete_t2

clpn_data <- cbind(
  t1_data %>% setNames(paste0(names(.), "_T1")),
  t2_data %>% setNames(paste0(names(.), "_T2"))
)

################## Cross-sectional Network analysis ############################

cat("Start the complete network analysis process...\n")

source(here("INS_prediction", "Program", "Network_basicnet.R"))

# 1. 分析T1网络
cat("=== Analyze T1 network ===\n")

t1_output_dir <- file.path(output_base_dir, "T1")
# if (!dir.exists(t1_output_dir)) dir.create(t1_output_dir, recursive = TRUE)
t1_results <- analyze_single_timepoint(
  t1_data, "T1", file.path(t1_output_dir),
  communities, community_colors, communities_times, lognames
)

# 2. 分析T2网络
cat("=== Analyze T2 network ===\n")

t2_output_dir <- file.path(output_base_dir, "T2")
# if (!dir.exists(t1_output_dir)) dir.create(t2_output_dir, recursive = TRUE)
t2_results <- analyze_single_timepoint(
  t2_data, "T2", file.path(t2_output_dir),
  communities, community_colors, communities_times, lognames
)


# 3. 比较T1和T2网络
cat("=== Compare T1 and T2 networks ===\n")

source(here("INS_prediction", "Program", "Network_compare.R"))

compare_output_dir <- file.path(output_base_dir, "NCT_Comparison_T1_T2")
if (!dir.exists(compare_output_dir)) dir.create(compare_output_dir, recursive = TRUE)

nct_results <- compare_two_networks_nct(
  D1 = t1_data,
  D2 = t2_data,
  type_vec = type_vec,
  level_vec = level_vec,
  output_dir = compare_output_dir,
  n_iter = 1000
)

################ simplely compare  ##############################
# # 如果数据是数据框，转换为矩阵
# t1_mat <- as.matrix(t1_data)
# t2_mat <- as.matrix(t2_data)
# 
# # 检查列名
# if (is.null(colnames(t1_mat))) {
#   colnames(t1_mat) <- paste0("V", 1:ncol(t1_mat))
# }
# if (is.null(colnames(t2_mat))) {
#   colnames(t2_mat) <- paste0("V", 1:ncol(t2_mat))
# }
# 
# # 尝试使用最基本的参数
# cat("使用NetworkComparisonTest...\n")
# 
# nct_result <- NCT(
#   data1 = t1_mat,
#   data2 = t2_mat,
#   it = 10,  # 先用10测试
#   # binary.data = FALSE,
#   paired = TRUE,
#   # estimator = "mgm",
#   # estimatorArgs = list(
#   #   type = type_vec,
#   #   level = level_vec,
#   #   lambdaSel = "EBIC",
#   #   lambdaGam = 0.25
#   # ),
#   test.edges = TRUE,  # 先不测试边缘
#   test.centrality = TRUE,  # 先不测试中心性
#   # test.invariance = FALSE,  # 先不测试不变性
#   verbose = TRUE
# )
# 
# cat("NCT成功运行!\n")
# cat("全局不变性p值:", nct_result$glstrinv.pval, "\n")
# cat("网络不变性p值:", nct_result$nwinv.pval, "\n")
# 
# if (nct_result$glstrinv.pval < 0.05) {
#   cat("结论: 网络存在显著差异 (p =", nct_result$glstrinv.pval, ")\n")
# } else {
#   cat("结论: 网络无显著差异 (p =", nct_result$glstrinv.pval, ")\n")
# }


################## Cross-lagged Panel Network analysis ############################
# 4. T1->T2交叉滞后网络分析
cat("=== Cross-lagged Panel Network analysis of two waves ===\n")

source(here("INS_prediction", "Program", "Network_CLPN.R"))
clpn_output_dir <- file.path(output_base_dir, "CLPN")

network_results <- analyze_clpn_network(
  t1_data = t1_data,
  t2_data = t2_data,
  output_dir = clpn_output_dir,
  lognames = lognames,
  var_mapping = var_mapping,
  communities_times = communities_times,
  community_colors = community_colors,
  threshold_value = 0.05,
  iter = 1000
)




