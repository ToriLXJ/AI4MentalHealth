################## Data prepare ############################
prepare_data <- function(input_dir, main_vars,
                         con_vars = c("gender_T1"),
                         time_points = c("T1", "T2"),
                         var_mapping,
                         na_handling = "median") {
  
  cat("Preparing data...\n")
  
  # Read data
  data_raw <- read.csv(input_dir, encoding = "UTF-8")
  cat("Raw data dimensions:", dim(data_raw), "\n")
  
  # Select variables
  selected_vars <- select_variables(data_raw, main_vars, con_vars, time_points)
  cat("Selected variables dimensions:", dim(selected_vars), "\n")
  
  # Preprocess data
  processed_data <- preprocess_data(selected_vars, na_handling)
  cat("Processed data dimensions:", dim(processed_data), "\n")
  
  # Extract and rename variables
  result <- extract_and_rename(processed_data, main_vars, con_vars, time_points, var_mapping)
  
  cat("Data preparation completed\n")
  return(result)
}

# 简化变量选择
select_variables <- function(data, main_vars, con_vars, time_points) {
  # Build variable patterns
  var_patterns <- c()
  
  # Add main variables
  for (var in main_vars) {
    for (time in time_points) {
      pattern <- paste0("^", var, "_", time, "_\\d+$")
      var_patterns <- c(var_patterns, pattern)
    }
  }
  
  # Add control variables
  for (con_var in con_vars) {
    pattern <- paste0("^", con_var, "$")
    var_patterns <- c(var_patterns, pattern)
  }
  
  # Select matching columns
  selected_cols <- c()
  for (pattern in var_patterns) {
    matched_cols <- grep(pattern, names(data), value = TRUE, perl = TRUE)
    selected_cols <- c(selected_cols, matched_cols)
  }
  
  # Remove duplicates and keep existing columns
  selected_cols <- unique(selected_cols)
  existing_cols <- selected_cols[selected_cols %in% names(data)]
  
  cat("Selected", length(existing_cols), "variables\n")
  return(data[, existing_cols, drop = FALSE])
}

# 简化数据预处理
preprocess_data <- function(data, na_handling = "median") {
  # Convert to numeric
  df_numeric <- data %>%
    mutate(across(everything(), ~ as.numeric(as.character(.x))))
  
  # Handle missing values
  if (na_handling == "median") {
    df_complete <- df_numeric %>%
      mutate(across(everything(), ~ {
        if (is.numeric(.x)) {
          ifelse(is.na(.x), median(.x, na.rm = TRUE), .x)
        } else {
          .x
        }
      }))
  } else if (na_handling == "omit") {
    df_complete <- df_numeric %>% na.omit()
  }
  
  return(df_complete)
}

# 简化变量提取和重命名
extract_and_rename <- function(data, main_vars, con_vars, time_points, var_mapping) {
  result <- list()
  
  # Process each time point
  for (time_point in time_points) {
    # Extract variables for this time point
    time_data <- extract_vars_by_time(data, main_vars, time_point)
    
    # Rename variables
    renamed_data <- rename_variables(time_data, main_vars, var_mapping, time_point)
    
    # Store result
    result_name <- paste0("complete_", tolower(time_point))
    result[[result_name]] <- renamed_data
    
    cat("Dataset", result_name, "dimensions:", dim(renamed_data), "\n")
  }
  
  return(result)
}

# 简化时间点变量提取
extract_vars_by_time <- function(data, main_vars, time_suffix) {
  # Build patterns
  patterns <- sapply(main_vars, function(var) {
    paste0("^", var, "_", time_suffix, "_\\d+$")
  })
  
  # Select matching columns
  selected_cols <- c()
  for (pattern in patterns) {
    matched <- grep(pattern, names(data), value = TRUE, perl = TRUE)
    selected_cols <- c(selected_cols, matched)
  }
  
  result <- data[, selected_cols, drop = FALSE]
  return(result)
}

# 简化变量重命名
rename_variables <- function(data, main_vars, var_mapping, time_point) {
  new_names <- names(data)
  
  # Apply mapping rules
  for (var in main_vars) {
    if (var %in% names(var_mapping)) {
      prefix <- var_mapping[[var]]
      pattern <- paste0("^", var, "_.*_(\\d+)$")
      replacement <- paste0(prefix, "\\1")
      new_names <- gsub(pattern, replacement, new_names)
    }
  }
  
  names(data) <- new_names
  return(data)
}