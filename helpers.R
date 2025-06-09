# Function to install and load required packages
install_and_load_packages <- function() {
  required_packages <- c("paradox", "DoubleML", "readr", "haven", "dplyr", "mlr3", "mlr3learners", "mlr3tuning",
                         "rpart", "ranger", "glmnet", "xgboost", "e1071", "ggplot2", "collapse", 
                         "ggcorplot2", "caret", "tidyr", "corrplot", "Hmisc", 
                         "remotes", "IRdisplay", "data.table", "tidyverse", "rstudioapi")
  
  install_if_missing <- function(pkg) {
    if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
  }
  
  sapply(required_packages, install_if_missing)
  
  # Load necessary libraries
  library(paradox)
  library(rlang)       
  library(collapse)    
  library(tidyverse)
  library(ggplot2)   
  library(tidyr)       
  library(readr)       
  library(corrplot)    
  library(haven)       
  library(Hmisc)       
  library(caret)       
  library(rpart)       
  library(ranger)      
  library(glmnet)      
  library(xgboost)     
  library(e1071)       
  library(mlr3tuning)         
  library(mlr3)        
  library(mlr3learners)
  library(DoubleML)    
  library(data.table)  
  library(rstudioapi)  
  
# Install and load dlsr package
  remotes::install_github("ChihYuChiang/dlsr")
  library(dlsr)
}

#Function to preprocess data

preprocess_data <- function(df, outcome_var, treatment_var, features_X) {
  # Convert data into df format
  df <- as.data.frame(df)
  
  # Check if outcome_var and treatment_var are in the data frame
  if (!(outcome_var %in% names(df))) {
    stop(paste("Outcome variable", outcome_var, "is not in the data frame."))
  }
  if (!(treatment_var %in% names(df))) {
    stop(paste("Treatment variable", treatment_var, "is not in the data frame."))
  }
  
  # Check if features_X are in the data frame
  missing_features <- features_X[!features_X %in% names(df)]
  if (length(missing_features) > 0) {
    stop(paste("The following features are not in the data frame:", paste(missing_features, collapse = ", ")))
  }
  
  # Scale features
  df[, features_X] <- scale(df[, features_X], center = TRUE, scale = TRUE)
  
  # Convert labeled variables to numeric
  df <- df %>% mutate(across(where(~ inherits(., "haven_labelled")), ~ as.numeric(.)))
  
  # Dropping irrelevant variables
  relevant <- c(outcome_var, treatment_var, features_X)
  
  df <- df[, relevant]
  
  # Omit missing values
  df <- na.omit(df)
  
  # Check if the data frame is empty after omitting missing values
  if (nrow(df) == 0) {
    stop("The data frame is empty after omitting missing values.")
  }
  
  return(df)
}

#Function to perform correlation analysis and generate a correlation heat-map

correlation_analysis <- function(df, top_n=10){
  #Perform corr analysis
  corrdata <- df %>% na.omit()
  corr <- rcorr(as.matrix(corrdata), type="pearson")
  corrsig <- round(corr$P, 3)
  corrval <- round(corr$r, 3)
  
  #Generate heatmap
  heatmap <- recordPlot({corrplot(corrval, type = "upper", method="number", order = "hclust",
           p.mat=corrsig, sig.level=0.1, insig="blank")
  })
  
  #Identify top N pairs of correlated variables
  corr_long <- corrval %>%
    as.data.frame()%>%
    rownames_to_column(var = "Variable1") %>%
    pivot_longer(cols = -Variable1, 
                 names_to="Variable2", values_to = "Correlation") %>%
    filter(Variable1 != Variable2) %>%
    mutate(AbsCorrelation = abs(Correlation))%>%
    arrange(desc(AbsCorrelation))%>%
    head(top_n)
  
  top_corr_pairs <- corr_long %>%
    select(Variable1, Variable2, Correlation) %>%
    as.data.frame()
  
  return(list(top_corr_pairs=top_corr_pairs, heatmap=heatmap))
}

#Function to perform Double Lasso Selection
double_lasso_selection <- function(df, outcome_var, treatment_var, features_X){
  #Perform Double Lasso Selection
  DLS <- doubleLassoSelect(df, treatment_var, outcome_var, features_X, k=100)
  selected_features <- names(DLS[, -c(1,2)])
  return(selected_features)
}

# Function to run Double Machine Learning with hyperparameter tuning
run_dml <- function(df, outcome_var, treatment_var, features_X, model_type) {
  if (length(features_X)<2){
    stop("Please make sure you have two or more features selected.")
  }
  # --------- Construct DoubleML Data Object ----------
  obj_PLR <- double_ml_data_from_data_frame(
    df,
    y_col = outcome_var,
    d_cols = treatment_var,
    x_cols = features_X,
    use_other_treat_as_covariate = TRUE
  )
  
  # --------- Set up regression tasks -----------------
  task_regrY <- TaskRegr$new(id = "Y", backend = df, target = outcome_var)
  task_regrD <- TaskRegr$new(id = "D", backend = df, target = treatment_var)
  
  # --------- Choose learners and tuning grid ---------
  if (model_type == "enet") {
    base_learner <- lrn("regr.glmnet", predict_type = "response")
    param_set <- ps(
      alpha = p_dbl(lower = 0, upper = 1),
      lambda = p_dbl(lower=1e-4, upper=1)
    )
  } else if (model_type == "trees") {
    base_learner <- lrn("regr.rpart", predict_type = "response")
    param_set <- ps(
      cp = p_dbl(lower = 0.001, upper = 0.1),
      minsplit = p_int(lower = 2, upper = 50)
    )
  } else if (model_type == "forest") {
    base_learner <- lrn("regr.ranger", predict_type = "response")
    param_set <- ps(
      num.trees = p_int(lower = 5, upper = 50),
      min.node.size = p_int(lower = 1, upper = 20)
    )
  } else if (model_type == "boosted") {
    base_learner <- lrn("regr.xgboost", predict_type = "response")
    param_set <- ps(
      eta = p_dbl(lower = 0.01, upper = 0.3),
      max_depth = p_int(lower = 1, upper = 10)
    )
  } else {
    stop("Unknown model type")
  }
  
  # --------- Set up the autotuner for learners -------
  autotuner_Y <- AutoTuner$new(
    learner = base_learner$clone(deep = TRUE),
    resampling = rsmp("cv", folds = 5),
    measure = msr("regr.mse"),
    search_space = param_set,
    tuner = tnr("grid_search"),
    terminator = trm("evals", n_evals = 10)
  )
  
  autotuner_D <- AutoTuner$new(
    learner = base_learner$clone(deep = TRUE),
    resampling = rsmp("cv", folds = 5),
    measure = msr("regr.mse"),
    search_space = param_set,
    tuner = tnr("grid_search"),
    terminator = trm("evals", n_evals = 10)
  )
  
  autotuner_Y$train(task_regrY)
  autotuner_D$train(task_regrD)
  
  tuned_Y <- autotuner_Y$learner
  tuned_D <- autotuner_D$learner
  
  # --------- Run DoubleML ---------------------------
  PLR <- DoubleMLPLR$new(
    data = obj_PLR,
    ml_l = tuned_Y,
    ml_m = tuned_D,
    score = "partialling out",
    dml_procedure = "dml1",
    n_folds = 5,
    n_rep = 15
  )
  
  PLR$fit()
  return(PLR)
}
