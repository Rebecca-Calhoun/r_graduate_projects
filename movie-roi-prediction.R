#movie-roi-prediction
df <- read.csv("/cloud/project/movie_dataset.csv", stringsAsFactors = FALSE)
#Cleaning
df$revenue <- as.numeric(df$revenue)
df$budget <- as.numeric(df$budget)
df$profit <- df$revenue - df$budget
df$roi <- NULL
df$budget <- as.numeric(df$budget)
df$profit <- as.numeric(df$profit)
df$roi <- ifelse(df$budget > 0, df$profit / df$budget, NA)
df$log_roi <- ifelse(df$roi > 0, log1p(df$roi), NA)
summary(df$roi)
hist(df$roi, breaks = 50, main = "ROI Distribution", xlab = "ROI (Profit / Budget)")
df[sample(nrow(df), 10), c("title", "budget", "revenue", "profit", "roi")]
library(dplyr)
model_data <- df %>%
  filter(!is.na(log_roi), !is.na(budget), !is.na(popularity), !is.na(runtime)) %>%
  mutate(
    has_tagline = ifelse(tagline == "" | is.na(tagline), 0, 1),
    num_cast = stringr::str_count(cast, ",") + 1,
    num_production_companies = stringr::str_count(production_companies, ",") + 1
  ) %>%
  dplyr::select(log_roi, budget, popularity, runtime, num_cast, num_production_companies,
         has_tagline)
#random forest
library(caret)
set.seed(123)
split <- createDataPartition(model_data$log_roi, p = 0.8, list = FALSE)
train <- model_data[split, ]
test <- model_data[-split, ]
library(randomForest)
set.seed(123)
rf_model <- randomForest(log_roi ~ ., data = train, ntree = 500, importance = TRUE)
predictions <- predict(rf_model, newdata = test)
rmse <- sqrt(mean((predictions - test$log_roi)^2))
cat("Random Forest RMSE on log(ROI):", rmse, "\n")
varImpPlot(rf_model)
importance(rf_model)
# Predict
predictions <- predict(rf_model, newdata = test)
rmse <- sqrt(mean((predictions - test$log_roi)^2))
cat("Final Random Forest RMSE (log ROI):", rmse, "\n")


#trying xg boost
library(xgboost)
# Convert to matrix
x_train <- as.matrix(train[, -1])
y_train <- train$log_roi
x_test <- as.matrix(test[, -1])
y_test <- test$log_roi
#train xg boost
set.seed(123)
xgb_model <- xgboost(
  data = x_train,
  label = y_train,
  nrounds = 100,
  objective = "reg:squarederror",
  max_depth = 6,
  eta = 0.1,
  verbose = 0
)
#test xgboost
xgb_predictions <- predict(xgb_model, x_test)
# RMSE on log(ROI)
xgb_rmse <- sqrt(mean((xgb_predictions - y_test)^2))
cat("XGBoost RMSE on log(ROI):", xgb_rmse, "\n")
importance_matrix <- xgb.importance(model = xgb_model)
xgb.plot.importance(importance_matrix)
#tuning xgboost
set.seed(123)
xgb_model_tuned <- xgboost(
  data = x_train,
  label = y_train,
  nrounds = 300,
  max_depth = 4,
  eta = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.8,
  objective = "reg:squarederror",
  verbose = 0
)
xgb_predictions_tuned <- predict(xgb_model_tuned, x_test)
xgb_rmse_tuned <- sqrt(mean((xgb_predictions_tuned - y_test)^2))
cat("Tuned XGBoost RMSE on log(ROI):", xgb_rmse_tuned, "\n")
#MAE RMSE and R2
y_true <- test$log_roi
# Random Forest
rf_mae <- mean(abs(rf_pred_log - y_true))
rf_r2 <- 1 - sum((rf_pred_log - y_true)^2) / sum((y_true - mean(y_true))^2)
# XGBoost Default
xgb_mae <- mean(abs(xgb_predictions - y_true))
xgb_r2 <- 1 - sum((xgb_predictions - y_true)^2) / sum((y_true - mean(y_true))^2)
# XGBoost Tuned
xgb_tuned_mae <- mean(abs(xgb_predictions_tuned - y_true))
xgb_tuned_r2 <- 1 - sum((xgb_predictions_tuned - y_true)^2) / sum((y_true -
                                                                     mean(y_true))^2)
# Print
cat("Model Performance on log(ROI):\n")
cat("Random Forest - MAE:", round(rf_mae, 4), " R²:", round(rf_r2, 4), "\n")
cat("XGBoost (Default) - MAE:", round(xgb_mae, 4), " R²:", round(xgb_r2, 4), "\n")
cat("XGBoost (Tuned) - MAE:", round(xgb_tuned_mae, 4), " R²:", round(xgb_tuned_r2, 4),
    "\n")
model_metrics <- data.frame(
  Model = c("Random Forest", "XGBoost (Default)", "XGBoost (Tuned)"),
  RMSE = c(rf_rmse, xgb_rmse, xgb_rmse_tuned),
  MAE = c(rf_mae, xgb_mae, xgb_tuned_mae),
  R2 = c(rf_r2, xgb_r2, xgb_tuned_r2)
)
print(model_metrics)
#multiple linear regression
mlr_model <- lm(log_roi ~ ., data = train)
summary(mlr_model)
# Predict on test
mlr_predictions <- predict(mlr_model, newdata = test)
# RMSE, MAE, R2
mlr_rmse <- sqrt(mean((mlr_predictions - test$log_roi)^2))
mlr_mae <- mean(abs(mlr_predictions - test$log_roi))
mlr_r2 <- 1 - sum((mlr_predictions - test$log_roi)^2) / sum((test$log_roi -
                                                               mean(test$log_roi))^2)
# Print results
cat("Multiple Linear Regression:\n")
cat("RMSE:", round(mlr_rmse, 4), "\n")
cat("MAE:", round(mlr_mae, 4), "\n")
cat("R²:", round(mlr_r2, 4), "\n")
model_metrics <- rbind(
  model_metrics,
  data.frame(
    Model = "Multiple Linear Regression",
    RMSE = mlr_rmse,
    MAE = mlr_mae,
    R2 = mlr_r2
  )
)
print(model_metrics)
mlr_model <- lm(log_roi ~ ., data = train)
mlr_predictions <- predict(mlr_model, newdata = test)
par(bg = "black", col.axis = "white", col.lab = "white", col.main = "white", fg = "white")
#model performance bar chart
library(ggplot2)
library(tidyr)
model_metrics_long <- model_metrics %>%
  pivot_longer(cols = c(RMSE, MAE, R2), names_to = "Metric", values_to = "Value")
# Custom colors
metric_colors <- c("RMSE" = "firebrick",
                   "MAE" = "darkorange",
                   "R2" = "darkslategray")
# bar chart
ggplot(model_metrics_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual(values = metric_colors) +
  labs(title = "Model Performance Comparison",
       y = "Metric Value", x = "Model") +
  theme_minimal(base_size = 12) +
  theme(
    plot.background = element_rect(fill = "black", color = NA),
    panel.background = element_rect(fill = "black", color = NA),
    panel.grid.major = element_line(color = "gray30"),
    panel.grid.minor = element_blank(),
    axis.text = element_text(color = "white"),
    axis.title = element_text(color = "white"),
    plot.title = element_text(color = "white", face = "bold"),
    legend.background = element_rect(fill = "black"),
    legend.text = element_text(color = "white"),
    legend.title = element_text(color = "white"),
    axis.text.x = element_text(angle = 20, hjust = 1)
  )
#log scale
# 3 plots
par(mfrow = c(1, 3),
    bg = "black", col.axis = "white", col.lab = "white", col.main = "white", fg = "white")
# MLR
plot(test$log_roi, mlr_predictions,
     main = "MLR",
     xlab = "Actual log(ROI)", ylab = "Predicted log(ROI)",
     col = "darkorange", pch = 19)
abline(0, 1, col = "gray", lwd = 2)
# Random Forest
plot(test$log_roi, rf_pred_log,
     main = "Random Forest",
     xlab = "Actual log(ROI)", ylab = "Predicted log(ROI)",
     col = "darkslategray", pch = 19)
abline(0, 1, col = "gray", lwd = 2)
# XGBoost (Tuned)
plot(test$log_roi, xgb_predictions_tuned,
     main = "XGBoost ",
     xlab = "Actual log(ROI)", ylab = "Predicted log(ROI)",
     col = "firebrick", pch = 19)
abline(0, 1, col = "gray", lwd = 2)
# Reset layout
par(mfrow = c(1, 1))
# Convert predicted log(ROI) back to ROI
mlr_roi <- exp(mlr_predictions) - 1
rf_roi <- exp(rf_pred_log) - 1
xgb_tuned_roi <- exp(xgb_predictions_tuned) - 1
# have the actual ROI
actual_roi <- exp(test$log_roi) - 1
#ROI scale
# Set up 3 side-by-side plots with black background
par(mfrow = c(1, 3),
    bg = "black", col.axis = "white", col.lab = "white", col.main = "white", fg = "white")
# MLR
plot(actual_roi, mlr_roi,
     main = "MLR",
     xlab = "Actual ROI", ylab = "Predicted ROI",
     col = "darkorange", pch = 19, cex = 0.6,
     xlim = c(0, quantile(actual_roi, 0.95)),
     ylim = c(0, quantile(mlr_roi, 0.95)))
abline(0, 1, col = "gray", lwd = 2)
# Random Forest
plot(actual_roi, rf_roi,
     main = "Random Forest",
     xlab = "Actual ROI", ylab = "Predicted ROI",
     col = "darkslategray", pch = 19, cex = 0.6,
     xlim = c(0, quantile(actual_roi, 0.95)),
     ylim = c(0, quantile(rf_roi, 0.95)))
abline(0, 1, col = "gray", lwd = 2)
# XGBoost (Tuned)
plot(actual_roi, xgb_tuned_roi,
     main = "XGBoost",
     xlab = "Actual ROI", ylab = "Predicted ROI",
     col = "firebrick", pch = 19, cex = 0.6,
     xlim = c(0, quantile(actual_roi, 0.95)),
     ylim = c(0, quantile(xgb_tuned_roi, 0.95)))
abline(0, 1, col = "gray", lwd = 2)
# Reset layout
par(mfrow = c(1, 1))
# Residuals on ROI scale
mlr_resid <- mlr_roi - actual_roi
rf_resid <- rf_roi - actual_roi
xgb_resid <- xgb_tuned_roi - actual_roi
#residuals
# Set up side-by-side layout
par(mfrow = c(1, 3),
    bg = "black", col.axis = "white", col.lab = "white", col.main = "white", fg = "white")
# MLR Residuals
plot(actual_roi, mlr_resid,
     main = "MLR",
     xlab = "Actual ROI", ylab = "Residual",
     col = "darkorange", pch = 19,
     xlim = c(0, quantile(actual_roi, 0.95)))
abline(h = 0, col = "gray", lwd = 2)
# Random Forest Residuals
plot(actual_roi, rf_resid,
     main = "Random Forest",
     xlab = "Actual ROI", ylab = "Residual",
     col = "darkslategray", pch = 19,
     xlim = c(0, quantile(actual_roi, 0.95)))
abline(h = 0, col = "gray", lwd = 2)
# XGBoost Tuned Residuals
plot(actual_roi, xgb_resid,
     main = "XGBoost Tuned",
     xlab = "Actual ROI", ylab = "Residual",
     col = "firebrick", pch = 19,
     xlim = c(0, quantile(actual_roi, 0.95)))
abline(h = 0, col = "gray", lwd = 2)
par(mfrow = c(1, 1))
library(xgboost)
importance_matrix <- xgb.importance(model = xgb_model_tuned)
print(importance_matrix)
xgb.plot.importance(importance_matrix,
                    top_n = 10,
                    rel_to_first = TRUE,
                    col = "firebrick",
                    main = "XGBoost Feature Importance")