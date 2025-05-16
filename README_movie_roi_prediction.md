# movie-roi-prediction.R

This project uses R to model and predict the return on investment (ROI) of movies based on budget, revenue, cast size, popularity, and other metadata. 
After extensive data cleaning and feature engineering, multiple machine learning models—including Random Forest, XGBoost, and Multiple Linear Regression—are trained and evaluated.

## Highlights

- Created a custom ROI metric (`profit / budget`) and applied `log(ROI)` transformation for modeling.
- Engineered features such as cast size, number of production companies, and tagline presence.
- Trained and tuned Random Forest and XGBoost models to predict log(ROI).
- Compared model performance using RMSE, MAE, and R² metrics.
- Visualized predictions and residuals across models and plotted feature importance.
- Used a clean, dark-themed visualization layout to improve clarity and focus.

## Tools Used

- R (tidyverse, caret, randomForest, xgboost, ggplot2)
- Data cleaning, feature engineering, predictive modeling
- Evaluation metrics: RMSE, MAE, R²
- Visualizations: Bar charts, prediction vs. actual plots, residual analysis, feature importance

## Dataset

- `movie_dataset.csv`: Includes metadata about films (budget, revenue, popularity, cast, etc.) sourced from [Utkarsh Singh's dataset on Kaggle](https://www.kaggle.com/datasets/utkarshx27/movies-dataset)
- ROI and profit were calculated within the project using available fields.
  
## Performance Summary (log(ROI))

| Model                  | RMSE     | MAE      | R²       |
|------------------------|----------|----------|----------|
| Random Forest          | 5.2642893  | 0.5564059  | 0.4198017  |
| XGBoost (Default)      | 0.7540943  | 0.5663611  | 0.3990475  |
| XGBoost (Tuned)        | 0.7391169*  | 0.5638525  | 0.4226821  |
| Multiple Linear Reg.   | 0.9093914  | 0.6729439  | 0.1260421  |

## Feature Importance Table

| Feature                  | Gain     | Cover    | Frequency |
|--------------------------|----------|----------|------------|
| `budget`                 | 0.5623   | 0.3755   | 0.3143     |
| `popularity`             | 0.2479   | 0.3243   | 0.2995     |
| `runtime`                | 0.1101   | 0.2150   | 0.2477     |
| `num_production_companies` | 0.0726 | 0.0794   | 0.1229     |
| `has_tagline`            | 0.0070   | 0.0058   | 0.0139     |
| `num_cast`               | 0.0001   | 0.0000   | 0.0017     |




## Goal

To identify which production-related features drive movie ROI and explore whether machine learning methods outperform linear models in predicting financial success.

## Author
**Rebecca Calhoun**
M.S. in Data Analytics (July 2025)  
 [LinkedIn](https://www.linkedin.com/in/rebecca-calhoun9/) | 

