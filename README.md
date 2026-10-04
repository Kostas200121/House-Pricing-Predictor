# House Price Prediction

## Project Overview

This project builds a machine learning regression model to predict **median house values** from housing and geographic characteristics.

The workflow explores the dataset, cleans missing values, visualizes feature relationships, engineers additional variables, compares multiple regression models, and tunes a Random Forest model using cross-validation.

In the recorded notebook run, the final tuned model achieved an **R² score of approximately 0.820** on the held-out test set.

## Model Results

| Model | Test R² |
| --- | ---: |
| Linear Regression | 0.6593 |
| Random Forest Regressor | 0.8140 |
| Tuned Random Forest | **0.8196** |

The tuned Random Forest produced the strongest test performance in the project.

## Dataset

The project uses:

```text
housing.csv
```

The original dataset contains:

```text
20,640 rows
10 columns
```

The available features are:

```text
longitude
latitude
housing_median_age
total_rooms
total_bedrooms
population
households
median_income
median_house_value
ocean_proximity
```

The prediction target is:

```text
median_house_value
```

### Missing Values

The dataset contains missing values in:

```text
total_bedrooms
```

There are 207 missing values in this column.

The current project handles them by removing rows containing missing values:

```python
data.dropna(inplace=True)
```

After cleaning, the dataset contains:

```text
20,433 rows
```

## Project Workflow

The notebook follows this general machine learning workflow:

```text
Load Dataset
      ↓
Inspect and Clean Data
      ↓
Train / Test Split
      ↓
Exploratory Data Analysis
      ↓
Feature Transformation
      ↓
Categorical Encoding
      ↓
Feature Engineering
      ↓
Feature Scaling
      ↓
Linear Regression Baseline
      ↓
Random Forest Regression
      ↓
GridSearchCV
      ↓
Final Test Evaluation
```

## Train and Test Split

The target variable is separated from the input features:

```python
X = data.drop(["median_house_value"], axis=1)
y = data["median_house_value"]
```

The dataset is then divided using:

```python
train_test_split(
    X,
    y,
    test_size=0.2
)
```

This produces approximately:

```text
Training rows: 16,346
Test rows: 4,087
```

## Exploratory Data Analysis

The notebook uses pandas, matplotlib, and seaborn to explore the data.

### Feature Distributions

Histograms are generated for the numerical variables:

```python
train_data.hist(figsize=(15, 8))
```

This helps visualize the distributions of variables such as:

```text
housing_median_age
total_rooms
total_bedrooms
population
households
median_income
median_house_value
```

### Correlation Analysis

A correlation heatmap is used to examine relationships between numerical features:

```python
sns.heatmap(
    numeric_data.corr(),
    annot=True,
    cmap="YlGnBu"
)
```

Additional heatmaps are generated after feature transformation and feature engineering to inspect how the new variables relate to house values.

### Geographic Visualization

The project also plots geographic coordinates and colors observations according to house value:

```python
sns.scatterplot(
    x="latitude",
    y="longitude",
    data=train_data,
    hue="median_house_value",
    palette="coolwarm"
)
```

This provides a visual representation of how housing values vary by location.

## Feature Transformation

Several numerical features have highly skewed distributions.

The notebook applies logarithmic transformations to:

```text
total_rooms
total_bedrooms
population
households
```

Example:

```python
train_data["total_rooms"] = np.log(
    train_data["total_rooms"] + 1
)
```

The same transformation is applied to the other selected variables.

## Encoding Ocean Proximity

The categorical feature:

```text
ocean_proximity
```

is converted into numerical indicator columns using one-hot encoding:

```python
pd.get_dummies(train_data.ocean_proximity)
```

The original categorical column is then removed.

The dataset contains categories such as:

```text
<1H OCEAN
INLAND
NEAR OCEAN
NEAR BAY
ISLAND
```

## Feature Engineering

Two additional features are created.

### Bedroom Ratio

```python
train_data["bedroom_ratio"] = (
    train_data["total_bedrooms"]
    / train_data["total_rooms"]
)
```

This represents the relationship between bedrooms and total rooms.

### Household Rooms

```python
train_data["household_rooms"] = (
    train_data["total_rooms"]
    / train_data["households"]
)
```

This provides an additional measure of housing density.

## Feature Scaling

The model input features are standardized using:

```python
StandardScaler()
```

Example:

```python
scaler = StandardScaler()

X_train_s = scaler.fit_transform(X_train)
```

The fitted scaler is then used to transform the test features:

```python
X_test_s = scaler.transform(X_test)
```

## Model 1: Linear Regression

The first model provides a regression baseline:

```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train_s, y_train)
```

The recorded test score was:

```text
R² = 0.6593055407
```

or approximately:

```text
65.93%
```

This establishes a useful baseline before moving to a more flexible nonlinear model.

## Model 2: Random Forest Regressor

The project then trains a Random Forest regression model:

```python
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()
forest.fit(X_train_s, y_train)
```

The recorded test result improved significantly:

```text
R² = 0.8139532638
```

or approximately:

```text
81.40%
```

This showed that the nonlinear ensemble model captured the relationships in the data considerably better than the Linear Regression baseline.

## Hyperparameter Tuning

The Random Forest model is further optimized using:

```python
GridSearchCV
```

The parameter grid tests:

```python
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_features": [None, 4, 8],
    "min_samples_split": [2, 4]
}
```

The search uses:

```text
5-fold cross-validation
Negative Mean Squared Error scoring
```

The best estimator from the recorded run was:

```python
RandomForestRegressor(
    max_features=8,
    n_estimators=300
)
```

## Final Model Performance

The tuned model achieved:

```text
R² = 0.8195617939
```

or approximately:

```text
81.96%
```

on the held-out test set.

This was the best result among the models evaluated in the notebook.

## Test Data Preparation

The same feature transformations used for the training set are applied to the test set:

```text
Log transformation
One-hot encoding
Bedroom ratio
Household rooms
Column alignment
Standard scaling
```

The encoded test data is reindexed to ensure that its columns match the training dataset:

```python
test_data = test_data.reindex(
    columns=train_data.columns,
    fill_value=0
)
```

This avoids mismatches when categorical values produce different dummy columns between the two subsets.

## Technologies Used

```text
Python
pandas
NumPy
Matplotlib
Seaborn
scikit-learn
Jupyter Notebook
```

## Machine Learning Concepts Demonstrated

This project demonstrates practical experience with:

1. Regression problems
2. Data cleaning
3. Missing-value handling
4. Train and test splitting
5. Exploratory data analysis
6. Distribution visualization
7. Correlation analysis
8. Log transformations
9. One-hot encoding
10. Feature engineering
11. Feature scaling
12. Linear Regression
13. Random Forest Regression
14. Hyperparameter tuning
15. Grid search
16. Cross-validation
17. Model comparison
18. R² evaluation

## Project Structure

```text
house-price-prediction/
│
├── housing.ipynb
├── housing.csv
└── README.md
```

## How to Run

Install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Place the files in the same project directory:

```text
housing.ipynb
housing.csv
```

If necessary, update the dataset path in the notebook to:

```python
data = pd.read_csv("housing.csv")
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
housing.ipynb
```

and run the cells in order.

## Possible Improvements

Future improvements could include:

1. Set a fixed `random_state` for reproducible train/test splits and Random Forest results.
2. Replace row deletion with an imputation strategy for missing `total_bedrooms` values.
3. Use a single scikit-learn preprocessing pipeline for transformations, encoding, feature engineering, and modeling.
4. Add RMSE and MAE alongside R² for easier interpretation of prediction error.
5. Compare additional regression algorithms such as Gradient Boosting, XGBoost, or HistGradientBoosting.
6. Investigate feature importance from the Random Forest.
7. Perform residual analysis to understand where the model performs poorly.
8. Use cross-validation for direct model comparison rather than relying only on one train/test split.
9. Tune additional Random Forest parameters such as `max_depth`, `min_samples_leaf`, and `bootstrap`.
10. Avoid unnecessary scaling for tree-based models and keep scaling only where the selected estimator benefits from it.

## Conclusion

This project demonstrates an end-to-end house price regression workflow, progressing from exploratory data analysis and a Linear Regression baseline to feature engineering, Random Forest modeling, and hyperparameter optimization.

The progression of test results was:

```text
Linear Regression          0.6593 R²
Random Forest              0.8140 R²
Tuned Random Forest        0.8196 R²
```

The final tuned Random Forest achieved the best recorded performance with an **R² score of approximately 0.820**.
