import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('train.csv')

# Separate target and features, drop ID column
data = data.drop(columns=['Id'])
X = data.drop(columns=['SalePrice'])
y = data['SalePrice']

# Define numerical and categorical columns
num_cols = X.select_dtypes(exclude='object').columns
cat_cols = X.select_dtypes(include='object').columns

# Define preprocessing for numeric and categorical features
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessors
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply log transformation to target
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# Preprocess features
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Define models
linear_model = LinearRegression()
ridge_model = Ridge(alpha=1.0)
stacking_model = StackingRegressor(
    estimators=[('lr', linear_model), ('ridge', ridge_model)],
    final_estimator=DecisionTreeRegressor(max_depth=5)
)

# Dictionary to store results
results = {}

# Train and evaluate Linear Regression
linear_model.fit(X_train_processed, y_train_log)
y_pred_linear_log = linear_model.predict(X_test_processed)
results['Linear Regression'] = {
    'MSE': mean_squared_error(y_test_log, y_pred_linear_log),
    'R2': r2_score(y_test_log, y_pred_linear_log)
}

# Train and evaluate Ridge Regression
ridge_model.fit(X_train_processed, y_train_log)
y_pred_ridge_log = ridge_model.predict(X_test_processed)
results['Ridge Regression'] = {
    'MSE': mean_squared_error(y_test_log, y_pred_ridge_log),
    'R2': r2_score(y_test_log, y_pred_ridge_log)
}

# Train and evaluate Stacking Model
stacking_model.fit(X_train_processed, y_train_log)
y_pred_stacking_log = stacking_model.predict(X_test_processed)
results['Stacking Model'] = {
    'MSE': mean_squared_error(y_test_log, y_pred_stacking_log),
    'R2': r2_score(y_test_log, y_pred_stacking_log)
}

# Display results
results_df = pd.DataFrame(results).T
print(results_df)
