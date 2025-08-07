import pandas as pd
import numpy as np
import os
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
from catboost import CatBoostRegressor
from statsmodels.formula.api import ols
import scipy.stats as stats

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore
!pip install scikit-optimize
!pip install lightgbm
!pip install xgboost
!pip install catboost

cwd = os.getcwd()
print(cwd)
os.chdir("C:/Users/sanja/downloads")
train= pd.read_csv("C:/Users/sanja/downloads/train.csv")
train.loc[train["Item_Fat_Content"] == 'LF', 'Item_Fat_Content'] = 'Low Fat'
train.loc[train["Item_Fat_Content"] == 'reg', 'Item_Fat_Content'] = 'Regular'
train.loc[train["Item_Fat_Content"] == 'low fat', 'Item_Fat_Content'] = 'Low Fat'
null_counts = train.isnull().sum()
print(null_counts)
train.loc[
    (train['Outlet_Type'] == 'Grocery Store') & (train['Outlet_Size'].isnull()),
    'Outlet_Size'
] = 'Small'

train.loc[
    (train['Outlet_Type'] == 'Supermarket Type1') &
    (train['Outlet_Location_Type'] == 'Tier 2') &
    (train['Outlet_Size'].isnull()),
    'Outlet_Size'
] = 'Small'

null_counts = train.isnull().sum()
print(null_counts)




# Load your train dataset
df = train

# Categorical columns in your dataset
categorical_cols = [
    'Item_Fat_Content',
    'Item_Type',
    'Outlet_Establishment_Year',
    'Outlet_Size',
    'Outlet_Location_Type',
    'Outlet_Type'
]

# Function to calculate eta squared
def eta_squared(anova_table):
    return anova_table['sum_sq'][0] / anova_table['sum_sq'].sum()

# Loop through each categorical column and compute eta squared
for col in categorical_cols:
    print(f"\n {col} vs Item_Weight")
    
    # Drop rows with missing values in relevant columns
    df_clean = df[[col, 'Item_Weight']].dropna()
    
    # Run ANOVA
    model = ols(f'Item_Weight ~ C({col})', data=df_clean).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    # Compute and print Eta Squared
    eta_sq = eta_squared(anova_table)
    print(f"Eta Squared: {eta_sq:.4f}")





# Load data
df = train

# Define columns
categorical_cols = [
    'Item_Fat_Content', 'Item_Type', 'Outlet_Establishment_Year',
    'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'
]
continuous_cols = [
    'Item_Weight', 'Item_Visibility', 'Item_MRP', 'Item_Outlet_Sales'
]

# Clean up label mismatches (example)
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
    'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'
})

# Helper: Eta Squared for Categorical vs Continuous
def eta_squared(col, target):
    model = ols(f'{target} ~ C({col})', data=df.dropna(subset=[col, target])).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table['sum_sq'][0] / anova_table['sum_sq'].sum()

# Helper: Cramér’s V for Categorical vs Categorical
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k-1, r-1))

# Initialize empty matrix
all_cols = continuous_cols + categorical_cols
corr_matrix = pd.DataFrame(index=all_cols, columns=all_cols)

# Fill correlation matrix
for col1 in all_cols:
    for col2 in all_cols:
        if col1 == col2:
            corr_matrix.loc[col1, col2] = 1.0
        elif col1 in continuous_cols and col2 in continuous_cols:
            corr = df[[col1, col2]].corr().iloc[0, 1]
            corr_matrix.loc[col1, col2] = corr
        elif col1 in categorical_cols and col2 in continuous_cols:
            eta = eta_squared(col1, col2)
            corr_matrix.loc[col1, col2] = eta
        elif col1 in continuous_cols and col2 in categorical_cols:
            eta = eta_squared(col2, col1)
            corr_matrix.loc[col1, col2] = eta
        elif col1 in categorical_cols and col2 in categorical_cols:
            cramers = cramers_v(df[col1], df[col2])
            corr_matrix.loc[col1, col2] = cramers

# Convert to numeric matrix
corr_matrix = corr_matrix.astype(float)

# Heatmap visualization
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Correlation Matrix (Pearson, Eta Squared, Cramér's V)")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Flatten the correlation matrix and remove self-pairs and duplicates
corr_pairs = (
    corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    .stack()
    .reset_index()
)
corr_pairs.columns = ['Variable_1', 'Variable_2', 'Correlation']

# Sort by absolute correlation value
corr_pairs['Abs_Correlation'] = corr_pairs['Correlation'].abs()
top_corr = corr_pairs.sort_values(by='Abs_Correlation', ascending=False)

top_corr=pd.DataFrame(top_corr)

# Show top 10 most correlated variable pairs
print(top_corr.head(15))





# Load data
df = train

# Continuous columns
cont_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Item_Outlet_Sales']

# IQR-based outlier detection
print(" IQR-based Outlier Summary:")
for col in cont_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers")

# Z-score-based outlier detection
print("\n Z-score-based Outlier Summary:")
z_scores = df[cont_cols].apply(zscore)
for col in cont_cols:
    outliers = df[(z_scores[col].abs() > 3)]
    print(f"{col}: {len(outliers)} outliers")


plt.figure(figsize=(12, 8))
for i, col in enumerate(cont_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(data=df, y=col)
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()


# List of categorical columns to use for grouping
group_cols = [
    'Item_Fat_Content',
    'Item_Type',
    'Outlet_Establishment_Year',
    'Outlet_Size',
    'Outlet_Location_Type',
    'Outlet_Type'
]

# Create group-based mean weight mapping
group_mean = df.groupby(group_cols)['Item_Weight'].transform('mean')

# Fill missing Item_Weight with the grouped mean
df['Item_Weight'] = df['Item_Weight'].fillna(group_mean)

# Optional: If still any nulls remain (i.e., group not found), fill with global mean
df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())

# Confirm missing values filled
print("Remaining nulls in Item_Weight:", df['Item_Weight'].isna().sum())

null_counts = train.isnull().sum()
print(null_counts)


# Load data
df = train

# Continuous columns
cont_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Item_Outlet_Sales']

# Function to remove IQR-based outliers
def remove_outliers_iqr(data, columns):
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        original_shape = data.shape[0]
        data = data[(data[col] >= lower) & (data[col] <= upper)]
        removed = original_shape - data.shape[0]
        print(f"Removed {removed} outliers from '{col}'")
    return data

# Apply function
df_cleaned = remove_outliers_iqr(df, cont_cols)

# Result
print(f"\n Final dataset shape after outlier removal: {df_cleaned.shape}")

train_cleaned=df_cleaned

test= pd.read_csv("C:/Users/sanja/downloads/test.csv")
test.loc[test["Item_Fat_Content"] == 'LF', 'Item_Fat_Content'] = 'Low Fat'
test.loc[test["Item_Fat_Content"] == 'reg', 'Item_Fat_Content'] = 'Regular'
test.loc[test["Item_Fat_Content"] == 'low fat', 'Item_Fat_Content'] = 'Low Fat'
null_counts = test.isnull().sum()
print(null_counts)

test.loc[
    (test['Outlet_Type'] == 'Grocery Store') & (test['Outlet_Size'].isnull()),
    'Outlet_Size'
] = 'Small'

test.loc[
    (test['Outlet_Type'] == 'Supermarket Type1') &
    (test['Outlet_Location_Type'] == 'Tier 2') &
    (test['Outlet_Size'].isnull()),
    'Outlet_Size'
] = 'Small'


import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Load data
df_test = test

# Continuous columns
cont_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP']

# IQR-based outlier detection
print(" IQR-based Outlier Summary:")
for col in cont_cols:
    Q1 = df_test[col].quantile(0.25)
    Q3 = df_test[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df_test[(df_test[col] < lower) | (df_test[col] > upper)]
    print(f"{col}: {len(outliers)} outliers")

# Z-score-based outlier detection
print("\n Z-score-based Outlier Summary:")
z_scores = df_test[cont_cols].apply(zscore)
for col in cont_cols:
    outliers = df_test[(z_scores[col].abs() > 3)]
    print(f"{col}: {len(outliers)} outliers")


# List of categorical columns to use for grouping
group_cols = [
    'Item_Fat_Content',
    'Item_Type',
    'Outlet_Establishment_Year',
    'Outlet_Size',
    'Outlet_Location_Type',
    'Outlet_Type'
]

# Create group-based mean weight mapping
group_mean = df_test.groupby(group_cols)['Item_Weight'].transform('mean')

# Fill missing Item_Weight with the grouped mean
df_test['Item_Weight'] = df_test['Item_Weight'].fillna(group_mean)

# Optional: If still any nulls remain (i.e., group not found), fill with global mean
df_test['Item_Weight'] = df_test['Item_Weight'].fillna(df_test['Item_Weight'].mean())

# Confirm missing values filled
print("Remaining nulls in Item_Weight:", df_test['Item_Weight'].isna().sum())

test_cleaned=df_test


### Checking evaluation metrices in Stacked modelling algorithm of LGBM and XGBoost

# ------------------ Data Preparation ------------------

# Add interaction features
for df in [train_cleaned, test_cleaned]:
    df['MRP_Visibility'] = df['Item_MRP'] * df['Item_Visibility']
    df['Type_Size_Interaction'] = df['Item_Type'] + '_' + df['Outlet_Size'].astype(str)

# Training data prep
df = train_cleaned.copy() 
df['Years_Since_Est'] = 2025 - df['Outlet_Establishment_Year']
df = df.dropna()

X = df.drop(columns=['Item_Outlet_Sales'])
y = np.log1p(df['Item_Outlet_Sales'])

# Encode categorical features
cat_cols = X.select_dtypes(include='object').columns.tolist()
original_cats = X[cat_cols].copy()
encoder = OrdinalEncoder()
X[cat_cols] = encoder.fit_transform(X[cat_cols])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------ Hyperparameter Tuning ------------------

search_lgbm = BayesSearchCV(
    LGBMRegressor(random_state=42),
    search_spaces={
        'n_estimators': Integer(100, 500),
        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'max_depth': Integer(3, 10),
        'subsample': Real(0.5, 1.0),
        'colsample_bytree': Real(0.5, 1.0)
    },
    n_iter=50,
    scoring='r2',
    cv=5,
    verbose=0,
    n_jobs=-1,
    random_state=42
)
search_lgbm.fit(X_train, y_train)

search_xgb = BayesSearchCV(
    XGBRegressor(random_state=42, verbosity=0),
    search_spaces={
        'n_estimators': Integer(100, 500),
        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'max_depth': Integer(3, 10),
        'subsample': Real(0.5, 1.0),
        'colsample_bytree': Real(0.5, 1.0)
    },
    n_iter=50,
    scoring='r2',
    cv=5,
    verbose=0,
    n_jobs=-1,
    random_state=42
)
search_xgb.fit(X_train, y_train)

lgbm_best = search_lgbm.best_estimator_
xgb_best = search_xgb.best_estimator_

# ------------------ Stacking Model ------------------

stacked_model = StackingRegressor(
    estimators=[
        ('lgbm', lgbm_best),
        ('xgb', xgb_best)
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    n_jobs=-1
)
stacked_model.fit(X_train, y_train)

# ------------------ Evaluation ------------------

y_pred_log = stacked_model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test)

r2 = r2_score(y_test_actual, y_pred)
rmse = mean_squared_error(y_test_actual, y_pred, squared=False)

print(f"Final R² Score (Blended): {r2:.3f}")
print(f"Final RMSE (Blended): {rmse:.2f}")

cv_scores = cross_val_score(stacked_model, X, y, cv=5, scoring='r2')
print(f"Average CV R² (Blended): {cv_scores.mean():.3f}")

# ------------------ Feature Importance ------------------

feature_names = X.columns
lgbm_importance = pd.Series(lgbm_best.feature_importances_, index=feature_names)
xgb_importance = pd.Series(xgb_best.feature_importances_, index=feature_names)
combined_importance = (lgbm_importance + xgb_importance) / 2

plt.figure(figsize=(12, 6))
sns.barplot(x=combined_importance.nlargest(15), y=combined_importance.nlargest(15).index)
plt.title("Top 15 Important Features")
plt.tight_layout()
plt.show()

# ------------------ Prediction on Test Set ------------------

test_df = test_cleaned.copy()
test_df['Years_Since_Est'] = 2025 - test_df['Outlet_Establishment_Year']
test_df['MRP_Visibility'] = test_df['Item_MRP'] * test_df['Item_Visibility']
test_df['Type_Size_Interaction'] = test_df['Item_Type'] + '_' + test_df['Outlet_Size'].astype(str)

test_df[cat_cols] = encoder.transform(test_df[cat_cols])

test_preds_log = stacked_model.predict(test_df)
test_preds = np.expm1(test_preds_log)
test_df['Item_Outlet_Sales'] = np.round(test_preds, 4)

# Decode categorical features back
decoded_ids = encoder.inverse_transform(test_df[cat_cols])
decoded_df = pd.DataFrame(decoded_ids, columns=cat_cols)
test_df[cat_cols] = decoded_df

final_predictions = test_df[['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales']]
print("\nFinal Predictions:")
print(final_predictions.head())

# Optionally save
# final_predictions.to_csv("submission.csv", index=False)


final_predictions.to_csv('submission4.csv')

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

####   Checking evaluation metrices in Random Forest Modelling, produced less R2 and RMSE as compared to LGBM & xGBOOST stacked alogrithm
## Taking too long to run in my system

# ------------------ Data Preparation ------------------

# train_rf = train_cleaned.copy()
# test_rf = test_cleaned.copy()

# for df in [train_rf, test_rf]:
#     df['MRP_Visibility'] = df['Item_MRP'] * df['Item_Visibility']
#     df['Type_Size_Interaction'] = df['Item_Type'] + '_' + df['Outlet_Size'].astype(str)
#     df['Years_Since_Est'] = 2025 - df['Outlet_Establishment_Year']

# train_rf = train_rf.dropna()

# X_rf = train_rf.drop(columns=['Item_Outlet_Sales'])
# y_rf = np.log1p(train_rf['Item_Outlet_Sales'])

# cat_cols_rf = X_rf.select_dtypes(include='object').columns.tolist()
# encoder_rf = OrdinalEncoder()
# X_rf[cat_cols_rf] = encoder_rf.fit_transform(X_rf[cat_cols_rf])

# X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(
#     X_rf, y_rf, test_size=0.2, random_state=42
# )

# # ------------------ Hyperparameter Tuning ------------------

# search_rf = BayesSearchCV(
#     RandomForestRegressor(random_state=42, n_jobs=-1),
#     search_spaces={
#         'n_estimators': Integer(100, 500),
#         'max_depth': Integer(5, 30),
#         'min_samples_split': Integer(2, 10),
#         'min_samples_leaf': Integer(1, 5)
#     },
#     n_iter=50,
#     cv=5,
#     scoring='r2',
#     verbose=0,
#     random_state=42,
#     n_jobs=-1
# )

# search_rf.fit(X_rf_train, y_rf_train)

# # ------------------ Evaluation ------------------

# rf_best = search_rf.best_estimator_

# y_rf_pred_log = rf_best.predict(X_rf_test)
# y_rf_pred = np.expm1(y_rf_pred_log)
# y_rf_test_actual = np.expm1(y_rf_test)

# r2_rf = r2_score(y_rf_test_actual, y_rf_pred)
# rmse_rf = mean_squared_error(y_rf_test_actual, y_rf_pred, squared=False)
# cv_rf = cross_val_score(rf_best, X_rf, y_rf, cv=5, scoring='r2')

# print(f"\n Random Forest (Tuned) R²: {r2_rf:.3f}")
# print(f" Random Forest (Tuned) RMSE: {rmse_rf:.2f}")
# print(f" Random Forest (Tuned) CV R² (5-fold): {cv_rf.mean():.3f}")

# # ------------------ Feature Importance ------------------

# importances_rf = pd.Series(rf_best.feature_importances_, index=X_rf.columns)
# plt.figure(figsize=(12, 6))
# sns.barplot(x=importances_rf.nlargest(15), y=importances_rf.nlargest(15).index)
# plt.title("Top 15 Important Features - Random Forest (Tuned)")
# plt.tight_layout()
# plt.show()

# # ------------------ Final Prediction on Test Set ------------------

# test_rf['MRP_Visibility'] = test_rf['Item_MRP'] * test_rf['Item_Visibility']
# test_rf['Type_Size_Interaction'] = test_rf['Item_Type'] + '_' + test_rf['Outlet_Size'].astype(str)
# test_rf['Years_Since_Est'] = 2025 - test_rf['Outlet_Establishment_Year']
# test_rf[cat_cols_rf] = encoder_rf.transform(test_rf[cat_cols_rf])

# test_preds_log_rf = rf_best.predict(test_rf)
# test_preds_rf = np.expm1(test_preds_log_rf)
# test_rf['Item_Outlet_Sales'] = np.round(test_preds_rf, 4)

# decoded_rf = pd.DataFrame(encoder_rf.inverse_transform(test_rf[cat_cols_rf]), columns=cat_cols_rf)
# test_rf[cat_cols_rf] = decoded_rf

# final_rf_predictions = test_rf[['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales']]
# print("\n Final Predictions from Tuned Random Forest:")
# print(final_rf_predictions.head())

# Save if needed:
# final_rf_predictions.to_csv("rf_bayes_submission.csv", index=False)

#----------------------------------------------------------------------------------------------------------------------------------------------
### Checking evaluation metrices in Cat Boost Algorithm, produced less R2 and RMSE as compared to LGBM & xGBOOST stacked alogrithm

####Taking too long to run in my system

# ------------- STEP 1: Copy & Clean Data -------------
# train_cb = train_cleaned.copy()
# test_cb = test_cleaned.copy()

# # ------------- STEP 2: Feature Engineering -------------

# def feature_engineering(df):
#     df['MRP_Visibility'] = df['Item_MRP'] * df['Item_Visibility']
#     df['Type_Size_Interaction'] = df['Item_Type'] + '_' + df['Outlet_Size'].astype(str)
#     df['Years_Since_Est'] = 2025 - df['Outlet_Establishment_Year']
#     return df

# train_cb = feature_engineering(train_cb)
# test_cb = feature_engineering(test_cb)

# # Drop NAs
# train_cb = train_cb.dropna()

# # ------------- STEP 3: Encode + Split -------------

# X_cb = train_cb.drop(columns=['Item_Outlet_Sales'])
# y_cb = np.log1p(train_cb['Item_Outlet_Sales'])

# # Identify categorical columns
# cat_cols_cb = X_cb.select_dtypes(include='object').columns.tolist()

# # Ensure categorical columns are string (required by CatBoost)
# for col in cat_cols_cb:
#     X_cb[col] = X_cb[col].astype(str)
#     test_cb[col] = test_cb[col].astype(str)

# # Get categorical column indices
# cat_features_idx_cb = [X_cb.columns.get_loc(col) for col in cat_cols_cb]

# # Train-validation split
# X_train_cb, X_val_cb, y_train_cb, y_val_cb = train_test_split(X_cb, y_cb, test_size=0.2, random_state=42)

# # ------------- STEP 4: Bayesian Hyperparameter Tuning -------------

# search_cb = BayesSearchCV(
#     CatBoostRegressor(
#         silent=True,
#         random_state=42,
#         thread_count=-1
#     ),
#     search_spaces={
#         'depth': Integer(4, 10),
#         'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
#         'iterations': Integer(100, 500),
#         'l2_leaf_reg': Real(1, 10, prior='log-uniform')
#     },
#     n_iter=30,
#     scoring='r2',
#     cv=5,
#     n_jobs=-1,
#     random_state=42
# )

# search_cb.fit(X_train_cb, y_train_cb, cat_features=cat_features_idx_cb)

# # ------------- STEP 5: Evaluate Best Model -------------

# best_cb = search_cb.best_estimator_

# y_pred_log_cb = best_cb.predict(X_val_cb)
# y_pred_cb = np.expm1(y_pred_log_cb)
# y_val_actual_cb = np.expm1(y_val_cb)

# r2_cb = r2_score(y_val_actual_cb, y_pred_cb)
# rmse_cb = mean_squared_error(y_val_actual_cb, y_pred_cb, squared=False)
# cv_cb = cross_val_score(best_cb, X_cb, y_cb, cv=5, scoring='r2', n_jobs=-1)

# print(f"\n CatBoost Final R² Score: {r2_cb:.3f}")
# print(f" CatBoost Final RMSE: {rmse_cb:.2f}")
# print(f" CatBoost Average CV R²: {cv_cb.mean():.3f}")

# # ------------- STEP 6: Feature Importance -------------

# importances_cb = pd.Series(best_cb.feature_importances_, index=X_cb.columns)
# plt.figure(figsize=(12, 6))
# sns.barplot(x=importances_cb.nlargest(15), y=importances_cb.nlargest(15).index)
# plt.title("Top 15 Important Features - CatBoost")
# plt.tight_layout()
# plt.show()

# # ------------- STEP 7: Predict on Test Data -------------

# test_preds_log_cb = best_cb.predict(test_cb)
# test_preds_cb = np.expm1(test_preds_log_cb)
# test_cb['Item_Outlet_Sales'] = np.round(test_preds_cb, 4)

# # Final prediction dataframe
# final_cb_preds = test_cb[['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales']]
# print("\n Final CatBoost Predictions:")
# print(final_cb_preds.head())

# # Optional: Save
# # final_cb_preds.to_csv("catboost_bayes_submission.csv", index=False)



















