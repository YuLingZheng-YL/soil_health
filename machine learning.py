import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor as XGBR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
df = pd.read_csv('tax.csv',index_col=0)
df = df.T
X = df.drop('SHI',axis=1)
y = df['SHI']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = preprocessing.StandardScaler().fit(X_train) 
X_train = scaler.transform(X_train, copy=True)
X_test = scaler.transform(X_test, copy=True) 
--------------------------------------------------------------
# RF
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [1,10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 3, 5],
    'max_features': ['sqrt', 'log2', None]
}
model = RandomForestRegressor(random_state=42)
#SVR
param_grid = {
    'C': [1, 10, 50, 100, 200, 500, 1000],
    'epsilon': [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1]
}
model = SVR()
#XGBoost
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.02, 0.05,0.1,0.2],
    'max_depth': [3,5,10]
}
model = XGBR(random_state=42)
---------------------------------------------------------------
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring='r2', n_jobs=10, verbose=1)
grid_search.fit(X_train, y_train)
print("optimal hyperparameters：", grid_search.best_params_)
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'best_estimator_model.joblib')
scoring = {
    'r2': 'r2',
    'rmse': make_scorer(mean_squared_error, squared=False),
    'mae': make_scorer(mean_absolute_error)
}
cv_results = cross_validate(
    best_model, 
    X_train, 
    y_train, 
    cv=10, 
    scoring=scoring,
    return_train_score=False
)
r2_scores = cv_results['test_r2']
rmse_scores = cv_results['test_rmse']
mae_scores = cv_results['test_mae']
print("R2 cross-validation:", r2_scores)
print("RMSE cross-validation:", rmse_scores)
print("MAE cross-validation:", mae_scores)
y_train_pred = best_model.predict(X_train)
print(f"Predicted SHI in the training dataset: ",y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
mae_train = mean_absolute_error(y_train, y_train_pred)
print(f"Measured SHI in training dataset: ",y_train)
print(f"R2 on training dataset for model: ", r2_train)
print(f"RMSE on training dataset for model: ", rmse_train)
print(f"MAE on training dataset for model: ", mae_train)
y_test_pred = best_model.predict(X_test)
print(f"Predicted SHI in the testing dataset: ",y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
mae_test = mean_absolute_error(y_test, y_test_pred)
print(f"Measured SHI in testing dataset: ",y_test)
print(f"R2 on testing dataset for model: ", r2_test)
print(f"RMSE on testing dataset for model: ", rmse_test)
print(f"MAE on testing dataset for model: ", mae_test)
import shap
import matplotlib.pyplot as plt
import pandas as pd
name = pd.read_csv('name.csv')
feature_names = name.Taxonomy
best_model= joblib.load("best_estimator_model.joblib")
explainer = shap.Explainer(best_model,X_train)
shap_values = explainer(X_train)
shap.summary_plot(shap_values, X_train, feature_names = feature_names, show=False, plot_size=(20.00, 10.00),max_display=40)
