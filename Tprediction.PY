import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,\
AdaBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings("ignore")

# The Goal is to train a regression model based on the data before 2010 and tetst on the rest.
# We have three section 1. Data Processing 2. Choosing the best Model 3. Tuning

#------------------------------------------------------------------------------------------
# 1. Processing the Data
df = pd.read_csv("Volve.csv", index_col=0)

# 1.1 We must truncate the first part of the data where everyting is zero (Measurement Has not started)
# Plotting the section that contains the kick off the production operation.

df.loc["2007-03-16":"2007-09-01"].plot(subplots=True, figsize=(8, 4))
df_production = df.loc["2007-05-17":]

# 1.2 Plotting the correlation between feature. 
plt.figure(figsize=(8, 4))
sns.heatmap(df_production.corr(), annot=True)
plt.show()

# 1.3 Because the P also is broken after 2010, P must be removed.
df_production = df.drop(['AVG_DOWNHOLE_PRESSURE'], axis=1)

# 1.4 Dividing the data based on the point when the T and P sensor got broken (2010-01-14)
# In this plot you can see that day.
df_production["AVG_DOWNHOLE_TEMPERATURE"].loc["2010-01-01":"2010-03-01"].plot(subplots=True, figsize=(8, 4))

# 1.4.1 We will train our model on this part od data because we have the Temperature in this part.
df1 = df_production.loc[:"2010-01-13"]
X = df1.drop("AVG_DOWNHOLE_TEMPERATURE", axis=1) 
y = df1["AVG_DOWNHOLE_TEMPERATURE"]

# 1.4.2. We want to predict the T in this Part because the T sensor is broken.
df2 = df_production.loc["2010-01-14":]
X_pred = df2.drop("AVG_DOWNHOLE_TEMPERATURE", axis=1)

#----------------------------------------------------------------------------------------------
# 2. Choosing the best models based for tuning using Pycaret library

try:
    from pycaret.regression import *
    s = setup(df1, target="AVG_DOWNHOLE_TEMPERATURE")
    best = compare_models()
    plot_model(best, "feature")
    plot_model(best, "residuals")

    ada_model = create_model("ada")
    gbr_model = create_model("gbr")
    plot_model(ada_model, plot="feature")
    plot_model(gbr_model, plot="feature")
    # As can be seen, in ADA other features like GAS VOL and WHP WHR are also important.
    
except Exception:
    pass

# Result: Running Pycaret Several times, It seems that Random Forest Regressor, Gradient Boosting Regressor,
# Extra Trees Regressor and AdaBoost Regressor are providing the best result.

#----------------------------------------------------------------------------------------------
# 3. Tuning Hyper Parameters for Models (1.rf 2.et 3.gbr 4.ada)
n_iteration = 5

# 3.1 Random Forest
rf = RandomForestRegressor()

#3.1.2 Gridding of Parameters using Random Grid
n_estimators = [5, 20, 50, 100]
criterion = ['squared_error']
max_depth = [int(x) for x in np.linspace(5, 100, num=10)]
min_samples_split = [2, 3, 4, 5, 6]
min_samples_leaf = [1, 2, 3, 4]
random_grid_rf = {"n_estimators": n_estimators, "criterion":criterion ,\
                  "max_depth":max_depth, "min_samples_split":min_samples_split,\
               "min_samples_leaf":min_samples_leaf}

# 3.1.3 Cross Validation with 10 folds
rf_randomgrid = RandomizedSearchCV(estimator=rf, param_distributions=random_grid_rf,\
                                    n_iter=n_iteration, random_state=22, n_jobs=-1, cv=10)
rf_randomgrid.fit(X, y)
print("The Tuned Parameters for Random Forest are:\n", rf_randomgrid.best_params_)

# 3.1.4. Training and Predicting based on Tuned Parameters
rf_tuned = RandomForestRegressor(n_estimators=rf_randomgrid.best_params_["n_estimators"],\
                                 criterion=rf_randomgrid.best_params_["criterion"],\
                                 max_depth=rf_randomgrid.best_params_["max_depth"],\
                                    min_samples_leaf=rf_randomgrid.best_params_["min_samples_leaf"],\
                                        min_samples_split=rf_randomgrid.best_params_["min_samples_split"])
rf_tuned.fit(X, y)
y_pred_rf = rf_tuned.predict(X)

# 3.1.5. Error Calcualtion
error = {}
r2 = r2_score(y, y_pred_rf)
mbe = mean_absolute_error(y, y_pred_rf)
mse = mean_squared_error(y, y_pred_rf)
error["rf"] = {"r2": r2, "mbe":mbe, "mse":mse}

#---------------------------------------------------------------------------------------------
# 3.2 Extra Tree
et = ExtraTreesRegressor()

#3.2.2 Gridding of Parameters using Random Grid
n_estimators = [150, 160, 170]
criterion = ['squared_error']
max_depth = [int(x) for x in np.linspace(5, 100, num=10)]
min_samples_split = [2, 3, 4, 5, 6]
min_samples_leaf = [1, 2, 3, 4]
random_grid_et = {"n_estimators": n_estimators, "criterion":criterion , "max_depth":max_depth, "min_samples_split":min_samples_split,\
               "min_samples_leaf":min_samples_leaf}

# 3.2.3 Cross Validation with 10 folds
et_randomgrid = RandomizedSearchCV(estimator=et, param_distributions=random_grid_et, n_iter=n_iteration, random_state=32, n_jobs=-1, cv=10)
et_randomgrid.fit(X, y)
print("The Tuned Parameters for Extra Tree are:\n", et_randomgrid.best_params_)

# 3.2.4. Training and Predicting based on Tuned Parameters
et_tuned = RandomForestRegressor(n_estimators=et_randomgrid.best_params_["n_estimators"],\
                                 criterion=et_randomgrid.best_params_["criterion"],\
                                 max_depth=et_randomgrid.best_params_["max_depth"],\
                                    min_samples_leaf=et_randomgrid.best_params_["min_samples_leaf"],\
                                        min_samples_split=et_randomgrid.best_params_["min_samples_split"])
et_tuned.fit(X, y)
y_pred_et = et_tuned.predict(X)

# 3.2.5. Error Calcualtion
r2 = r2_score(y, y_pred_et)
mbe = mean_absolute_error(y, y_pred_et)
mse = mean_squared_error(y, y_pred_et)
error["et"] = {"r2": r2, "mbe":mbe, "mse":mse}

#-------------------------------------------------------------------------------------------
# 3.3 Gradient Boost Regressor
gbr = GradientBoostingRegressor()

#3.3.2 Gridding of Parameters using Random Grid
n_estimators = [100, 125, 150]
criterion = ['squared_error']
max_depth = [int(x) for x in np.linspace(8, 32, num=10)]
min_samples_split = [2, 3, 4, 5, 6]
min_samples_leaf = [1, 2, 3, 4]
random_grid_gbr = {"n_estimators": n_estimators, "criterion":criterion ,\
                    "max_depth":max_depth, "min_samples_split":min_samples_split,\
               "min_samples_leaf":min_samples_leaf}

# 3.3.3 Cross Validation with 10 folds
gbr_randomgrid = RandomizedSearchCV(estimator=gbr, param_distributions=random_grid_gbr, n_iter=n_iteration,\
                                     random_state=42, n_jobs=-1, cv=10)
gbr_randomgrid.fit(X, y)
print("The Tuned Parameters for Gradient Boost Regression are:\n", gbr_randomgrid.best_params_)

# 3.3.4. Training and Predicting based on Tuned Parameters
gbr_tuned = RandomForestRegressor(n_estimators=gbr_randomgrid.best_params_["n_estimators"],\
                                 criterion=gbr_randomgrid.best_params_["criterion"],\
                                 max_depth=gbr_randomgrid.best_params_["max_depth"],\
                                    min_samples_leaf=gbr_randomgrid.best_params_["min_samples_leaf"],\
                                        min_samples_split=gbr_randomgrid.best_params_["min_samples_split"])
gbr_tuned.fit(X, y)
y_pred_gbr = gbr_tuned.predict(X)

# 3.3.5. Error Calcualtion
r2 = r2_score(y, y_pred_gbr)
mbe = mean_absolute_error(y, y_pred_gbr)
mse = mean_squared_error(y, y_pred_gbr)
error["gbr"] = {"r2": r2, "mbe":mbe, "mse":mse}

#-------------------------------------------------------------------------------------------
# 3.4 Ada Boost Regressor
ada = AdaBoostRegressor()

#3.4.2 Gridding of Parameters using Random Grid
n_estimators = [100, 125, 150]
learning_rate = [0.1, 0.3, 0.7, 1.0]
loss = ['linear', 'square', 'exponential']

# 3.4.3 Cross Validation with 10 folds
random_grid_ada = {"n_estimators":n_estimators, "learning_rate":learning_rate, "loss":loss}
ada_randomgrid = RandomizedSearchCV(estimator=ada, param_distributions=random_grid_ada,\
                                    n_iter=n_iteration, random_state=52, n_jobs=-1, cv=10)
ada_randomgrid.fit(X, y)
print("The best parameters for ADA Boost Regressor are:\n", ada_randomgrid.best_params_)

# 3.4.4. Training and Predicting based on Tuned Parameters
ada_tuned = AdaBoostRegressor(n_estimators=ada_randomgrid.best_params_["n_estimators"],\
                              learning_rate=ada_randomgrid.best_params_["learning_rate"],\
                                loss=ada_randomgrid.best_params_["loss"])
ada_tuned.fit(X, y)
y_pred_ada = ada_tuned.predict(X)

# 3.4.5. Error Calcualtion
r2 = r2_score(y, y_pred_ada)
mbe = mean_absolute_error(y, y_pred_ada)
mse = mean_squared_error(y, y_pred_ada)
error["ada"] = {"r2":r2, "mbe":mbe, "mse":mse}
print("The error metrics for trained Models are:\n", error)

# 3.1.6 Temperatue Prediction / time plot
df_T_prediction = df[['BORE_OIL_VOL', 'ON_STREAM_HRS',"AVG_DOWNHOLE_TEMPERATURE",\
                      'AVG_DP_TUBING']]
df_T_prediction["T_et_Prediction"] = df["AVG_DOWNHOLE_TEMPERATURE"]
df_T_prediction["T_et_Prediction"].loc["2010-01-14":] = et_tuned.predict(X_pred)

df_T_prediction["T_rf_Prediction"] = df["AVG_DOWNHOLE_TEMPERATURE"]
df_T_prediction["T_rf_Prediction"].loc["2010-01-14":] = rf_tuned.predict(X_pred)

df_T_prediction["T_gbr_Prediction"] = df["AVG_DOWNHOLE_TEMPERATURE"]
df_T_prediction["T_gbr_Prediction"].loc["2010-01-14":] = gbr_tuned.predict(X_pred)

df_T_prediction["T_ada_Prediction"] = df["AVG_DOWNHOLE_TEMPERATURE"]
df_T_prediction["T_ada_Prediction"].loc["2010-01-14":] = ada_tuned.predict(X_pred)

df_T_prediction.plot(subplots=True, figsize=(8, 4))

# As is obvious there is a difference between ADA and the other Ensemble Regression methods
# To adress this difference we are going to check the features for models

