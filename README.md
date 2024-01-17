* Exercise: Temperature Prediction

* Problem: We have a database from an oil field, including production data.
P and T sensors are broken after a time. Using the early  data, we want to predict
the T and P where the sensors are broken.

* Used Liberaris
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,\
AdaBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

* Pycaret: If you have it, you could see the comparison of models and if not,
it will be ignored automatically.
