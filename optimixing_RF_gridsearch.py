# Importing the libraries
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('test_folder/SOL-USD_mod_1h.csv')
x = dataset.iloc[:, 5:-2].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,
                                                    random_state = 0
                                                    )


# Training the Random Forest Regression model on the Training set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=70, criterion="absolute_error",
                                   random_state=0
                                   )
regressor.fit(x_train, y_train)

# Evaluating regressions
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
y_pred = regressor.predict(x_test)
score = r2_score(y_test, y_pred)
print(f"R2 score: {score:.5f}")
adj_score = 1 - (1-score)*(len(x_train)-1)/(len(x_train)-1-8)
print(f"adjusted R2 score: {adj_score:.5f}")
cross = cross_val_score(estimator=regressor, X=x_train, y=y_train, cv=10)
print(f"cross validation score: {cross.mean():.5f}")
print(f"std dev of c.v. score: {cross.std():.5f}")
print("-" * 50)

# Grid Searching
# from sklearn.model_selection import GridSearchCV
# parameters = [{"n_estimators": [i for i in range(20, 81, 10)],
#                "criterion": ["squared_error", "absolute_error", "poisson"],
#                # "max_depth": [i for i in range(1,10)],
#                "min_samples_split": [i for i in range(2,5)],
#                "min_samples_leaf": [i for i in range(1,4)],
#                "min_weight_fraction_leaf": [i/10 for i in range(3)],
#                "max_features": ["auto", "sqrt", "log2"],
#                }]
# grid_search = GridSearchCV(estimator=regressor, param_grid=parameters, scoring="r2", cv=10, n_jobs=-1)
# grid_search.fit(x_train, y_train)
# best_accuracy = grid_search.best_score_
# best_parameter = grid_search.best_params_
# print(f"best accuracy: {best_accuracy.mean()*100:.2f}")
# print("best parameters: ", best_parameter)

# predicting result
np.set_printoptions(precision=2, threshold=sys.maxsize, suppress=True)
y_pred_2d = y_pred.reshape(len(y_pred), 1)
y_test_2d = y_test.reshape(len(y_test), 1)
with open("random_forest_result.txt", "w") as f:
    print(np.concatenate((y_pred_2d, y_test_2d), 1), file=f)