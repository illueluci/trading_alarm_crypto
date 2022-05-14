# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

# Importing the dataset
dataset = pd.read_csv('test_folder/SOL-USD_mod_1h.csv')
x = dataset.iloc[:, 5:-2].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,
                                                    random_state = 0
                                                    )

# Training the XGBRegression model on the Training set
from xgboost import XGBRegressor
regressor = XGBRegressor()
regressor.fit(x_train, y_train)

# predicting results
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2, threshold=sys.maxsize, suppress=True)
y_pred_2d = y_pred.reshape(len(y_pred), 1)
y_test_2d = y_test.reshape(len(y_test), 1)
with open("xgboost_result.txt", "w") as f:
    print(np.concatenate((y_pred_2d, y_test_2d), 1), file=f)

# evaluating results
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
score = r2_score(y_test, y_pred)
print(f"R2 score: {score:.5f}")
adj_score = 1 - (1-score)*(len(x_train)-1)/(len(x_train)-1-8)
print(f"adjusted R2 score: {adj_score:.5f}")
cross = cross_val_score(estimator=regressor, X=x_train, y=y_train, cv=10)
print(f"cross validation score: {cross.mean():.5f}")
print(f"std dev of c.v. score: {cross.std():.5f}")
print("-" * 50)
