# Artificial Neural Network Regression

# Importing the libraries
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
print(f"tensorflow version: {tf.__version__}")

# Importing the dataset
dataset = pd.read_csv('test_folder/SOL-USD_mod_1h.csv')
x = dataset.iloc[:, 5:-2].values
y = dataset.iloc[:, -1].values
print(x)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

while True:
    print("1: train new ANN")
    print("2: use last pickled ANN")
    answer = input("Please enter 1 or 2.")
    if answer == "1" or answer == "2":
        break

if answer == "1":
    # Building the ANN
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=40, activation="relu"))
    ann.add(tf.keras.layers.Dense(units=40, activation="relu"))
    ann.add(tf.keras.layers.Dense(units=1))
    ann.compile(optimizer='adam', loss="mean_squared_error")
    # training the ANN
    ann.fit(x_train, y_train, batch_size=32, epochs=2000)

if answer == "2":
    with open("ann_pickled.pickle", "rb") as pickled_ann_file:
        ann = pickle.load(pickled_ann_file)

# predicting the result
y_pred = ann.predict(x_test)
np.set_printoptions(precision=2, threshold=sys.maxsize, suppress=True)
y_pred_2d = y_pred.reshape(len(y_pred), 1)
y_test_2d = y_test.reshape(len(y_test), 1)
with open("ann_result.txt", "w") as f:
    print(np.concatenate((y_pred_2d, y_test_2d), 1), file=f)

# evaluating results
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
score = r2_score(y_test, y_pred)
print(f"R2 score: {score:.5f}")
adj_score = 1 - (1-score)*(len(x_train)-1)/(len(x_train)-1-8)
print(f"adjusted R2 score: {adj_score:.5f}")
print("-" * 50)

# pickling the ANN
with open("ann_pickled.pickle", "wb") as pickled_ann_file:
    pickle.dump(ann, pickled_ann_file)