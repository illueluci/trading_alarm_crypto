# Artificial Neural Network Regression

# Importing the libraries
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
print(f"tensorflow version: {tf.__version__}")

# Importing the dataset
dataset = pd.read_csv('test_folder/SOL-USD_mod.csv')
x = dataset.iloc[:, 5:-3].values
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
    ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
    ann.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy"])
    # training the ANN
    ann.fit(x_train, y_train, batch_size=32, epochs=1000)

if answer == "2":
    with open("ann_pickled_c.pickle", "rb") as pickled_ann_file:
        ann = pickle.load(pickled_ann_file)

# predicting the result
y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)
np.set_printoptions(precision=2, threshold=sys.maxsize, suppress=True)
y_pred_2d = y_pred.reshape(len(y_pred), 1)
y_test_2d = y_test.reshape(len(y_test), 1)
with open("ann_result_c.txt", "w") as f:
    print(np.concatenate((y_pred_2d, y_test_2d), 1), file=f)

# evaluating results
from sklearn.metrics import confusion_matrix, accuracy_score
cf_m = confusion_matrix(y_test_2d, y_pred_2d)
print(f"confusion matrix of ANN: ")
print(cf_m)
acc_score = accuracy_score(y_test_2d, y_pred_2d)
print(f"accuracy score of ANN: {acc_score}")
print("-" * 50)

# pickling the ANN
with open("ann_pickled_c.pickle", "wb") as pickled_ann_file:
    pickle.dump(ann, pickled_ann_file)