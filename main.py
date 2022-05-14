from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import yfinance as yf
import time
import datetime
import smtplib
import pickle
import csv_processing_functions_ver
import pygame

coin_list = [
             "BTC-USD", "ETH-USD", "BNB-USD",
             "XRP-USD", "SOL-USD", "LUNA1-USD", "ADA-USD",
             "DOGE-USD", "AVAX-USD", "DOT-USD", "SHIB-USD",
             "MATIC-USD", "NEAR-USD", "TRX-USD", "LTC-USD", "BCH-USD",
             "ATOM-USD", "APE3-USD", "ETC-USD", "GMT3-USD", "WBNB-USD",
             "FTM-USD", "ENS-USD", "SLP-USD", "WETH-USD",
            ]

# BUSD, USDC, USDT, UST are stablecoins
# FLEX untradeable


def get_data_from_yf():
    for stock in coin_list:
        data = yf.Ticker(stock).history(period="3mo", interval="1h")
        data.to_csv(path_or_buf="three_month_csv/" + stock + ".csv")


def process_csv():
    for stock in coin_list:
        csv_processing_functions_ver.make_csv_mod_and_last_row("three_month_mod_csv/" + stock + "_mod.csv",
                                                               "three_month_last_row_csv/" + stock + "_last_row.csv",
                                                               "three_month_csv/" + stock + ".csv",
                                                               method_of_download="auto")


def unpickle():
    with open("pickled_objects/regressor_list_pickle.pickle", "rb") as regressor_list_file:
        regressor_list = pickle.load(regressor_list_file)
    with open("pickled_objects/adj_r2_score_list_pickle.pickle", "rb") as score_list_file:
        adj_score_list = pickle.load(score_list_file)
    return regressor_list, adj_score_list


def predict_with_regressors(regressor_list, adj_score_list):
    prediction_list = []
    for i, stock in enumerate(coin_list):
        dataset = pd.read_csv('three_month_last_row_csv/' + stock + '_last_row.csv')
        x = dataset.iloc[:, 5:-2].values
        y_pred = regressor_list[i].predict(x)
        if y_pred >= 1:
            value_tuple = y_pred[0], stock, adj_score_list[i]
            prediction_list.append(value_tuple)

    prediction_list.sort(reverse=True)
    prediction_string = ""
    for a, b, c in prediction_list:
        prediction_string += f"{a:.2f}% increase for {b} (score: {c:.5f})\n"
    return prediction_list, prediction_string



def main():
    last_sent_time = datetime.datetime.min
    pygame.mixer.init()
    while True:
        print()
        print("getting data from yahoo finance...")
        print()
        get_data_from_yf()
        print()
        print("processing csv...")
        print()
        process_csv()
        print()
        print("unpickling regressors...")
        print()
        reg, score = unpickle()
        print()
        print("making predictions...")
        print()
        pred_list, pred_str = predict_with_regressors(reg, score)
        print()
        print("This is prediction for 3 hours ahead.")
        print(pred_str)
        last_sent_time = datetime.datetime.now()
        print(f"Printed on: {last_sent_time}")
        sound = pygame.mixer.Sound('mel_waow.wav')
        pygame.mixer.Sound.play(sound)
        time.sleep(600)


if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception as e:
            print("An exception has occurred.")
            print(e)
            time.sleep(60)
