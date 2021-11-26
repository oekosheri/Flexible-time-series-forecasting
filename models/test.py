import sys

sys.path.append("..")

import numpy as np
import pandas as pd
import tensorflow as tf

# tf.autograph.set_verbosity(0)
from sklearn.metrics import mean_absolute_error, mean_squared_error

import models.models_nn as nn
from data.create_datasets import WindowGenerator

df = pd.read_csv("../datasets/df_feat_enged")

# split data
n = len(df)
train_df = df[0 : int(n * 0.7)]
val_df = df[int(n * 0.7) : int(n * 0.9)]
test_df = df[int(n * 0.9) :]

# Normalize data
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

models = [nn.conv_model, nn.LSTM_model_encoder_decoder, nn.dense_model]
models_n = ["conv", "LSTM_en_de", "dense"]

scores = {}

for i in range(len(models)):

    print("steps:", i)

    recursive = nn.Recursive_Forecast(
        output_time_step=15,
        input_width=24,
        feature_columns=["p (mbar)", "T (degC)", "VPdef (mbar)"],
        label_columns=["p (mbar)", "T (degC)", "VPdef (mbar)"],
    )
    recursive.datasets(train_set=train_df, valid_set=val_df, test_set=test_df)
    recursive.model_fit(models[i], patience=5, max_epochs=30)
    print("\n\n")
    print("Fitting done!", i)
    # IPython.display.clear_output()
    preds, labels = recursive.recursive_predictions(data="test")

    print("preds worked!", i)

    scores[models_n[i]] = mean_squared_error(
        preds.flatten(), labels.flatten(), squared=False
    )

    # scores[models_n[i]] = mean_squared_error(
    #     recursive.recursive_predictions(data="test")[0].flatten(),
    #     recursive.recursive_predictions(data="test")[1].flatten(),
    #     squared=False,
    # )

# print(scores)
