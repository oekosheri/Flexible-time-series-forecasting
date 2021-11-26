import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from data.create_datasets import WindowGenerator


def any_model(window, model, X, y, test_data):

    X, y, data = in_reshape(X), in_reshape(y), in_reshape(test_data)
    model.fit(X, y)
    return out_reshape(window=window, data=model.predict(data))


def in_reshape(data):
    """converting 3d array to 2d array for ML classic models"""
    dim_0 = data.shape[0]
    return np.reshape(data, (dim_0, -1))


def out_reshape(window=window, data=None):
    """converting 2d model results to 3d array"""
    return np.reshape(
        data, (data.shape[0], window.label_width, window.number_label_features)
    )


class Direct_Forecast:
    def __init__(
        self,
        output_time_step=15,
        input_width=24,
        feature_columns=None,
        label_columns=None,
    ):
        self.windows = {}
        self.output_time_step = output_time_step

        for i in range(self.output_time_step):
            self.windows[i] = WindowGenerator(
                input_width=input_width,
                label_width=1,
                shift=i + 1,
                label_columns=label_columns,
                feature_columns=feature_columns,
            )

    def datasets(self, train_set=None, test_set=None):
        self.datasets = {}
        for i in range(self.output_time_step):
            (
                self.datasets["input_train_" + str(i)],
                self.datasets["label_train_" + str(i)],
            ) = self.windows[i].create_dataset(train_set)
            (
                self.datasets["input_test_" + str(i)],
                self.datasets["label_test_" + str(i)],
            ) = self.windows[i].create_dataset(test_set)

    def model_fit(self, model):
        self.models = {}
        self.predictions = {}

        for i in range(self.output_time_step):
            self.models[i] = model(window=self.windows[i])
            print("Model on time step {} is being trained!".format(i + 1))
            input_train, label_train = (
                self.datasets["input_train_" + str(i)],
                self.datasets["label_train_" + str(i)],
            )
            input_val, label_val = (
                self.datasets["input_val_" + str(i)],
                self.datasets["label_val_" + str(i)],
            )
            input_test, label_test = (
                self.datasets["input_test_" + str(i)],
                self.datasets["label_test_" + str(i)],
            )
            self.history[i] = compile_and_fit(
                self.models[i],
                X=input_train,
                y=label_train,
                val_X=input_val,
                val_y=label_val,
                patience=patience,
                batch_size=batch_size,
                max_epochs=max_epochs,
            )
            self.predictions["train_" + str(i)] = self.models[i].predict(input_train)
            self.predictions["val_" + str(i)] = self.models[i].predict(input_val)
            self.predictions["test_" + str(i)] = self.models[i].predict(input_test)

    def direct_predictions(self, data="train"):
        dim1, _, dim2 = self.predictions[
            data + "_" + str(self.output_time_step - 1)
        ].shape
        self.prediction = np.zeros((dim1, self.output_time_step, dim2))
        self.labels = np.zeros((dim1, self.output_time_step, dim2))
        for i in range(self.output_time_step):
            j = self.output_time_step - i
            pred = np.array(self.predictions[data + "_" + str(i)])
            label = np.array(self.datasets["label_" + data + "_" + str(i)])
            self.prediction[:, i : i + 1, :] = pred[: (pred.shape[0] - (j - 1)), :, :]
            self.labels[:, i : i + 1, :] = label[: (label.shape[0] - (j - 1)), :, :]

        return self.prediction, self.labels

    def plot_forecast_direct(self, data="test", plot_col="T (degC)", ax=None):

        if ax is None:
            ax = plt.gca()

        window = self.windows[self.output_time_step - 1]
        input_data = self.datasets[
            "input_" + data + "_" + str(self.output_time_step - 1)
        ]
        self.direct_predictions(data=data)

        if plot_col not in (window.feature_columns and window.label_columns):
            raise ValueError(
                "The chosen plot column does not exist in input/label data!"
            )

        if self.labels.shape[2] == 1:
            label_col_index = 0
        elif self.labels.shape[2] > 1:
            label_col_index = window.label_columns.index(plot_col)
        #  print(label_col_index)

        if input_data.shape[2] == 1:
            input_col_index = 0
        elif input_data.shape[2] > 1:
            input_col_index = window.feature_columns.index(plot_col)
        #  print(input_col_index)

        output_index_start = window.input_indices + 1
        label_indices = np.arange(
            output_index_start[-1], self.output_time_step + output_index_start[-1]
        )

        ax.plot(
            label_indices, self.labels[-1, :, label_col_index], "bo-", label="labels"
        )
        ax.plot(
            window.input_indices,
            input_data[-1, :, input_col_index],
            "go-",
            label="inputs",
        )
        ax.plot(
            label_indices,
            self.prediction[-1, :, label_col_index],
            "ro",
            label="predictions",
        )
        ax.legend(loc="best")
        ax.set_xlabel("time index")
        ax.set_ylabel(plot_col)
