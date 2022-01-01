import numpy as np
import matplotlib.pyplot as plt
from data.create_datasets import WindowGenerator


def in_reshape(data):
    """converting 3d array to 2d array for ML classic models"""
    dim_0 = data.shape[0]
    return np.reshape(data, (dim_0, -1))


def out_reshape(data, window=None):
    """converting 2d model results to 3d array"""
    return np.reshape(
        data, (data.shape[0], window.label_width, window.number_label_features)
    )


def plot_forecast(
    model, input_data=None, label_data=None, window=None, plot_col="T (degC)", ax=None
):
    if ax is None:
        ax = plt.gca()

    if plot_col not in (window.feature_columns and window.label_columns):
        raise ValueError("The chosen plot column does not exist in input/label data!")

    if label_data.shape[2] == 1:
        label_col_index = 0
    elif label_data.shape[2] > 1:
        label_col_index = window.label_columns.index(plot_col)

    if input_data.shape[2] == 1:
        input_col_index = 0
    elif input_data.shape[2] > 1:
        input_col_index = window.feature_columns.index(plot_col)

    preds = model.predict(in_reshape(input_data))
    preds = out_reshape(preds, window=window)

    ax.plot(
        window.label_indices, label_data[-1, :, label_col_index], "bo-", label="labels"
    )
    ax.plot(
        window.input_indices, input_data[-1, :, input_col_index], "go-", label="inputs"
    )
    ax.plot(
        window.label_indices, preds[-1, :, label_col_index], "ro", label="predictions"
    )
    ax.legend(loc="best")
    ax.set_xlabel("time index")
    ax.set_ylabel(plot_col)
    return


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

    def datasets(self, train_set=None, valid_set=None, test_set=None):
        self.datasets = {}
        for i in range(self.output_time_step):
            (
                self.datasets["input_train_" + str(i)],
                self.datasets["label_train_" + str(i)],
            ) = self.windows[i].create_dataset(train_set)
            (
                self.datasets["input_val_" + str(i)],
                self.datasets["label_val_" + str(i)],
            ) = self.windows[i].create_dataset(valid_set)
            (
                self.datasets["input_test_" + str(i)],
                self.datasets["label_test_" + str(i)],
            ) = self.windows[i].create_dataset(test_set)

    def model_fit(self, model):
        self.models = {}
        self.predictions = {}

        for i in range(self.output_time_step):

            input_train, label_train = (
                self.datasets["input_train_" + str(i)],
                self.datasets["label_train_" + str(i)],
            )

            input_test, label_test = (
                self.datasets["input_test_" + str(i)],
                self.datasets["label_test_" + str(i)],
            )

            self.models[i] = model.fit(in_reshape(input_train), in_reshape(label_train))
            print("Model on time step {} is being trained!".format(i + 1))

            self.predictions["train_" + str(i)] = out_reshape(
                self.models[i].predict(in_reshape(input_train)), window=self.windows[i]
            )

            self.predictions["test_" + str(i)] = out_reshape(
                self.models[i].predict(in_reshape(input_test)), window=self.windows[i]
            )

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

    def direct_future(self, data):

        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a three dimensional numpy array!")
        forecast = np.zeros(
            (1, self.output_time_step, self.windows[0].number_label_features)
        )
        for i in range(self.output_time_step):
            forecast[:, i : i + 1, :] = out_reshape(
                self.models[i].predict(in_reshape(data)), window=self.windows[i]
            )

        return forecast

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


class Recursive_Forecast(WindowGenerator):
    def __init__(
        self,
        output_time_step=15,
        input_width=24,
        feature_columns=None,
        label_columns=None,
    ):
        super().__init__(
            input_width=input_width,
            feature_columns=feature_columns,
            label_columns=label_columns,
        )
        self.output_time_step = output_time_step
        self.label_width = 1
        self.shift = 1

    def datasets(self, train_set=None, valid_set=None, test_set=None):
        self.datasets = {}
        (
            self.datasets["inputs_train"],
            self.datasets["labels_train"],
        ) = self.create_dataset(train_set)
        self.datasets["inputs_val"], self.datasets["labels_val"] = self.create_dataset(
            valid_set
        )
        (
            self.datasets["inputs_test"],
            self.datasets["labels_test"],
        ) = self.create_dataset(test_set)

        print(
            "train_set inputs and labels shape:{}{}\n"
            "valid_set inputs and labels shape:{}{}\n"
            "test_set inputs and labels shape:{}{}".format(
                self.datasets["inputs_train"].shape,
                self.datasets["labels_train"].shape,
                self.datasets["inputs_val"].shape,
                self.datasets["labels_val"].shape,
                self.datasets["inputs_test"].shape,
                self.datasets["labels_test"].shape,
            )
        )

    def model_fit(self, model):
        self.model = model.fit(
            in_reshape(self.datasets["inputs_train"]),
            in_reshape(self.datasets["labels_train"]),
        )

    def recursive_predictions(self, data="test"):

        if self.number_input_features != self.number_label_features:
            raise ValueError(
                "The number of input and label features should "
                "be equal for a recursive prediction"
            )

        output = np.array(
            out_reshape(
                self.model.predict(in_reshape(self.datasets["inputs_" + data])),
                window=self,
            )
        )
        dim1, dim2, dim3 = output.shape
        predictions = np.zeros((dim1, self.output_time_step, dim3))
        inputs = self.datasets["inputs_" + data]

        for n in range(self.output_time_step):
            prediction = np.array(
                out_reshape(self.model.predict(in_reshape(inputs)), window=self)
            )
            predictions[:, n : n + 1, :] = prediction
            inputs = inputs[:, -(self.input_width - 1) :, :]
            inputs = np.concatenate((inputs, prediction), axis=1)
        self.predictions = predictions[: (dim1 - self.output_time_step + 1), :, :]

        labels_revived = np.zeros(self.predictions.shape)
        for i in range(labels_revived.shape[0]):
            labels_revived[i, :, :] = self.datasets["labels_" + data][
                i : self.output_time_step + i, :, :
            ].reshape(self.output_time_step, -1)
        self.labels = labels_revived

        return self.predictions, self.labels

    def recursive_future(self, data):

        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a three dimensional numpy array!")

        forecast = np.zeros((1, self.output_time_step, self.number_label_features))

        for i in range(self.output_time_step):
            prediction = out_reshape(
                np.array(self.model.predict(in_reshape(data))), window=self
            )
            forecast[:, i : i + 1, :] = prediction
            data = data[:, -(self.input_width - 1) :, :]
            data = np.concatenate((data, prediction), axis=1)
        return forecast

    def plot_forecast_recursive(self, data="test", plot_col="T (degC)", ax=None):

        if ax is None:
            ax = plt.gca()

        self.recursive_predictions(data=data)
        dim1 = self.datasets["inputs_" + data].shape[0]
        input_data = self.datasets["inputs_" + data][
            : (dim1 - self.output_time_step + 1), :, :
        ]

        if self.labels.shape[2] == 1:
            label_col_index = 0
        elif self.labels.shape[2] > 1:
            label_col_index = self.label_columns.index(plot_col)
        if input_data.shape[2] == 1:
            input_col_index = 0
        elif input_data.shape[2] > 1:
            input_col_index = self.feature_columns.index(plot_col)

        output_index_start = self.input_indices[-1] + 1

        label_indices = np.arange(
            output_index_start, self.output_time_step + output_index_start
        )

        ax.plot(
            label_indices, self.labels[-1, :, label_col_index], "bo-", label="labels"
        )
        ax.plot(
            self.input_indices,
            input_data[-1, :, input_col_index],
            "go-",
            label="inputs",
        )
        ax.plot(
            label_indices,
            self.predictions[-1, :, label_col_index],
            "ro",
            label="prediction",
        )
        ax.legend(loc="best")
        ax.set_xlabel("time index")
        ax.set_ylabel(plot_col)
