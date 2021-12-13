import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from data.create_datasets import WindowGenerator


def conv_model(window, filters=10, kernel_size=3, activation="elu"):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation=activation,
                # activity_regularizer=regularizers.l2(1e-3),
            ),
            tf.keras.layers.MaxPool1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                units=(window.label_width * window.number_label_features)
            ),
            # Add back the time dimension.
            # Shape: (outputs) => (1, outputs)
            tf.keras.layers.Reshape([window.label_width, window.number_label_features]),
        ]
    )
    return model


def conv_model_recursive_train(window, filters=10, kernel_s=3, activation="elu"):

    if window.number_input_features != window.number_label_features:
        raise ValueError(
            "The number of input and label features should "
            "be equal for a recursive prediction"
        )

    # global layers

    Input_layer = tf.keras.Input(
        shape=(window.input_width, window.number_input_features)
    )
    Conv_layer = tf.keras.layers.Conv1D(
        filters=filters, kernel_size=kernel_s, activation=activation
    )
    Flatten_layer = tf.keras.layers.Flatten()
    Dense_layer = tf.keras.layers.Dense(window.number_label_features)
    Reshape_layer = tf.keras.layers.Reshape([1, window.number_label_features])
    Lambda_layer = tf.keras.layers.Lambda(
        lambda x: x[:, -(window.input_width - 1) :, :]
    )
    concat_layer = tf.keras.layers.Concatenate(axis=1)

    # warm up
    def Conv_round(Input_layer):
        output = Conv_layer(Input_layer)
        output = Flatten_layer(output)
        prediction = Dense_layer(output)
        return prediction

    predictions = []

    # Insert the first prediction
    prediction = Conv_round(Input_layer)
    predictions.append(prediction)

    for n in range(1, window.label_width):
        # Use the last prediction as input.
        x1 = prediction
        x1 = Reshape_layer(x1)
        x2 = Lambda_layer(Input_layer)
        x3 = concat_layer([x2, x1])

        # Execute one Conv block.
        prediction = Conv_round(x3)
        predictions.append(prediction)

    # predictions.shape => (time, batch, features)
    predictions = tf.stack(predictions)
    # predictions.shape => (batch, time, features)
    predictions = tf.transpose(predictions, [1, 0, 2])

    model = tf.keras.Model(inputs=Input_layer, outputs=predictions)
    return model


def LSTM_model_vec_out(window, units=10):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(units, return_sequences=False),
            tf.keras.layers.Dense(
                units=(window.label_width * window.number_label_features)
            ),
            tf.keras.layers.Reshape([window.label_width, window.number_label_features]),
        ]
    )

    return model


def LSTM_model_encoder_decoder(window, encoder_units=10, decoder_units=10):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(encoder_units, return_sequences=False),
            tf.keras.layers.RepeatVector(window.label_width),
            tf.keras.layers.LSTM(decoder_units, return_sequences=True),
            tf.keras.layers.Dense(window.number_label_features),
        ]
    )

    return model


def LSTM_model_recursive_train(window, units=10):

    if window.number_input_features != window.number_label_features:
        raise ValueError(
            "The number of input and label features should "
            "be equal for a recursive prediction"
        )

    # global layers
    Input_layer = tf.keras.Input(
        shape=(window.input_width, window.number_input_features)
    )
    LSTM_layer = tf.keras.layers.LSTM(units, return_state=True)
    # LSTM_cell = tf.keras.layers.LSTMCell(LSTM_units)
    # LSTM_RNN = tf.keras.layers.RNN(LSTM_cell, return_state=True)
    Reshape_layer = tf.keras.layers.Reshape((1, window.number_label_features))
    Dense_layer = tf.keras.layers.Dense(window.number_label_features)

    # warm up
    output, *state = LSTM_layer(Input_layer)
    # output, *state = LSTM_RNN(Input_layer)
    prediction = Dense_layer(output)

    predictions = []

    # Insert the first prediction
    predictions.append(prediction)

    for n in range(1, window.label_width):
        # Use the last prediction as input.
        x = prediction
        x = Reshape_layer(x)
        # Execute one lstm step.
        x, *state = LSTM_layer(x, initial_state=state)
        # x, state = LSTM_cell(x, states=state)
        # Convert the lstm output to a prediction.
        prediction = Dense_layer(x)
        # Add the prediction to the output
        predictions.append(prediction)
    # predictions.shape => (time, batch, features)
    predictions = tf.stack(predictions)
    # predictions.shape => (batch, time, features)
    predictions = tf.transpose(predictions, [1, 0, 2])

    model = tf.keras.Model(inputs=Input_layer, outputs=predictions)
    return model


def dense_model(window, units=10, activation="elu"):

    model = tf.keras.Sequential(
        [
            # Shape: (time, features) => (time*features)
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                units=units,
                activation=activation,
                # activity_regularizer=regularizers.l2(1e-3),
            ),
            tf.keras.layers.Dense(
                units=(window.label_width * window.number_label_features)
            ),
            # Add back the time dimension.
            # Shape: (outputs) => (1, outputs)
            tf.keras.layers.Reshape([window.label_width, window.number_label_features]),
        ]
    )
    return model


def compile_and_fit(
    model,
    X=None,
    y=None,
    val_X=None,
    val_y=None,
    patience=5,
    batch_size=32,
    max_epochs=30,
):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, mode="min"
    )

    model.compile(
        loss=tf.losses.MeanSquaredError(),
        optimizer=tf.optimizers.Adam(),
        metrics=[tf.metrics.MeanAbsoluteError()],
    )

    history = model.fit(
        X,
        y,
        epochs=max_epochs,
        batch_size=batch_size,
        validation_data=(val_X, val_y),
        callbacks=[early_stopping],
    )
    return history


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

    preds = model.predict(input_data)
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
    ax.xlabel("time index")
    ax.ylabel(plot_col)
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

    #             print("train_set inputs and labels shape for {} timestep:{}{}\n"
    #                   "valid_set inputs and labels shape:{}{}\n"
    #                   "test_set inputs and labels shape:{}{}".format(i, self.datasets["input_train_"+ str(i)].shape, self.datasets["label_train_"+ str(i)].shape,
    #                                                              self.datasets["input_val_"+ str(i)].shape, self.datasets["label_val_"+ str(i)].shape,
    #                                                              self.datasets["input_test_"+ str(i)].shape, self.datasets["label_test_"+ str(i)].shape))

    def model_fit(self, model, patience=5, batch_size=32, max_epochs=30):
        self.models = {}
        self.history = {}
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
        print("preds and labels", self.prediction.shape, self.labels.shape)
        for i in range(self.output_time_step):
            j = self.output_time_step - i
            pred = np.array(self.predictions[data + "_" + str(i)])
            label = np.array(self.datasets["label_" + data + "_" + str(i)])
            print(pred.shape, pred[: (pred.shape[0] - (j - 1)), :, :].shape)
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

    def model_fit(self, model, patience=5, batch_size=32, max_epochs=30):
        self.model = model(window=self)
        self.history = compile_and_fit(
            self.model,
            X=self.datasets["inputs_train"],
            y=self.datasets["labels_train"],
            val_X=self.datasets["inputs_val"],
            val_y=self.datasets["labels_val"],
            patience=patience,
            batch_size=batch_size,
            max_epochs=max_epochs,
        )

    def recursive_predictions(self, data="test"):

        if self.number_input_features != self.number_label_features:
            raise ValueError(
                "The number of input and label features should "
                "be equal for a recursive prediction"
            )

        output = np.array(self.model.predict(self.datasets["inputs_" + data]))
        dim1, dim2, dim3 = output.shape
        predictions = np.zeros((dim1, self.output_time_step, dim3))
        inputs = self.datasets["inputs_" + data]

        for n in range(self.output_time_step):
            prediction = np.array(self.model.predict(inputs))
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
