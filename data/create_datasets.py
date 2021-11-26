import numpy as np
import pandas as pd


class WindowGenerator:
    def __init__(
        self,
        input_width=24,
        label_width=1,
        shift=1,
        feature_columns=None,
        label_columns=None,
    ):

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + self.shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        self.features = feature_columns
        self.labels = label_columns

    def __repr__(self):
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.input_indices}",
                f"Label indices: {self.label_indices}",
                f"Label column name(s): {self.labels}",
                f"Feature column name (s): {self.features}",
            ]
        )

    def create_dataset(self, data, stride=1, shuffle=False):

        # work out the column indices of data
        self.column_indices = {name: i for i, name in enumerate(data.columns)}
        if self.features is None:
            self.feature_columns = list(data.columns)
        else:
            self.feature_columns = self.features

        if self.labels is None:
            self.label_columns = list(data.columns)
        else:
            self.label_columns = self.labels

        data_set = []
        j = np.arange(0, len(data) - self.total_window_size, stride)
        for i in j:
            v = data.iloc[i : (i + self.total_window_size)].values
            data_set.append(v)

        final_dataset = np.array(data_set, dtype="object")
        if shuffle:
            np.random.shuffle(final_dataset)

        return self.split_window(final_dataset)

    def split_window(self, dataset):

        inputs = dataset[:, self.input_slice, :]
        labels = dataset[:, self.labels_slice, :]

        inputs = np.stack(
            [inputs[:, :, self.column_indices[name]] for name in self.feature_columns],
            axis=-1,
        )
        self.number_input_features = len(self.feature_columns)

        labels = np.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1,
        )
        self.number_label_features = len(self.label_columns)

        return inputs.astype("float64"), labels.astype("float64")
