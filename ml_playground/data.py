import pandas as pd


class DataContainer:
    def __init__(self, filename=None):

        self._x_columns = []
        self._y_column = None
        self._raw = None

        if filename:
            self.load(filename)

    @property
    def x(self):
        columns = [
            f"{name}_trans" if f"{name}_trans" in self._raw else name
            for name in self._x_columns
        ]
        return self._raw[columns].values

    @x.setter
    def x(self, value):
        if isinstance(value, list):
            self._x_columns.extend(value)
        else:
            self._x_columns = [value]

    @property
    def y(self):
        return self._raw[self._y_column].values

    @y.setter
    def y(self, value):
        self._y_column = value

    def apply(self, func, *arg, **kwargs):
        """
        Call a function on the raw data
        """
        ret = getattr(self._raw, func)(*arg, **kwargs)
        if isinstance(ret, pd.DataFrame):
            self._raw = ret

    def load(self, filename):
        """
        Load a new raw dataset
        """
        self._x_columns = []
        self._y_column = None
        self._raw = pd.read_csv(filename)

    def transform(self, column, transformer):
        self._raw[f"{column}_trans"] = transformer.transform(self._raw[column])


class TextTransformer:
    @staticmethod
    def transform(column):
        map_ = {value: i for i, value in enumerate(column.unique())}
        return [map_[value] for value in column]
