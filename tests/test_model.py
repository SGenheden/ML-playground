import pytest
import numpy as np
import pandas as pd

from ml_playground.data import DataContainer


@pytest.fixture
def simple_data(mocker):
    data_frame = pd.DataFrame.from_dict(
        {"a": [1, 2, 3], "b": [30, 40, 50], "c": [1, 1, 2]}
    )
    read_csv_mock = mocker.patch(
        "ml_playground.data.pd.read_csv", return_value=data_frame
    )
    data = DataContainer("dummy")
    read_csv_mock.assert_called_with("dummy")
    return data


def test_x_one_column(simple_data):
    simple_data.x = "a"
    x = simple_data.x
    assert np.array_equal(x.flatten(), [1, 2, 3])


def test_x_two_columns(simple_data):
    simple_data.x = ["a", "b"]
    x = simple_data.x
    assert np.array_equal(x, [[1, 30], [2, 40], [3, 50]])


def test_y(simple_data):
    simple_data.y = "c"
    y = simple_data.y
    assert np.array_equal(y, [1, 1, 2])
