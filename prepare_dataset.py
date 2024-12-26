from typing import Iterator

import numpy as np
import pandas as pd



HOUSES = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
HOUSES_COLUMN = "Hogwarts House"


# DATE_FEATURES = ['year', 'quarter', 'month', 'dayofweek', 'day']
DATE_FEATURES = ['year']

SHARED_FEATURES = ['Astronomy', 'Herbology', 'Defense Against the Dark Arts']

HOUSE_TO_FEATURES = {
    # 'Ravenclaw': ['Charms'],
    'Ravenclaw': ['Charms', 'Muggle Studies'],
    'Slytherin': ['Divination', 'Flying', 'Best Hand'],
    # 'Gryffindor': ['Transfiguration'],
    'Gryffindor': ['Transfiguration', 'Flying', 'History of Magic'],
    'Hufflepuff': []
}



TrainDataIterator = Iterator[tuple[str, np.ndarray, np.ndarray]]

class TrainData:
    def __init__(self, data: pd.DataFrame) -> None:
        self.__data = data.dropna() # INFO: drop nan
        # self.__data = data

    def __iter__(self) -> TrainDataIterator:
        for house, features in HOUSE_TO_FEATURES.items():
            house_features = SHARED_FEATURES + features + DATE_FEATURES
            y = self.__data[HOUSES_COLUMN] == house
            y = y.astype(int).to_numpy().reshape(-1, 1)
            x = self.__data[house_features].to_numpy()
            # INFO: here replace nan values with mean
            # x = self.__data[house_features]
            # x = x.fillna(0)
            # x = x.fillna(x.mean(axis=0))
            # x = x.to_numpy()
            yield house, x, y


TestDataIterator = Iterator[tuple[str, np.ndarray]]

class TestData:
    def __init__(self, data: pd.DataFrame) -> None:
        self.__data = data

    def __iter__(self) -> TestDataIterator:
        for house, features in HOUSE_TO_FEATURES.items():
            house_features = SHARED_FEATURES + features + DATE_FEATURES
            # x = self.__data[house_features].to_numpy()
            # INFO: here replace nan values with 0
            x = self.__data[house_features]
            x = x.fillna(x.mean(axis=0))
            # x = x.fillna(0)
            x = x.to_numpy()
            yield house, x


def __load_dataset_with_basic_preparation(path: str) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=['Birthday'])
    data['year'] = data['Birthday'].dt.year - 2000
    # data['month'] = data['Birthday'].dt.month
    # data['day'] = data['Birthday'].dt.day
    # data['dayofweek'] = data['Birthday'].dt.dayofweek
    # data['quarter'] = data['Birthday'].dt.quarter
    data = data.drop(columns=['Index', 'First Name', 'Last Name', 'Birthday'])
    data['Best Hand'] = data['Best Hand'].map(lambda hand: 1 if hand == 'Right' else 0)
    return data


def load_train_dataset_as_dataframe(path: str) -> pd.DataFrame:
    data = __load_dataset_with_basic_preparation(path)
    return data.sample(frac=1)


def load_train_dataset(path: str) -> TrainData:
    data = load_train_dataset_as_dataframe(path)
    return TrainData(data)


def load_test_dataset(path: str) -> TestData:
    data = __load_dataset_with_basic_preparation(path)
    return TestData(data)

