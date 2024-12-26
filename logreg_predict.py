import sys

import numpy as np
import pandas as pd

from model import Model
from prepare_dataset import TestData, load_test_dataset, HOUSES, HOUSES_COLUMN


def predict(data: TestData, weights_file: str, output_file: str) -> None:
    weights_file += '' if weights_file.endswith('.npz') else '.npz'
    model = Model(weights_file=weights_file)

    house_predictions: dict[str, np.ndarray] = {}
    for house, x in data:
        model(house=house)
        y = model.predict_proba(x)
        house_predictions[house] = y

    predictions = [ house_predictions[house] for house in HOUSES ]
    predictions = np.stack(predictions, axis=1)
    predictions = predictions.argmax(axis=1)

    output_df = pd.DataFrame({HOUSES_COLUMN: predictions})
    output_df[HOUSES_COLUMN] = output_df[HOUSES_COLUMN].map(lambda idx: HOUSES[idx])
    output_df.to_csv(output_file, index_label='Index')


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <dataset_test.csv> <weights.npy>")
        return

    dataset_file = sys.argv[1]
    weights_file = sys.argv[2]
    output_file  = 'output/houses.csv'

    try:
        x = load_test_dataset(dataset_file)
        predict(x, weights_file, output_file)

    except FileNotFoundError as error:
        print(f"File not found, Please double check !")
        print(error)
    except Exception as error:
        print("Internal Server Error: 500")
        print(error)


if __name__ == "__main__":
    main()

