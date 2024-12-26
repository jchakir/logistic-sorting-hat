import sys

import numpy as np
from matplotlib import pyplot as plt

from model import Model
from prepare_dataset import TrainData, load_train_dataset


History = tuple[list[float], list[float]]


def __train(data: TrainData) -> History:
    model = Model()
    history = ([], [])
    for house, x, y in data:
        input_shape = x.shape[1]

        model(house=house, input=input_shape)
        history = model.fit(x, y, alpha=0.01, epochs=25, batch_size=25, optimizer='mini-batch', verbose=0)

        loss, accuracy = model.test(x, y)
        print(f'{house:10}:   loss: {loss:.5f}   accuracy: {accuracy:.5f}')
    model.save('output/model')
    return history


def __do_multi_visualization_with_smoothing(
    values: list[list[float]],
    labels: list[str],
    colors: list[str]
) -> None:
    window = 250
    weights = np.ones(window) / window
    plt.figure(figsize=(15, 5))
    for value, label, color in zip(values, labels, colors):
        smoothed = np.convolve(value, weights, mode='valid')
        plt.plot(smoothed, label=label, color=color)
    plt.legend()
    plt.grid(True)
    plt.show()


def main() -> None:
    if len(sys.argv) != 2:
        print(f'Bad Arguments, expected: {sys.argv[0]} [data_file_name].csv')
        return
    try:
        data = load_train_dataset(sys.argv[1])
        loss, accuracy = __train(data)
        # __do_multi_visualization_with_smoothing(
        #     [loss, accuracy], ["train loss", "train acc"], ["green", "blue"]
        # )
    except FileNotFoundError as error:
        print(f"Dataset file `{sys.argv[1]}' not found, Please double check !")
        print(error)
    except Exception as error:
        print("Internal Server Error: 500")
        print(error)


if __name__ == '__main__':
    main()

