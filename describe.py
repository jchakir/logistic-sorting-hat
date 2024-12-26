import sys
import math

import pandas as pd
import numpy as np

from prettytable import PrettyTable

from prepare_dataset import load_train_dataset_as_dataframe



def __calc_variance(data: np.ndarray, mean: float, count: int) -> float:
    total: float = sum((num - mean) ** 2 for num in data)
    return total / (count - 1)


def __get_middle(data: np.ndarray) -> float:
    count = len(data)

    if count % 2: # odd
        middle = count // 2
        quartile = data[middle]
    else: # even
        middle = count // 2
        quartile = (data[middle - 1] + data[middle]) / 2
    return quartile


def __describe_column(data: pd.Series) -> tuple[float, ...]:
    count, total = .0, .0
    min, max = .0, .0
    q1, q2, q3 = .0, .0, .0
    mean, var, std = .0, .0, .0

    # filter data from NaN
    data = filter(lambda num: not np.isnan(num), data)
    # sort data asc
    data = sorted(data)

    count, total = len(data), sum(data)
    min, max = data[0], data[-1]

    mean = total / count
    var = __calc_variance(data, mean, count)
    std = math.sqrt(var)

    q2 = __get_middle(data)

    if count % 2: # odd
        middle = count // 2
        q1 = __get_middle(data[:middle + 1])
        q3 = __get_middle(data[middle:])
    else: # even
        middle = count // 2
        q1 = __get_middle(data[:middle])
        q3 = __get_middle(data[middle:])

    return tuple(map(lambda itm: f'{itm:.6f}', (count, mean, var, std, min, q1, q2, q3, max)))


def describe(df: pd.DataFrame) -> None:
    table = PrettyTable()

    table.add_column(' ', ('Count', 'Mean', 'Var', 'Std', 'Min', '25%', '50%', '75%', 'Max'))
    numerical_df = df.select_dtypes(include=['number'])
    for column in numerical_df.columns:
        described_col = __describe_column(numerical_df[column])
        table.add_column(column, described_col)
    print(table)


def main() -> None:
    if len(sys.argv) != 2:
        print(f'Bad Arguments, expected: {sys.argv[0]} [data_file_name].csv')
        return

    try:
        data = load_train_dataset_as_dataframe(sys.argv[1])
        describe(data)

    except FileNotFoundError as error:
        print(f"Dataset file `{sys.argv[1]}' not found, Please double check !")
        print(error)
    except Exception as error:
        print("Internal Server Error: 500")
        print(error)


if __name__ == '__main__':
    main()

