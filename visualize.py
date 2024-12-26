import sys

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from prepare_dataset import load_train_dataset_as_dataframe, HOUSES_COLUMN, HOUSES


def histogram(df: pd.DataFrame) -> None:
    rows, cols = 4, 4
    _, ax = plt.subplots(rows, cols, figsize=(15, 15))
    for i, course in enumerate(df.columns[1:]):
        for house in HOUSES:
            ax[i // cols, i % cols].hist(df[df[HOUSES_COLUMN] == house][course], bins=20, alpha=0.3, label=house)

        ax[i // cols, i % cols].set_title(course)
        ax[i // cols, i % cols].legend()
    plt.tight_layout()
    plt.show()


def scatter_plot_pure_data(df: pd.DataFrame) -> None:
    rows, cols = 7, 10
    _, ax = plt.subplots(rows, cols, figsize=(35, 25))

    counter = 0
    for i, feature in enumerate(df.columns[:-1]):
        for other_f in df.columns[i+1:]:
            current_ax = ax[counter // cols, counter % cols]
            current_ax.scatter(df[feature], range(len(df[feature])), alpha=0.1, label=feature, color='r')
            current_ax.scatter(df[other_f], range(len(df[other_f])), alpha=0.1, label=other_f, color='b')
            current_ax.legend()
            counter += 1
    plt.tight_layout()
    plt.show()


def scatter_plot_normilized_data(df: pd.DataFrame) -> None:
    rows, cols = 7, 10
    _, ax = plt.subplots(rows, cols, figsize=(35, 25))

    min_max: dict[str, tuple] = { f: (min(df[f]), max(df[f])) for f in df.columns }

    data_range = range(df.shape[0])
    counter = 0
    for i, feature in enumerate(df.columns[:-1]):
        f_min, f_max = min_max.get(feature, (0, 1))
        feature_data = (df[feature] - f_min) / (f_max - f_min)

        for other_f in df.columns[i+1:]:
            o_f_min, o_f_max = min_max.get(other_f, (0, 1))
            other_f_data = (df[other_f] - o_f_min) / (o_f_max - o_f_min)

            current_ax = ax[counter // cols, counter % cols]

            current_ax.scatter(feature_data, data_range, alpha=0.2, label=feature, color='r')
            current_ax.scatter(other_f_data, data_range, alpha=0.2, label=other_f, color='b')

            current_ax.legend()
            counter += 1
    plt.tight_layout()
    plt.show()


def density_plot_normilized_data(df: pd.DataFrame) -> None:
    rows, cols = 7, 10
    _, ax = plt.subplots(rows, cols, figsize=(35, 25))

    min_max: dict[str, tuple] = { f: (min(df[f]), max(df[f])) for f in df.columns }

    counter = 0
    for i, feature in enumerate(df.columns[:-1]):
        f_min, f_max = min_max.get(feature, (0, 1))
        feature_data = (df[feature] - f_min) / (f_max - f_min)

        for other_f in df.columns[i+1:]:
            o_f_min, o_f_max = min_max.get(other_f, (0, 1))
            other_f_data = (df[other_f] - o_f_min) / (o_f_max - o_f_min)

            current_ax = ax[counter // cols, counter % cols]

            sns.kdeplot(feature_data, ax=current_ax, label=feature)
            sns.kdeplot(other_f_data, ax=current_ax, label=other_f)

            current_ax.legend()
            counter += 1
    plt.tight_layout()
    plt.show()


def pair_plot(df: pd.DataFrame) -> None:
    fig = sns.pairplot(df, hue=HOUSES_COLUMN, diag_kind='kde', kind='scatter', corner=True)
    # fig = sns.pairplot(df, hue=HOUSES_COLUMN, diag_kind='kde', kind='scatter')
    for i, var in enumerate(fig.x_vars):
        ax = fig.axes[i, i]
        ax.text(
            0.5,
            0.5,
            var,
            fontsize=9,
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes
        )
    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) != 3:
        print("Usage: python visualize.py <dataset.csv> <plot_type>")
        print("<plot_type> options: ")
        print("    histogram, scatter, scatter_normilized, density, pair")
        return

    first_question = ['Arithmancy', 'Potions', 'Care of Magical Creatures']
    # second_question = ['Transfiguration']

    plot_type = sys.argv[2]
    try:
        data = load_train_dataset_as_dataframe(sys.argv[1])

        if plot_type == "histogram":
            histogram(data)

        elif plot_type == "scatter":
            data.drop(columns=first_question, inplace=True)
            data.drop(columns=[HOUSES_COLUMN], inplace=True)
            scatter_plot_pure_data(data)

        elif plot_type == "scatter_normilized":
            data.drop(columns=first_question, inplace=True)
            data.drop(columns=[HOUSES_COLUMN], inplace=True)
            scatter_plot_normilized_data(data)

        elif plot_type == "density":
            data.drop(columns=first_question, inplace=True)
            data.drop(columns=[HOUSES_COLUMN], inplace=True)
            density_plot_normilized_data(data)

        elif plot_type == "pair":
            data.drop(columns=first_question, inplace=True)
            # data.drop(columns=second_question, inplace=True)
            pair_plot(data)

        else:
            print("Invalid plot_type. Options are:")
            print("    histogram, scatter, scatter_normilized, density, pair")

    except FileNotFoundError as error:
        print(f"Dataset file `{sys.argv[1]}' not found, Please double check !")
        print(error)
    except Exception as error:
        print("Internal Server Error: 500")
        print(error)


if __name__ == "__main__":
    main()

