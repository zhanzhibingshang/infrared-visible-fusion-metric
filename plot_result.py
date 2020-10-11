# coding: utf-8

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_result():
    result_csv_path = glob.glob('result/*.csv')
    columns = None

    result_np = []
    for result_path in result_csv_path:
        result = pd.read_csv(result_path)
        if columns is None:
            columns = result.columns
        result_np.append(result.to_numpy())
    result_np = np.array(result_np)
    result_np = result_np[0]
    plt.figure(figsize=(10, 4))
    for i, column_name in enumerate(columns[1:]):
        plt.plot(result_np[:, 0], result_np[:, i+1], lw=1)
        plt.scatter(result_np[:, 0], result_np[:, i+1])
        plt.xticks(result_np[:, 0])
        plt.title(column_name)
        plt.show()


if __name__ == '__main__':
    plot_result()
