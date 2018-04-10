import sys

import matplotlib.pyplot as plt
import pandas as pd


def read_data():
    train_path = './data/train.csv'
    test_path = './data/test.csv'
    try:
        train_path = sys.argv[1]
        test_path = sys.argv[2]
    except Exception as e:
        print('Exception while reading sys arguments!')

    print(train_path)
    print(test_path)

    df = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    return df, df_test


if __name__ == '__main__':
    train_data, test_data = read_data()
    print(train_data)
    brain_size = train_data['size']
    brain_weight = train_data['weight']
    plt.plot(brain_size, brain_weight, 'ro')
    plt.show()
