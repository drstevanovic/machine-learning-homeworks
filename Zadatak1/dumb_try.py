import sys

import pandas as pd


def read_data():
    # train_path = './data/train.csv'
    # test_path = './data/test.csv'
    try:
        train_path = sys.argv[1]
        test_path = sys.argv[2]
    except Exception as e:
        # print('Exception while reading sys arguments!')
        print(666)

    df = pd.read_csv(train_path)
    df_test = df = pd.read_csv(test_path)

    return df, df_test


def train(data):
    # prepare data and calculate means
    data['xy'] = data['size'] * data['weight']
    data['x_squared'] = data['size'] * data['size']

    mean = data.mean()
    mean_x = mean['size']  # 3646.070652173913
    mean_y = mean['weight']  # 1287.358695652174
    mean_xy = mean['xy']
    mean_squared_x = mean['x_squared']
    squared_mean_x = mean_x * mean_x

    # find the model  --  y = mx + b
    m = (mean_x * mean_y - mean_xy) / (squared_mean_x - mean_squared_x)
    b = mean_y - m * mean_x
    return m, b


# calculate the RMSE

def calculate_rmse(df_test, m, b):
    df_test['predicted'] = df_test['size'] * m + b
    return ((df_test['predicted'] - df_test['weight']) ** 2).mean() ** .5


train_data, test_data = read_data()
m, b = train(train_data)
rmse = calculate_rmse(test_data, m, b)
print(rmse)
