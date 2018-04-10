import numpy
import pandas


def label_encoding(data, col_name="", col_values=[]):
    """
        Transforms values of the column named col_name to correspondent numbers from a set [0..N]
        where N is the number of different values.

    :param data: dataframe
    :param col_name: column to encode
    :param col_values: possible values in the column
    :return: dataset with encoded values
    """
    for i in range(len(col_values)):
        data[col_name][data[col_name] == col_values[i]] = i
    return data


def one_hot_encoding(data, col_name="", col_values=[]):
    """
    Generates a column for each possible of the column named 'col_name' and populates it with 0s and 1s,
    depending on the value of 'col_name'. The column named 'col_name' will be dropped, i.e. swapped with new columns.

    :param data: dataframe
    :param col_name: column to encode
    :param col_values: possible values in the column
    :return: dataset with encoded values
    """
    for col_val in col_values:
        data[col_name + '_' + col_val] = numpy.zeros(data[col_name].size)
        data[col_name + '_' + col_val][data[col_name] == col_val] = 1
    data = data.drop(columns=[col_name])
    return data


def train(x, y):
    pass


def main():
    dataset = pandas.read_csv('../data/test_preview.csv')

    dataset = label_encoding(dataset, 'sex', ['Female', 'Male'])
    dataset = label_encoding(dataset, 'discipline', ['A', 'B'])
    dataset = one_hot_encoding(dataset, 'rank', ['Prof', 'AsstProf', 'AssocProf'])

    x = dataset.drop(columns=['salary']).iloc[:, :].values
    y = dataset['salary'].iloc[:].values

    print(x)
    print(y)

    model = train(x, y)


if __name__ == '__main__':
    main()
