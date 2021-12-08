import os
import pandas as pd


def create_csv(data_part, filter_function, columns):
    path = 'dataset/VisDrone2020-CC/raw/annotations'
    files = filter(filter_function, os.listdir(path))
    files = map(lambda x: os.path.join(path, x), files)
    dfs = [pd.read_csv(ann, names=columns) for ann in files]
    for i in range(len(dfs)):
        dfs[i]['video'] = i
    df = pd.concat(dfs)
    print(df)
    df.to_csv(os.path.join(path, '..', 'heads', data_part), index=False)


if __name__ == '__main__':
    train = ('train.csv', lambda x: 'test' not in x, ['frame', 'x', 'y'])
    test = ('test.csv', lambda x: 'test' in x, ['frame', 'head_id', 'x', 'y', 'width', 'height', 'out', 'occl', 'undefinied1', 'undefinied2'])

    create_csv(*train)
    create_csv(*test)

# validator.expect_column_values_to_not_be_null(column="video", result_format='COMPLETE')
# validator.expect_column_values_to_not_be_null(column="frame", result_format='COMPLETE')
# validator.expect_column_values_to_not_be_null(column="x", result_format='COMPLETE')
# validator.expect_column_values_to_not_be_null(column="y", result_format='COMPLETE')
#
# validator.expect_column_values_to_be_integer_parseable(column="video", result_format='COMPLETE')
# validator.expect_column_values_to_be_integer_parseable(column="frame", result_format='COMPLETE')
# validator.expect_column_values_to_be_integer_parseable(column="x", result_format='COMPLETE')
# validator.expect_column_values_to_be_integer_parseable(column="y", result_format='COMPLETE')
#
# validator.expect_column_values_to_be_between(column="video", min_value=0, result_format='COMPLETE')
# validator.expect_column_values_to_be_between(column="frame", min_value=0, result_format='COMPLETE')
# validator.expect_column_values_to_be_between(column="x", min_value=0, result_format='COMPLETE')
# validator.expect_column_values_to_be_between(column="y", min_value=0, result_format='COMPLETE')
#
# from expect_column_values_to_be_integer_parseable import ExpectColumnValuesToBeIntegerParseable
#
# import os
# import numpy as np
# os.getcwd()
# import sys
# sys.path.insert(0, os.path.abspath('../plugins'))