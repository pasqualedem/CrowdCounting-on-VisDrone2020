import os

import pandas as pd
from sklearn import model_selection


def load_test_dataset(data_folder, dataclass, make_dataframe_fun):
    # Load the test dataframe
    test_df = make_dataframe_fun(data_folder)

    # Instantiate the dataset
    test_data = dataclass(data_folder, test_df)
    return test_data

# TO DO
"""def load_train_valid_datasets(valid_size=0.1):
    # Load the train dataframe
    filepath = os.path.join(ROOT_DATAPATH, 'train.csv')
    df = pd.read_csv(
        filepath, sep=',', usecols=['filename', 'class'],
        converters={'class': lambda c: 1 if c == 'positive' else 0}
    )

    # Split the dataframe in train and validation
    train_df, valid_df = model_selection.train_test_split(
        df, test_size=valid_size, shuffle=True, stratify=df['class']
    )

    # Instantiate the datasets (notice data augmentation on train data)
    train_data = PeopleFlowsDataset(TRAIN_DATAPATH, train_df)
    valid_data = PeopleFlowsDataset(TRAIN_DATAPATH, valid_df)
    return train_data, valid_data"""