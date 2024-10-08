import pandas as pd
import pathlib
import numpy as np


def load_data(data_path):
    df = pd.read_parquet(data_path)
    return df

def save_data(train, validate, test, output_path):
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    train.to_csv(output_path + 'train.csv', index=False)
    validate.to_csv(output_path + 'validate.csv', index=False)
    test.to_csv(output_path + 'test.csv', index=False)


def make_columns(df):
    df[['dyu', 'fr']] = df['translation'].apply(pd.Series)
    df.drop(columns=['translation'], inplace=True)
    return df

def lower_case(df):
    df['dyu'] = df['dyu'].apply(lambda x: x.lower())
    df['fr'] = df['fr'].apply(lambda x: x.lower())
    return df

def sentence_length(df):
    df['dyu_length'] = df['dyu'].apply(lambda x: len(x.split()))
    df['fr_length'] = df['fr'].apply(lambda x: len(x.split()))
    return df



if __name__ == "__main__":
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    train_path = home_dir.as_posix() + '/data/raw/train-00000-of-00001.parquet'
    validate_path = home_dir.as_posix() + '/data/raw/validation-00000-of-00001.parquet'
    test_path = home_dir.as_posix() + '/data/raw/test-00000-of-00001.parquet'

    train = load_data(train_path)
    validate = load_data(validate_path)
    test = load_data(test_path)

    train = make_columns(train)
    validate = make_columns(validate)
    test = make_columns(test)

    train = lower_case(train)
    validate = lower_case(validate)
    test = lower_case(test)

    train = sentence_length(train)
    validate = sentence_length(validate)
    test = sentence_length(test)

    save_data(train, validate, test, home_dir.as_posix() + '/data/processed/')
