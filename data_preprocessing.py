import numpy as np
import pandas as pd
from os import PathLike
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.base import TransformerMixin
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, pacf, acf


class DataPreprocessing:
    @staticmethod
    def preprocess_dataset(df_dataset: pd.DataFrame, encoders=None) -> (pd.DataFrame, dict[str, TransformerMixin]):
        if encoders is None:
            encoders = dict()

        for column in df_dataset.columns:
            values = encoders.get(column)
            if column in ['store_nbr', 'family', 'type', 'work_day']:
                if values is None:
                    encoders[column] = preprocessing.OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='if_binary')
                    encoders[column].fit(df_dataset[column].values.reshape(-1, 1))

                encoded_values = encoders[column].transform(df_dataset[column].values.reshape(-1, 1))
                encoded_df = pd.DataFrame(encoded_values, columns=encoders[column].get_feature_names_out([column]))
                df_dataset = pd.concat([df_dataset, encoded_df], axis=1)

                df_dataset.drop(columns=[column], inplace=True)
            elif column in ['onpromotion', 'oil_price', 'cluster']:
                if values is None:
                    encoders[column] = preprocessing.MinMaxScaler()
                    encoders[column].fit(df_dataset[column].values.reshape(-1, 1))

                df_dataset[column] = encoders[column].transform(df_dataset[column].values.reshape(-1, 1))
            elif column in ['month_day', 'week_day', 'month']:
                if values is None:
                    encoders[column] = preprocessing.LabelEncoder()
                    encoders[column].fit(df_dataset[column])

                df_dataset[column] = encoders[column].transform(df_dataset[column])

        return df_dataset, encoders

    @staticmethod
    def group_by_gap(df_dataset: pd.DataFrame, encoders: dict[str, TransformerMixin]) -> list[pd.DataFrame]:
        df_list = []

        df_dataset['date'] = pd.to_datetime(df_dataset['date'])
        for name in ['store_nbr', 'family']:
            df_dataset[name] = df_dataset[encoders[name].get_feature_names_out([name])].idxmax(axis=1).str.replace(f'{name}_', '')

        for (_, data) in df_dataset.groupby(['store_nbr', 'family']):
            data = data.sort_values('date').reset_index(drop=True)

            data['diff'] = data['date'].diff().dt.days
            data['group'] = (data['diff'] > 1).cumsum()
            data = data.drop(columns=['store_nbr', 'family', 'diff'])

            for (_, grouped_data) in data.groupby(['group']):
                grouped_data = grouped_data.sort_values('date').reset_index(drop=True)
                grouped_data = grouped_data.drop(columns=['date', 'group'])

                df_list.append(grouped_data)

        return df_list

    @staticmethod
    def create_dataset(dataset_path: str | PathLike) -> pd.DataFrame:
        parent_path = Path(dataset_path).parent
        oil_path = Path(parent_path) / 'oil.csv'
        holidays_path = Path(parent_path) / 'holidays_events.csv'
        stores_path = Path(parent_path) / 'stores.csv'

        df_dataset = pd.read_csv(dataset_path)
        stores_dataset = pd.read_csv(stores_path)
        oil_dataset = DataPreprocessing.load_oil_dataset(oil_path)
        holidays_dataset = DataPreprocessing.load_holidays_dataset(holidays_path)

        df_dataset = df_dataset.merge(oil_dataset, how='left', on=['date'])
        df_dataset = df_dataset.merge(stores_dataset, how='left', on=['store_nbr'])
        df_dataset = df_dataset.merge(holidays_dataset, how='left', on=['date'])
        df_dataset = DataPreprocessing.process_holidays(df_dataset)

        df_dataset['month_day'] = pd.to_datetime(df_dataset['date']).dt.day
        df_dataset['week_day'] = pd.to_datetime(df_dataset['date']).dt.weekday
        df_dataset['month'] = pd.to_datetime(df_dataset['date']).dt.month

        df_dataset.drop(columns=['transferred', 'description', 'event_type', 'locale', 'locale_name', 'city', 'state', 'id'], inplace=True)

        return df_dataset

    @staticmethod
    def process_holidays(df: pd.DataFrame) -> pd.DataFrame:
        to_drop = pd.DataFrame()

        for (group_values, group_data) in df.groupby(['date', 'store_nbr', 'family']):
            print(f'Processing group {group_values}')
            if len(group_data) > 1:
                right_item_index = group_data[group_data.apply(lambda row: DataPreprocessing.calculate_regionality(row), axis=1)].head(1).index

                if right_item_index.empty:
                    right_item_index = group_data.head(1).index
                    df.loc[right_item_index, 'event_type'] = pd.NA

                drop_items = group_data.drop(right_item_index)
                to_drop = pd.concat([to_drop, drop_items])

        print('Dropping items')
        df.drop(index=to_drop.index, inplace=True)
        print('Items dropped')

        df['work_day'] = pd.to_datetime(df['date']).dt.weekday.apply(lambda x: x < 5)
        df['work_day'] = df.apply(lambda row: DataPreprocessing.calculate_work_day(row) if pd.notna(row['event_type']) else row['work_day'], axis=1)

        return df

    @staticmethod
    def calculate_work_day(row: any) -> bool:
        return row['event_type'] == 'Work Day' or (not (row['event_type'] != 'Work Day' and DataPreprocessing.calculate_regionality(row)))


    @staticmethod
    def calculate_regionality(row: any) -> bool:
        return (row['locale'] == 'National' or
         (row['locale'] == 'Regional' and row['locale_name'] == row['state']) or
         (row['locale'] == 'Local' and row['locale_name'] == row['city']))


    @staticmethod
    def load_holidays_dataset(holidays_path: str | PathLike) -> pd.DataFrame:
        df = pd.read_csv(holidays_path)

        df.drop(df.loc[df['transferred'] == True].index, inplace=True)
        df.rename(columns={'type': 'event_type'}, inplace=True)
        to_drop = pd.DataFrame()

        for (group_values, group_data) in df.groupby(['locale_name', 'locale', 'date']):
            drop_items = group_data.drop(group_data.head(1).index)
            to_drop = pd.concat([to_drop, drop_items])

        df.drop(index=to_drop.index, inplace=True)

        return df

    @staticmethod
    def load_oil_dataset(oil_path: str | PathLike) -> pd.DataFrame:
        df = pd.read_csv(oil_path, parse_dates=['date'])

        start_date = df['date'].min()
        end_date = df['date'].max()
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        if pd.isna(df.loc[0, 'dcoilwtico']):
            df.loc[0, 'dcoilwtico'] = df.loc[1, 'dcoilwtico']

        df = df.set_index('date').reindex(full_date_range)
        df['dcoilwtico'] = df['dcoilwtico'].interpolate(method='time')
        df['dcoilwtico'] = df['dcoilwtico'].ffill()

        df = df.reset_index().rename(columns={'index': 'date', 'dcoilwtico': 'oil_price'})
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')

        return df

    @staticmethod
    def plot_seasonality(dataframes: list[pd.DataFrame], columns: list[str], lags: int = 50):
        for column in columns:
            acf_values = []
            pacf_values = []

            for dataframe in dataframes:
                series = dataframe[column]
                if series.unique().size == 1:
                    continue

                acf_values.append(acf(series, nlags=lags))
                pacf_values.append(pacf(series, nlags=lags))

            acf_series = np.mean(acf_values, axis=0)
            pacf_series = np.mean(pacf_values, axis=0)

            plt.stem(range(len(acf_series)), acf_series)
            plt.title(f"Autocorrelation Plot for {column}")
            # plt.show()
            plt.savefig(f'acf_{column}.png')
            plt.close()

            # Partial autocorrelation plot
            plt.stem(range(len(pacf_series)), pacf_series)
            plt.title(f"Partial Autocorrelation Plot for {column}")
            # plt.show()
            plt.savefig(f'pacf_{column}.png')
            plt.close()
