import re  # For regex operations
import os  # For file and directory operations
import pickle
import numpy as np
import pandas as pd
import config # Custom configuration module for paths and settings
from typing import List # For type hinting lists
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

class LoadData:
    def _load_data(self):
        if not os.path.exists(config.DATA_PATH):
            raise FileExistsError(f'File Not Found !')

        self.df = pd.read_csv(config.DATA_PATH)

class PreprocessData(LoadData):
    def __init__(self):
        super().__init__()
        self.on_encode = OneHotEncoder()

    # Pipeline 1: Prediction
    # Method to clean data for prediction without saving the cleaned data
    def clean_df_predict(self, df=pd.DataFrame()):
        self.df = df
        self.df.drop_duplicates(inplace=True)             # Remove duplicate rows
        self._remove_extra_col([config.EXTRA_COL])        # Remove specified extra columns
        self._clean_display_col()                         # Clean the 'Display Size' column
        self._clean_weight_col()                          # Clean the 'Weight' column
        self._create_discount_col()                       # Create 'Discount Price' column
        self._fill_na_numerical_col(False)                # Fill missing values in numerical columns
        self._scale_data()                                # Scale numerical data
        self._remove_na_catogorical_col()                 # Remove missing values in categorical columns
        self._vectorize_catogorical_col(False)            # Encode categorical columns
        return self.df                                    # Return the cleaned dataframe

    # Pipeline 2: Cleaning
    # Method to clean data and save the cleaned result
    def clean_df(self):
        print('File clean started!')
        self._load_data()
        self.df.drop_duplicates(inplace=True)             # Remove duplicate rows
        self._remove_extra_col([config.EXTRA_COL])        # Remove specified extra columns
        self._clean_display_col()                         # Clean the 'Display Size' column
        self._clean_weight_col()                          # Clean the 'Weight' column
        self._create_discount_col()                       # Create 'Discount Price' column
        self._remove_outliers_from_imp_col()              # Remove outliers from important numerical columns
        self._fill_na_numerical_col()                     # Fill missing values in numerical columns
        self._scale_data()                                # Scale numerical data
        self._remove_na_catogorical_col()                 # Remove missing values in categorical columns
        self._vectorize_catogorical_col()                 # Encode categorical columns
        self._save_to_csv()                               # Save the cleaned data to CSV
        print(f'File clean End! and saved in location {config.CLEAN_FILE_PATH}')
        return self.df                                    # Return the cleaned dataframe

    # Method to remove extra columns from the dataframe
    def _remove_extra_col(self, extra_col: List[str]):
        self.df.drop(extra_col, axis=1, inplace=True)

    # Method to clean the 'Display Size' column
    def _clean_display_col(self):
        self.df['Display Size'].fillna('0.0 inches', inplace=True)
        self.df['Display Size'] = self.df['Display Size'].apply(
            lambda x: float(x.split()[0]))
        self.df['Display Size'].replace(0.0, np.nan, inplace=True)

    # Method to clean the 'Weight' column by converting ranges to averages
    def _clean_weight_col(self):
        cal = sum([int(x) for x in re.findall('\d+', '20 - 35 g ')]) / 2
        self.df['Weight'].replace('20 - 35 g', cal, inplace=True)

        cal = sum([int(x) for x in re.findall('\d+', '35 - 50 g')]) / 2
        self.df['Weight'].replace('35 - 50 g', cal, inplace=True)

        cal = sum([int(x) for x in re.findall('\d+', '50 - 75 g')]) / 2
        self.df['Weight'].replace('50 - 75 g', cal, inplace=True)

        self.df['Weight'].replace(
            '75g +', float(re.findall('\d+', '75g +')[0]), inplace=True)

        self.df['Weight'].replace('<= 20 g', float(
            re.findall('\d+', '<= 20 g')[0]), inplace=True)

    # Method to create a new column for discount price and drop the discount percentage column
    def _create_discount_col(self):
        self.df['Discount Price'] = (
            self.df['Original Price'] * (-self.df['Discount Percentage'])) / 100
        self.df.drop(['Discount Percentage'], axis=1, inplace=True)

    # Method to remove outliers using the Interquartile Range (IQR) method
    def _remove_outliers_IQR(self, data, col):
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data[(data[col] > lower_bound) & (data[col] < upper_bound)]

    # Method to remove outliers from important numerical columns
    def _remove_outliers_from_imp_col(self):
        import_col = config.IMP_COL_NUMERICAL
        for col in import_col:
            self.df = self._remove_outliers_IQR(self.df, col)

    # Method to fill missing values in numerical columns
    def _fill_na_numerical_col(self, save=True):
        if save:
            # Save the numerical columns to a file
            self.numerical_col = [
                feature for feature in self.df.columns if self.df[feature].dtype == 'float64']
            with open(os.path.join('dummys', 'numerical_col'), 'wb') as fp:
                pickle.dump(",".join(self.numerical_col), fp)
        else:
            # Load the numerical columns from a file
            with open(os.path.join('dummys', 'numerical_col'), 'rb') as fp:
                col = pickle.load(fp)
                self.numerical_col = col.split(',')
        # Fill missing values in numerical columns with the median
        for col in self.numerical_col:
            self.df[col].fillna(self.df[col].median(), inplace=True)

    # Method to scale numerical data using MinMaxScaler
    def _scale_data(self):
        scaler = MinMaxScaler()
        data = scaler.fit_transform(self.df[self.numerical_col[:-1]])
        data = pd.DataFrame(data, columns=self.numerical_col[:-1])
        self.df.drop(self.numerical_col[:-1], axis=1, inplace=True)
        self.df = pd.concat([self.df.reset_index(), data], axis=1)

    # Method to fill missing values in important categorical columns
    def _remove_na_catogorical_col(self):
        self.imp_col = config.IMP_COL_CATOGORICAL
        for col in self.imp_col[1:]:
            self.df[col].fillna('other', inplace=True)

    # Method to one-hot encode categorical columns
    def _vectorize_catogorical_col(self, save=True):
        brand_one_hot_df = self._one_hot_encode(self.df[['Brand']], 'Brand', save)
        model_name_one_hot_df = self._one_hot_encode(self.df[['Model Name']], 'Model Name', save)
        dial_shape_one_hot_df = self._one_hot_encode(self.df[['Dial Shape']], 'Dial Shape', save)
        strap_material_one_hot_df = self._one_hot_encode(self.df[['Strap Material']], 'Strap Material', save)

        self.df = pd.concat([self.df[self.numerical_col], brand_one_hot_df,
                             model_name_one_hot_df, dial_shape_one_hot_df, strap_material_one_hot_df], axis=1)

    # Method to encode a single categorical column and save/load encoder
    def _one_hot_encode(self, series_data, name, save=True):
        if save:
            self.on_encode.fit(series_data)
            self._save_encoder(name, self.on_encode)
        else:
            self.on_encode = self._load_encoder(name)
        brand_onehot = self.on_encode.transform(series_data)   # Transform the data using the encoder
        categories = self.on_encode.categories_[0]
        onehot_columns = [f'{name}_{cat}' for cat in categories]
        return pd.DataFrame(brand_onehot, columns=onehot_columns)

    # Method to save an encoder object to a file
    def _save_encoder(self, name, encoder):
        with open(os.path.join('dummys', name), 'wb') as fp:
            pickle.dump(encoder, fp)

    # Method to load an encoder object from a file
    def _load_encoder(self, name):
        with open(os.path.join('dummys', name), 'rb') as fp:
            return pickle.load(fp)

    # Method to convert a list into a dummy DataFrame
    def _convert_list_to_dummy(self, series, li):
        temp_df = pd.DataFrame(columns=li)
        temp_df.loc[len(temp_df)] = [(lambda x: 1 if x == series.iloc[0] else 0)(x) for x in li]
        return temp_df

    # Method to save the cleaned dataframe to a CSV file
    def _save_to_csv(self):
        self.df.to_csv(config.CLEAN_FILE_PATH, index=False)