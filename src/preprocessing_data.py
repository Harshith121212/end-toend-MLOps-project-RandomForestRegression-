from sklearn.model_selection import train_test_split
from data_loader import raw_data

import pandas as pd

train_val_data, test_data = train_test_split(raw_data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=42)
#print(f'No. of train data: {train_data.shape}')
#print(f'No. of val data: {val_data.shape}')
#print(f'No. of test data: {test_data.shape}')

input_cols = list(train_data.columns)[0:-1]
#print(input_cols)
target_col = 'Selling_Price'

#print(train_data[target_col].isna().sum())

train_inputs = train_data[input_cols].copy()
train_targets = train_data[target_col].copy()
val_inputs = val_data[input_cols].copy()
val_targets = val_data[target_col].copy()
test_inputs = test_data[input_cols].copy()
test_targets = test_data[target_col].copy()

import numpy as np
numeric_cols = train_inputs.select_dtypes(include = np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()
#print(categorical_cols)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(raw_data[numeric_cols])
train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

#print(train_inputs[numeric_cols].describe())

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(raw_data[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
#print(encoded_cols)

train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols]).toarray()
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols]).toarray()
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols]).toarray()

pd.set_option('display.max_columns', None)
#print(test_inputs.head())

X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]

#print(X_train.head(2))
