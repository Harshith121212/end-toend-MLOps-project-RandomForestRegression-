import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import os
import pandas as pd
import numpy as np
import sklearn
import plotly.express as px


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to the CSV file
csv_path = os.path.join(current_dir, 'cardekho_data.csv') # Assuming CSV is in the same directory as data_loader.py
#print(raw_data.head())
raw_data = pd.read_csv(csv_path)

cols = list(raw_data.columns)
cols.remove('Selling_Price')
raw_data = raw_data[cols + ['Selling_Price']]

  # Exclude the 'Owner' column
raw_data = raw_data.drop('Owner', axis=1)

#print(raw_data.head())

#fig = px.scatter(raw_data, x='Kms_Driven', y='Selling_Price', title='Selling Price vs. Kms Driven')
#fig.show()

#raw_data.isna().sum()