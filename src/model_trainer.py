from inspect import signature

from sklearn.tree import DecisionTreeRegressor
from preprocessing_data import X_train, train_targets, X_test, test_targets, X_val, val_targets, encoder, encoded_cols, numeric_cols, scaler, categorical_cols
import pandas as pd



model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, train_targets)

from sklearn.metrics import accuracy_score
train_preds = model.predict(X_train)
val_preds= model.predict(X_val)

#print("Model accuracy score is: ",model.score(X_train, train_targets))

from sklearn.metrics import mean_absolute_error

train_mae = mean_absolute_error(train_targets, train_preds)
#print(f"Mean Absolute Error on the training set: {train_mae}")

val_mae = mean_absolute_error(val_targets, val_preds)
#print(f"Mean Absolute Error on the training set: {val_mae}")

from sklearn.ensemble import RandomForestRegressor
params = {"random_state":42, "max_depth":25, "min_samples_split":10, "n_estimators":400}
model = RandomForestRegressor(**params)
model.fit(X_train, train_targets)
accuracy = model.score(X_train, train_targets)

val_pred = model.predict(X_val)

import joblib
joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(encoder, 'encoder.joblib')

#print("Model score after using random forest is: ",model.score(X_train, train_targets))

#print("Model score for validation dataset: ",model.score(X_val, val_targets))

def predict_car_price(input):
  new_input_df = pd.DataFrame([input])
  new_input_df[numeric_cols] = scaler.transform(new_input_df[numeric_cols])
  new_input_df[encoded_cols] = encoder.transform(new_input_df[categorical_cols]).toarray()
  X_new_input = new_input_df[numeric_cols + encoded_cols]
  #model = RandomForestRegressor(random_state=42, max_depth=30, min_samples_split=5, n_estimators=500)
  #model.fit(X_train, train_targets)
  selling_prices = model.predict(X_new_input)[0]
  return selling_prices

new_input = {'Car_Name': 'ciaz',
             'Year': 2017,
             'Present_Price': 6.12,
             'Kms_Driven': 7900,
             'Fuel_Type': 'Petrol',
             'Seller_Type': 'Dealer',
             'Transmission': 'Manual',
}
print("estimated value of your car is: ", predict_car_price(new_input))








