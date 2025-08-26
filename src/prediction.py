from inspect import signature

from sklearn.tree import DecisionTreeRegressor
from preprocessing_data import X_train, train_targets, X_test, test_targets, X_val, val_targets, encoder, encoded_cols, numeric_cols, scaler, categorical_cols
import pandas as pd
import mlflow
from mlflow.models import infer_signature


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


mlflow.set_tracking_uri(uri='http://localhost:5000')
mlflow.set_experiment("2nd MLFlow integration")

with mlflow.start_run():
  for key, value in params.items():
    mlflow.log_param(key, value)
  mlflow.log_metric("model score", accuracy)
  signature=infer_signature(X_train, model.predict(X_train))
  model_info = mlflow.sklearn.log_model(
    sk_model=model,
    name="model",
    signature=signature,
    input_example=X_train,
    registered_model_name="Used_Car_Price_Prediction"
  )
  mlflow.set_logged_model_tags(
    model_info.model_id, {"training": "Random Tree model"}
  )


loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
predictions = loaded_model.predict(X_test)
print(predictions)

import evidently
from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from evidently.ui.workspace import CloudWorkspace

ws = CloudWorkspace(token="dG9rbgFPI5BqKwdMOITPff6jrfMVNwIrZcAuBgdBWgrWucLpqQBQbyA6yRe//T+FClx8kAs7ugJQyKJlS1HPBxGy/IK/fE9+iTy/QbHGFdu9Pk9Q3AxTsEzcQZQ59oGAhPQLSlgzhsyB5vXtW0Fi0w5Y4HuyYAJlIanv", url="https://app.evidently.cloud")

project = ws.create_project(name="UsedCarPricePrediction", org_id='0198e453-938e-7b76-a3a2-6b50e5161c88')
project.description = "Learning Data Drift monitoring"
project.save()


import pandas as pd

constant_cols_train = [col for col in X_train.columns if X_train[col].std() == 0]
constant_cols_val = [col for col in X_val.columns if X_val[col].std() == 0]


cols_to_drop = list(set(constant_cols_train) | set(constant_cols_val))


X_train_clean = X_train.drop(columns=cols_to_drop)
X_val_clean = X_val.drop(columns=cols_to_drop)


from evidently import Dataset, Report
from evidently.presets import DataDriftPreset

eval_data_1 = Dataset.from_pandas(X_train_clean)
eval_data_2 = Dataset.from_pandas(X_val_clean)

report = Report([DataDriftPreset()])
my_eval = report.run(eval_data_1, eval_data_2)

my_eval.save_html('drift_report.html')

ws.add_run(project.id, my_eval, include_data=False)


