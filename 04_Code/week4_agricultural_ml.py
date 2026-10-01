import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA = "../03_Model_Data/agriculture_yield_ml_clean_features.csv"
df = pd.read_csv(DATA)

target = "yield_t_ha"
features = [
    "year_index","state","crop","annual_rainfall_mm","avg_temp_c","soil_n_mgkg",
    "soil_ph","irrigation_index","area_ha","rainfall_deviation_mm",
    "temp_stress_index","rainfall_irrigation_interaction","soil_n_ph_interaction",
    "previous_yield_t_ha"
]

train = df["year"] <= 2022
test = df["year"] >= 2023
X_train, X_test = df.loc[train, features], df.loc[test, features]
y_train, y_test = df.loc[train, target], df.loc[test, target]

num = [c for c in features if c not in ["state","crop"]]
cat = ["state","crop"]

prep_linear = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), num),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)
])

prep_tree = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)
])

models = {
    "Linear Regression": Pipeline([("prep", prep_linear), ("model", LinearRegression())]),
    "Decision Tree": Pipeline([("prep", prep_tree),
                               ("model", DecisionTreeRegressor(max_depth=5, random_state=42))]),
    "Random Forest": Pipeline([("prep", prep_tree),
                               ("model", RandomForestRegressor(n_estimators=250,
                                                               max_depth=8,
                                                               min_samples_leaf=2,
                                                               random_state=42,
                                                               n_jobs=-1))])
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(name)
    print("MAE :", mean_absolute_error(y_test, pred))
    print("RMSE:", mean_squared_error(y_test, pred) ** 0.5)
    print("R2  :", r2_score(y_test, pred))
    print("-" * 40)
