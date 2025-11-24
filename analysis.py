# Imports
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Read excel file.
data = pd.read_excel("group 1.xlsx");

data = data.iloc[1:].reset_index(drop=True)


question_cols = data.loc[:, "QID2" : "QID418"].columns
data[question_cols] = data[question_cols].apply(pd.to_numeric, errors = "coerce")

data["average_score"] = data[question_cols].mean(axis=1)

demographics = ["QID61", "QID62", "QID67"]

data_clean = data.dropna(subset = demographics + ["average_score"])

le = LabelEncoder()

for column in demographics:
    data_clean[column] = le.fit_transform(data_clean[column].astype(str))

feature = data_clean[demographics]
score = data_clean["average_score"]

feature_train, feature_test, score_train, score_test = train_test_split(feature, score, test_size = 0.2, random_state = 42)

rf = RandomForestRegressor(n_estimators=300, random_state = 42)
rf.fit(feature_train, score_train)
score_prediction = rf.predict(feature_test)

print(data.head())
print("R2 score:", r2_score(score_test, score_prediction))
print("MSE:", mean_squared_error(score_test, score_prediction))
 


