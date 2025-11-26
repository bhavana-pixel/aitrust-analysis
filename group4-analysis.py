# Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Read excel file from group 4 survey results.
data = pd.read_excel("group 4.xlsx");

# Metadata column names to be removed.
metadata_patterns = [
    "StartDate", "EndDate", "Status", "IPAddress", "Progress",
    "Duration", "Finished", "RecordedDate", "Recipient",
    "UserLanguage", "LocationLatitude", "LocationLongitude",
    "ResponseId", "DistributionChannel",
    "_Click", "_First Click", "_Last Click",
    "_Click Count", "_Page Submit", "_Submit"
]

# Array of columns to drop.
cols_to_drop = [col for col in data.columns if any(p in col for p in metadata_patterns)]

# Removing metadata columns.
data = data.drop(columns=cols_to_drop)

# Likert Scale mappings to numbers.
mapping = {
    "Not accurate at all": 1,
    "Not at all" : 1,
    "Somewhat":2,
    "Partially":3, 
    "Completely":4,
    "Completely accurate":4,
    "Yes, completely": 4,
    "Yes, absolutely":4,
    "No":1,
    "Don't Trust AI" : 0,
    "Trust AI":1
}

# Apply mapping to survey answers.
data = data.applymap(lambda x: mapping.get(x,x))

# Locating survey questions.
question_cols = data.loc[:, "QID2" : "QID418"].columns.tolist()

# Question IDs that measure individual trust level.
trust_qs = [
    "Q1D2", "QID10", "QID12", "Q1D13", "QID178", "QID198", "QID203", "QID208",
    "QID22", "QID224", "QID254", "QID284", "QID314", "QID32", "QID324",
    "QID354", "QID384", "QID414"
]

# Isolating questions that match array above.
trust_qs = [q for q in trust_qs if q in data.columns]

# Isolating predictor questions.
predictor = [c for c in question_cols if c not in trust_qs]

# Converting data to numerical values.
data[predictor] = data[predictor].apply(pd.to_numeric, errors="coerce")
data[trust_qs] = data[trust_qs].apply(pd.to_numeric, errors="coerce")

# Filling in NaN values.
data[predictor] = data[predictor].fillna(data[predictor].mean())
data[trust_qs] = data[trust_qs].fillna(data[trust_qs].mean())

# Calculating and printing mean trust score.
data["trust_score"] = data[trust_qs].mean(axis=1)
print("Average Trust Score:" , data["trust_score"].mean())

# Isolating demographic questions for future analysis.
demographics = ["QID61", "QID62", "QID67"]
le = LabelEncoder()
for column in demographics:
    data[column] = le.fit_transform(data[column].astype(str))

# Setting predictor and output values.
feature = data[predictor]
score = data["trust_score"]

# Training test data using Random Forest.
feature_train, feature_test, score_train, score_test = train_test_split(feature, score, test_size = 0.2, random_state = 42)
rf = RandomForestRegressor(n_estimators=300, random_state = 42)
rf.fit(feature_train, score_train)
score_prediction = rf.predict(feature_test)

# Printing training outputs.
print("R2 score:", r2_score(score_test, score_prediction))
print("MSE:", mean_squared_error(score_test, score_prediction))
 
 # Finding and printing 20 most significant questions leading to results.
importances = rf.feature_importances_
importance_df = pd.DataFrame({
    "feature": feature.columns,
    "importance":importances
}).sort_values(by="importance", ascending = False)
print("\nTop 20 most important predictors:")

# Average Trust Score: 0.625
# R2 Score: 0.275
# MSE: 0.041
