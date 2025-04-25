# Then in Python shell or script
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd

df = pd.read_csv('predictor/ml_model/cleaned_data.csv')  # your cleaned CSV
# Make sure df only includes 12 features + target (classification)

X = df[["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "hemo", "pcv", "htn", "dm"]]
X["htn"] = X["htn"].apply(lambda x: 1 if x == "yes" else 0)
X["dm"] = X["dm"].apply(lambda x: 1 if x == "yes" else 0)
y = df["classification"].apply(lambda x: 1 if x == "ckd" else 0)

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, 'predictor/ml_model/rf_ckd_model.pkl')