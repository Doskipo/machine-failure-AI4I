"""Train a predictive-maintenance classifier on AI4I 2020 (UCI id 601) and save model.joblib."""

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

CATEGORICAL = ["Type"]
NUMERIC = ["Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "Machine failure"

ai4i = fetch_ucirepo(id=601)
X_all, y_all = ai4i.data.features, ai4i.data.targets
print("Feature columns:", list(X_all.columns))
print("Target columns: ", list(y_all.columns))

# TWF/HDF/PWF/OSF/RNF are the failure *modes*: they're only known once the machine has failed
# (Machine failure = OR of them), so using them as inputs would leak the answer into the model.
X = X_all[FEATURES]
y = y_all[TARGET]
print(f"\nPositive rate ({TARGET}=1): {y.mean():.2%}  ({y.sum()} of {len(y)})")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="error"), CATEGORICAL),
    ("num", StandardScaler(), NUMERIC),
])
model = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nClassification report (test set):")
print(classification_report(y_test, y_pred, digits=3))
print("Confusion matrix [[TN FP] [FN TP]]:")
print(confusion_matrix(y_test, y_pred))

print("\nFeature ranges (full dataset) — use these to pick sensible API bounds:")
print(X[NUMERIC].agg(["min", "max"]).T)
print("Type categories:", sorted(X["Type"].unique()))

joblib.dump(model, "model.joblib")
print("\nSaved model.joblib")
