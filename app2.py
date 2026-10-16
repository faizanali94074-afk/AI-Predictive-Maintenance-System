
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

df = pd.read_csv("/content/ai4i2020.csv")

print("Dataset Shape:")
print(df.shape)

df.head()

print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print(df["Machine failure"].value_counts())

print(
    df["Machine failure"]
    .value_counts(normalize=True) * 100
)

plt.figure(figsize=(6,4))

sns.countplot(
    x="Machine failure",
    data=df
)

plt.title("Machine Failure Distribution")

plt.show()

plt.figure(figsize=(6,4))

sns.countplot(
    x="Type",
    data=df
)

plt.title("Machine Type Distribution")

plt.show()

failure_by_type = (
    df.groupby("Type")["Machine failure"]
    .mean() * 100
)

print(failure_by_type)

plt.figure(figsize=(10,8))

sns.heatmap(
    df.select_dtypes(include=['int64','float64']).corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()

encoder = LabelEncoder()

df["Type"] = encoder.fit_transform(
    df["Type"]
)

print(encoder.classes_)

df.drop(
    columns=[
        "UDI",
        "Product ID"
    ],
    inplace=True
)

X = df[
    [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]
]

y = df["Machine failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    "Accuracy:",
    round(accuracy*100,2),
    "%"
)

print(
    classification_report(
        y_test,
        y_pred
    )
)

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()

feature_importance = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }
)

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance)

plt.figure(figsize=(8,5))

sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance
)

plt.title("Feature Importance")

plt.show()

import joblib

joblib.dump(model, "predictive_maintenance_model2.pkl")

print("Model saved successfully!")

joblib.dump(encoder, "label_encoder2.pkl")

print("Encoder saved successfully!")

!ls

from google.colab import files

files.download("predictive_maintenance_model2.pkl")

files.download("label_encoder2.pkl")

# Train Final Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Save Final Files
joblib.dump(model, "predictive_maintenance_model2.pkl")
joblib.dump(encoder, "label_encoder2.pkl")
