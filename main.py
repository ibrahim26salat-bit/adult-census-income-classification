import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

import os

# Create images folder
os.makedirs("images", exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("dataset/adult.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# -----------------------------
# Data Cleaning
# -----------------------------
df.replace("?", np.nan, inplace=True)

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# Remove duplicates
df.drop_duplicates(inplace=True)

# -----------------------------
# Feature Engineering
# -----------------------------
label = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = label.fit_transform(df[col])

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True)
}

results = []

# -----------------------------
# Training and Evaluation
# -----------------------------
for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:,1]
    else:
        y_prob = model.decision_function(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    roc = roc_auc_score(y_test, y_prob)

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1,
        roc
    ])

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(name)
    plt.savefig(f"images/{name.replace(' ','_')}_CM.png")
    plt.close()

# -----------------------------
# Results Table
# -----------------------------
results_df = pd.DataFrame(
    results,
    columns=[
        "Algorithm",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]
)

print("\nPerformance Comparison")
print(results_df)

# -----------------------------
# Accuracy Comparison Graph
# -----------------------------
plt.figure(figsize=(8,5))
plt.bar(results_df["Algorithm"], results_df["Accuracy"])
plt.xticks(rotation=20)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.tight_layout()
plt.savefig("images/accuracy_comparison.png")
plt.close()

# -----------------------------
# Save Results
# -----------------------------
results_df.to_csv("model_results.csv", index=False)

print("\nProject Completed Successfully.")
