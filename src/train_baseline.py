from datasets import load_dataset
import pandas as pd

dataset = load_dataset("ImperialCollegeLondon/health_fact")

train_df = dataset["train"].to_pandas()
val_df = dataset["validation"].to_pandas()
test_df = dataset["test"].to_pandas()

print("Training data:", train_df.shape)
print("Validation data:", val_df.shape)
print("Test data:", test_df.shape)
# Data Cleaning 
def clean_dataset(df):
    df = df.copy()
    df = df.dropna(subset=["claim", "label"])

    # The health_fact dataset uses integer labels: 
    # 0: false, 1: mixture, 2: true, 3: unproven, -1: unknown
    label_mapping = {0: "false", 1: "mixture", 2: "true", 3: "unproven"}
    
    # Safely convert to numeric and map to string labels
    df["label"] = pd.to_numeric(df["label"], errors="coerce").map(label_mapping)
    
    # Drop rows that had unknown labels (like -1) which map to NaN
    df = df.dropna(subset=["label"])

    df["claim"] = df["claim"].astype(str).str.strip()
    df = df[df["claim"].str.len() > 0]

    return df


train_df = clean_dataset(train_df)
val_df = clean_dataset(val_df)
test_df = clean_dataset(test_df)

print("Cleaned training data:", len(train_df))
print("Cleaned validation data:", len(val_df))
print("Cleaned test data:", len(test_df))
# Prepare Inputs and Lables 
X_train = train_df["claim"]
y_train = train_df["label"]

X_val = val_df["claim"]
y_val = val_df["label"]

X_test = test_df["claim"]
y_test = test_df["label"]
# TF-IDF Vectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    max_features=50000,
    min_df=2,
    sublinear_tf=True
)
#Convert the text into numerical features
X_train_tfidf = tfidf.fit_transform(X_train)

X_val_tfidf = tfidf.transform(X_val)

X_test_tfidf = tfidf.transform(X_test)

print("Training matrix:", X_train_tfidf.shape)
print("Validation matrix:", X_val_tfidf.shape)
print("Test matrix:", X_test_tfidf.shape)
#Train Logistic Regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("Model training completed")
# Make predictions on the validation set
val_predictions = model.predict(X_val_tfidf)

print(val_predictions[:10])
#Evaluate the validation predictions
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("Validation accuracy:")
print(accuracy_score(y_val, val_predictions))

print("\nClassification report:")
print(classification_report(y_val, val_predictions, zero_division=0))

print("\nConfusion matrix:")
print(confusion_matrix(
    y_val,
    val_predictions,
    labels=["false", "mixture", "true", "unproven"]
))
#Calculate macro-F1
from sklearn.metrics import f1_score

macro_f1 = f1_score(
    y_val,
    val_predictions,
    average="macro"
)

print("Validation macro-F1:", macro_f1)
#Saving the baseline model
from pathlib import Path
import joblib

Path("models").mkdir(exist_ok=True)

joblib.dump(
    model,
    "models/logistic_regression.joblib"
)

joblib.dump(
    tfidf,
    "models/tfidf_vectorizer.joblib"
)

print("Model files saved")
# Plot confusion Matrix 
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

labels = ["false", "mixture", "true", "unproven"]

ConfusionMatrixDisplay.from_predictions(
    y_val,
    val_predictions,
    labels=labels,
    xticks_rotation=45
)

plt.title("Validation Confusion Matrix")
plt.tight_layout()
plt.savefig(
    "reports/figures/confusion_matrix_validation.png",
    dpi=300
)

plt.show()
# Examine misclassified examples
results_df = pd.DataFrame({
    "claim": X_val,
    "actual": y_val,
    "predicted": val_predictions
})

errors_df = results_df[
    results_df["actual"] != results_df["predicted"]
]

print(errors_df.head(20))