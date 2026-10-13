from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

from datasets import load_dataset


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR / "models" / "logistic_regression.joblib"
)

VECTORIZER_PATH = (
    BASE_DIR / "models" / "tfidf_vectorizer.joblib"
)

RESULTS_DIR = BASE_DIR / "reports" / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def load_test_data():

    dataset = load_dataset(
        "ImperialCollegeLondon/health_fact"
    )

    df = dataset["test"].to_pandas()

    df = df.dropna(
        subset=["claim", "label"]
    ).copy()

    # Map dataset integer labels (0, 1, 2, 3) to string names
    label_mapping = {
        0: "false",
        1: "mixture",
        2: "true",
        3: "unproven"
    }

    df["label"] = (
        pd.to_numeric(df["label"], errors="coerce")
        .map(label_mapping)
    )

    # Drop any unknown/unmapped labels (e.g. -1)
    df = df.dropna(subset=["label"])

    df["claim"] = df["claim"].astype(str).str.strip()

    valid_labels = {
        "false",
        "mixture",
        "true",
        "unproven"
    }

    df = df[
        df["label"].isin(valid_labels)
    ]

    df = df[
        df["claim"].str.len() > 0
    ]

    return df.reset_index(drop=True)


def calculate_metrics(y_true, y_pred):

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )

    return {
        "n": len(y_true),

        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "macro_precision": precision,

        "macro_recall": recall,

        "macro_f1": f1
    }


def main():

    model = joblib.load(
        MODEL_PATH
    )

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    df = load_test_data()

    X_test = df["claim"]
    y_test = df["label"]

    X_vectors = vectorizer.transform(
        X_test
    )

    predictions = model.predict(
        X_vectors
    )

    df["predicted"] = predictions

    df["word_count"] = (
        df["claim"]
        .str.split()
        .str.len()
    )

    # Define length groups using documented
    # word-count boundaries.

    df["length_group"] = pd.cut(
        df["word_count"],
        bins=[-1, 10, 30, float("inf")],
        labels=[
            "Short (0-10 words)",
            "Medium (11-30 words)",
            "Long (31+ words)"
        ]
    )

    length_results = []

    for group_name, group in df.groupby(
        "length_group",
        observed=True
    ):

        metrics = calculate_metrics(
            group["label"],
            group["predicted"]
        )

        metrics["length_group"] = str(
            group_name
        )

        length_results.append(
            metrics
        )

    length_df = pd.DataFrame(
        length_results
    )

    length_df.to_csv(
        RESULTS_DIR / "length_audit.csv",
        index=False
    )

    print("Claim-length audit:")
    print(length_df)

    # Optional topic audit.
    # Only use this if the dataset has a usable
    # subjects field.

    if "subjects" in df.columns:

        print("\nSubjects field is available.")

        print(
            "Inspect its structure and annotation "
            "quality before creating topic groups."
        )

        print(
            df["subjects"].head(10)
        )

    else:

        print(
            "\nNo subjects field found. "
            "A topic audit requires a documented "
            "topic-labelling method."
        )


if __name__ == "__main__":
    main()