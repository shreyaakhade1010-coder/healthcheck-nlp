from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR / "models" / "logistic_regression.joblib"
)

VECTORIZER_PATH = (
    BASE_DIR / "models" / "tfidf_vectorizer.joblib"
)


def test_model_files_exist():

    assert MODEL_PATH.exists()
    assert VECTORIZER_PATH.exists()


def test_model_prediction():

    model = joblib.load(
        MODEL_PATH
    )

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    claim = (
        "A clinical study investigated a new treatment."
    )

    vector = vectorizer.transform(
        [claim]
    )

    prediction = model.predict(
        vector
    )[0]

    valid_labels = {
        "false",
        "mixture",
        "true",
        "unproven"
    }

    assert prediction in valid_labels


def test_prediction_probabilities():

    model = joblib.load(
        MODEL_PATH
    )

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    claim = (
        "Researchers investigated a medical treatment."
    )

    vector = vectorizer.transform(
        [claim]
    )

    probabilities = model.predict_proba(
        vector
    )

    assert probabilities.shape == (1, 4)

    assert abs(
        probabilities.sum() - 1.0
    ) < 1e-5