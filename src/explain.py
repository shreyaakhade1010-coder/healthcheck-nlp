from pathlib import Path

import joblib
import numpy as np

from lime.lime_text import LimeTextExplainer


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR / "models" / "logistic_regression.joblib"
)

VECTORIZER_PATH = (
    BASE_DIR / "models" / "tfidf_vectorizer.joblib"
)


def load_baseline():
    """
    Load the trained Logistic Regression model
    and the matching TF-IDF vectoriser.
    """

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


def create_explainer(model, vectorizer):
    """
    Create a LIME text explainer.
    """

    class_names = [
        str(label)
        for label in model.classes_
    ]

    explainer = LimeTextExplainer(
        class_names=class_names,
        random_state=42
    )

    def predict_proba_for_text(texts):
        vectors = vectorizer.transform(texts)

        probabilities = model.predict_proba(
            vectors
        )

        return probabilities

    return explainer, predict_proba_for_text


def explain_claim(claim, num_features=10):
    """
    Explain the baseline model's prediction for a claim.

    Returns a list of influential features and weights.
    """

    claim = claim.strip()

    if not claim:
        raise ValueError(
            "Claim cannot be empty."
        )

    model, vectorizer = load_baseline()

    explainer, predict_function = create_explainer(
        model,
        vectorizer
    )

    explanation = explainer.explain_instance(
        claim,
        predict_function,
        num_features=num_features,
        num_samples=1000,
        labels=(
            int(
                np.argmax(
                    predict_function([claim])[0]
                )
            ),
        )
    )

    predicted_class_index = int(
        np.argmax(
            predict_function([claim])[0]
        )
    )

    feature_weights = explanation.as_list(
        label=predicted_class_index
    )

    return {
        "predicted_label": str(
            model.classes_[predicted_class_index]
        ),
        "features": [
            {
                "word": word,
                "weight": float(weight)
            }
            for word, weight in feature_weights
        ]
    }


if __name__ == "__main__":

    sample_claim = (
        "A new treatment has been proven to cure "
        "every form of cancer."
    )

    result = explain_claim(sample_claim)

    print("Predicted label:")
    print(result["predicted_label"])

    print("\nInfluential features:")

    for item in result["features"]:
        print(
            item["word"],
            round(item["weight"], 4)
        )