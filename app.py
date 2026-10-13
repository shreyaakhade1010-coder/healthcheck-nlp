from pathlib import Path

import streamlit as st
import joblib
import numpy as np

from lime.lime_text import LimeTextExplainer


# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="HealthCheck",
    page_icon="🩺",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "models" / "logistic_regression.joblib"
)

VECTORIZER_PATH = (
    BASE_DIR / "models" / "tfidf_vectorizer.joblib"
)


# --------------------------------------------------
# 2. LOAD THE TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Baseline model file was not found."
        )

    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "TF-IDF vectoriser file was not found."
        )

    model = joblib.load(
        MODEL_PATH
    )

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    return model, vectorizer


# --------------------------------------------------
# 3. LOAD LIME EXPLAINER
# --------------------------------------------------

@st.cache_resource
def load_explainer(class_names):

    return LimeTextExplainer(
        class_names=class_names,
        random_state=42
    )


# --------------------------------------------------
# 4. PAGE TITLE AND DESCRIPTION
# --------------------------------------------------

st.title("🩺 HealthCheck")

st.subheader(
    "Explainable NLP-Based Health Claim Classification"
)

st.write(
    """
    Enter a health-related claim to see how a machine-learning
    model classifies it.

    HealthCheck is an educational NLP research prototype.
    It does not independently verify medical facts.
    """
)


# --------------------------------------------------
# 5. MEDICAL DISCLAIMER
# --------------------------------------------------

with st.expander(
    "Important medical disclaimer",
    expanded=True
):

    st.warning(
        """
        HealthCheck may produce incorrect or outdated predictions.

        Its output is based on patterns learned from a labelled
        dataset. It is not medical advice, diagnosis, or treatment
        guidance.

        The model does not independently verify claims against
        current medical research.

        Do not use this application to make healthcare decisions.
        Consult a qualified healthcare professional and reliable
        medical sources for health-related questions.
        """
    )


# --------------------------------------------------
# 6. USER INPUT
# --------------------------------------------------

claim = st.text_area(
    "Enter a health claim",
    placeholder=(
        "Example: A particular food can cure every "
        "form of cancer."
    ),
    height=150,
    max_chars=5000
)


# --------------------------------------------------
# 7. ANALYSIS BUTTON
# --------------------------------------------------

if st.button(
    "Analyse Claim",
    type="primary"
):

    if not claim.strip():

        st.warning(
            "Please enter a health claim."
        )

    else:

        try:

            with st.spinner(
                "Analysing the claim..."
            ):

                model, vectorizer = load_model()

                # Generate the input features.

                claim_vector = vectorizer.transform(
                    [claim.strip()]
                )

                # Predict the label.

                prediction = model.predict(
                    claim_vector
                )[0]

                # Generate class probabilities.

                probabilities = model.predict_proba(
                    claim_vector
                )[0]

                class_names = [
                    str(label)
                    for label in model.classes_
                ]

                # ----------------------------------
                # 8. DISPLAY PREDICTION
                # ----------------------------------

                st.divider()

                st.header(
                    "Classification Result"
                )

                st.metric(
                    "Predicted classification",
                    str(prediction).upper()
                )

                st.caption(
                    "This is a model prediction, not a verified "
                    "medical verdict."
                )

                # ----------------------------------
                # 9. DISPLAY CLASS PROBABILITIES
                # ----------------------------------

                st.subheader(
                    "Model-estimated class probabilities"
                )

                probability_data = {
                    "Class": class_names,
                    "Probability": probabilities
                }

                st.bar_chart(
                    {
                        "Probability": probabilities
                    },
                    x=None,
                    y="Probability"
                )

                for class_name, probability in zip(
                    class_names,
                    probabilities
                ):

                    st.write(
                        f"**{class_name.title()}**: "
                        f"{probability:.2%}"
                    )

                st.caption(
                    "These are model-estimated probabilities. "
                    "They are not guaranteed measures of correctness."
                )

                # ----------------------------------
                # 10. LIME EXPLANATION
                # ----------------------------------

                st.divider()

                st.subheader(
                    "Explanation"
                )

                st.write(
                    """
                    The following terms indicate features that
                    influenced the model's local prediction.

                    They do not independently establish whether
                    the health claim is medically correct.
                    """
                )

                explainer = load_explainer(
                    class_names
                )

                def predict_proba_for_text(texts):

                    vectors = vectorizer.transform(
                        texts
                    )

                    return model.predict_proba(
                        vectors
                    )

                predicted_index = int(
                    np.argmax(probabilities)
                )

                explanation = explainer.explain_instance(
                    claim.strip(),
                    predict_proba_for_text,
                    num_features=10,
                    num_samples=1000,
                    labels=(predicted_index,)
                )

                features = explanation.as_list(
                    label=predicted_index
                )

                if features:

                    explanation_df = {
                        "Term": [
                            item[0]
                            for item in features
                        ],

                        "Local explanation weight": [
                            item[1]
                            for item in features
                        ]
                    }

                    st.dataframe(
                        explanation_df,
                        use_container_width=True
                    )

                else:

                    st.info(
                        "No explanatory features were returned."
                    )

                st.caption(
                    "LIME approximates model behaviour around "
                    "the supplied text. Its explanations may vary "
                    "and should not be interpreted as medical evidence."
                )

        except FileNotFoundError as error:

            st.error(
                str(error)
            )

            st.info(
                "Train the baseline model and save the model "
                "and vectoriser files before running the app."
            )

        except Exception as error:

            st.error(
                "The claim could not be analysed."
            )

            st.exception(
                error
            )


# --------------------------------------------------
# 11. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    """
    HealthCheck | University NLP Research Project

    This prototype classifies health claims using a trained
    machine-learning model. It is not a medical fact-checking
    authority.
    """
)