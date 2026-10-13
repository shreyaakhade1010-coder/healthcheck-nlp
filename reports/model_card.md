# HealthCheck Model Card

## 1. Model Overview

HealthCheck is an educational NLP research prototype that
classifies health claims into four categories: True, False,
Mixture, and Unproven.

## 2. Intended Use

The model is intended for academic experimentation and
demonstration of NLP-based health claim classification.

## 3. Out-of-Scope Uses

The model must not be used for diagnosis, treatment decisions,
clinical decision-making, or as an authoritative source of
medical fact-checking.

## 4. Dataset

Dataset: PUBHEALTH

Source:
https://huggingface.co/datasets/ImperialCollegeLondon/health_fact

Input: Claim text.

Target: Dataset-provided classification label.

## 5. Models

Baseline: TF-IDF + Logistic Regression.

Transformer: DistilBERT, fine-tuned for four-class classification.

## 6. Evaluation

Report accuracy, macro precision, macro recall, macro-F1,
per-class metrics, and confusion matrices.

Insert actual experimental results.

## 7. Explainability

LIME is used to estimate which input terms influence individual
baseline predictions.

LIME explanations do not establish medical truth or causality.

## 8. Limitations

The model learns patterns from the labelled dataset and does
not independently verify claims against current medical evidence.

Predictions may be inaccurate, biased, or poorly calibrated.

Performance on new topics, languages, or claim formats may differ
from performance on the evaluation dataset.

## 9. Medical Safety

The application is not medical advice, diagnosis, or treatment
guidance. Users should consult qualified healthcare professionals
and reliable medical sources.

## 10. Reproducibility

Document Python and library versions, dataset splits, random seeds,
model configuration, and evaluation procedures.