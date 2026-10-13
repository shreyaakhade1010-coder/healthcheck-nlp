# HealthCheck: Final Experimental Results

## 1. Project Objective

The objective of HealthCheck is to investigate the use of
natural language processing for classifying health-related
claims into False, Mixture, True, and Unproven categories.

The project also explores local explainability and evaluates
model performance using quantitative metrics.

## 2. Dataset

Dataset name:
PUBHEALTH / Health Fact

Source:
https://huggingface.co/datasets/ImperialCollegeLondon/health_fact

Dataset version or retrieval date:

Training records after preprocessing:

Validation records after preprocessing:

Test records after preprocessing:

Label distribution:

Preprocessing decisions:

## 3. Experimental Setup

Python version:

Scikit-learn version:

Transformers version:

Random seed:

Maximum sequence length:

Hardware used:

## 4. Baseline Model

Model:
TF-IDF + Logistic Regression

TF-IDF configuration:

Logistic Regression configuration:

Training procedure:

## 5. Transformer Model

Model:
DistilBERT

Checkpoint:

Tokenizer:

Maximum sequence length:

Learning rate:

Batch size:

Number of epochs:

Best validation checkpoint selection:

## 6. Evaluation Metrics

Report the following for each model:

- Accuracy
- Macro precision
- Macro recall
- Macro-F1
- Per-class precision
- Per-class recall
- Per-class F1-score

Insert actual measured results here.

## 7. Model Comparison

Compare both models on the same held-out test split.

Discuss:
- Which model achieved the higher macro-F1?
- Which labels were most difficult to classify?
- Did one model perform better on minority classes?
- What were the computational costs?
- Which model is more practical for deployment?

## 8. Confusion Matrix

Include the generated confusion matrix and explain
the most frequent classification errors.

## 9. Error Analysis

Inspect incorrectly classified claims.

Discuss possible confusion between:
- Mixture and Unproven
- True and Mixture
- False and Unproven

Use actual examples from the test set, while respecting
dataset licensing and privacy considerations.

## 10. Explainability

Describe the use of LIME.

Include an example explanation and discuss why local feature
weights do not establish medical truth or causation.

## 11. Audit

Report performance by claim length.

If reliable topic labels are available, conduct a topic audit.
Otherwise, explain why a topic-based audit could not be
performed reliably.

Report sample counts for every subgroup.

## 12. Application

Describe the Streamlit interface.

Document the input, prediction, class probabilities,
LIME explanation, and safety disclaimer.

## 13. Limitations

Discuss dataset coverage, label quality, domain shift,
class imbalance, model calibration, explainability limitations,
and the lack of independent evidence retrieval.

## 14. Conclusion

Summarise what was learned from the experiments.

Do not claim that HealthCheck independently verifies
medical facts unless a separate evidence-verification
