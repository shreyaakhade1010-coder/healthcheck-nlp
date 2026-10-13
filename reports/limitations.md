# HealthCheck Limitations

## Dataset Limitations

The dataset may not cover all health topics, populations,
languages, or forms of misinformation.

Its labels reflect the dataset's original annotation process.

## Model Limitations

The model learns statistical patterns from labelled claims.
It does not independently retrieve and verify current medical
evidence.

## Explainability Limitations

LIME produces local approximations of model behaviour.
Its highlighted words should not be treated as medical evidence.

## Generalisation

Performance on previously unseen claims may differ from the
reported evaluation performance.

## Medical Safety

The application must not be used to make healthcare decisions.

## Future Work

Potential improvements include evidence retrieval, citation-based
verification, better uncertainty estimation, and additional
evaluation datasets.