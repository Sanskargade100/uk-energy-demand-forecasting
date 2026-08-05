# Models

Serialized model artifacts produced by `make train`. Contents are git-ignored
(only this README and `.gitkeep` are tracked).

Expected naming: `<model_name>__<train_end_date>.joblib` (or `.pt` for deep-learning
checkpoints), with a sibling `<...>.json` holding metadata: feature list, config hash,
training window, and validation scores.

The `energy_forecasting.models.registry` module resolves the latest artifact per model
so the API and dashboard load a consistent version.
