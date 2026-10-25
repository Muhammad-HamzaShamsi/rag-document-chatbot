# MLOps Notes

MLOps applies DevOps practices to machine learning systems. The goal is to make model
training, deployment and monitoring reproducible, automated and observable.

## Experiment tracking
MLflow records parameters, metrics and artifacts for every training run, which makes it
easy to compare models and reproduce the best one.

## Model serving
Models are commonly served behind a REST API (for example FastAPI) and packaged as Docker
images so the same artifact runs in development, staging and production.

## Monitoring
Production models degrade when input data drifts away from the training distribution.
Drift is detected with statistical tests such as Kolmogorov–Smirnov or the Population
Stability Index (PSI). Alerts fire when drift crosses a threshold, triggering retraining.

## Infrastructure
Kubernetes schedules containers across a cluster of nodes and handles scaling, rolling
updates and self-healing. Terraform provisions the underlying cloud infrastructure as code.
