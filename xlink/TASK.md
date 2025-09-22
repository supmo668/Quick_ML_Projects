# Customer Churn Prediction

## Overview

You’re provided with a dataset containing customer and service-related attributes. Your goal is to build a model that predicts whether a customer is likely to churn. The dataset reflects real‑world nuances; focus on pragmatic choices, clear reasoning, and how you’d operate this in production.

- Emphasis: clarity, trade‑offs, and rationale over raw model performance

## What we evaluate

- Ability to work with data and design usable features
- Building and evaluating a pragmatic baseline model
- Deployment and monitoring considerations for production use
- Clear, concise communication of decisions and trade‑offs

## Provided files

- `customer_churn.csv` – Tabular dataset for modeling
- `data_dictionary.md` – Column descriptions and data notes
- `Instructions.pdf` – Original instructions (reference)
- This `TASK.md` – Reformatted task requirements

---

## Tasks

### Part A — Data exploration and feature design

Explore the dataset and decide how to prepare it for modeling.

Expected outputs:

- [ ] Brief EDA: data quality (missing, outliers), target distribution, key correlations/relationships
- [ ] Feature preparation decisions (encoding, imputation, scaling, text/date handling)
- [ ] 1–3 sentence rationale per notable decision (e.g., why one-hot vs. target encoding)

### Part B — Modeling

Build a baseline model to predict churn probability.

Requirements:

- [ ] Choose a simple, dependable approach (e.g., logistic regression, gradient-boosted trees)
- [ ] Report a few relevant metrics (e.g., ROC‑AUC, PR‑AUC, accuracy, recall@threshold, calibration)
- [ ] Explain why those metrics are appropriate and how they reflect model quality
- [ ] Describe at least one modeling decision and its trade‑offs (e.g., threshold choice, regularization, class weighting)
- [ ] Include a brief segment analysis (e.g., performance by tenure, plan type, or geography if available)

Notes:

- You do not need to optimize for the absolute best score; justify pragmatic choices.
- If class imbalance exists, consider strategies such as class weights, thresholding, or calibration.

### Part C — Deployment and monitoring

Write a short answer covering:

- [ ] How you would deploy this model in a real‑world system (batch vs. real‑time, serving stack)
- [ ] How you would ensure training/serving consistency (feature parity, preprocessing, versioning)
- [ ] What to monitor in production (data drift, model performance, and business KPIs)
- [ ] How you would decide when to retrain (trigger rules, cadence, and safeguards)

### Part D — Serving

Build a minimal service that exposes your trained model.

Requirements:

- [ ] A simple API endpoint (e.g., HTTP POST `/predict`) that accepts a JSON payload matching your model’s features and returns churn probability and a decision (based on your chosen threshold)
- [ ] Include `requirements.txt`
- [ ] Document any design choices (input schema, threshold, error handling)

Suggested (optional) details:

- Example request/response schema and a small input validator
- Simple health/readiness endpoints

---

## Deliverables

Provide one of the following for the full workflow (EDA → features → model → evaluation → segment analysis):

- `notebook.ipynb` (interactive, narrative) or `script.py` (scripted workflow)

Additionally include:

- `REPORT.md` – concise summary covering:
  - Data preparation and feature choices
  - Modeling approach and metrics (with brief rationale)
  - Segment analysis insights
  - Deployment and monitoring plan
- `requirements.txt` – for the serving component (and the notebook/script if needed)

---

## Guidance and constraints

- Keep the solution focused and readable; comment key decisions inline.
- Prefer reproducibility (seed setting, pinned packages, clear instructions).
- If you make assumptions due to missing context, state them briefly and proceed.

## Submission

Submit your code and artifacts as instructed by the hiring team (e.g., GitHub repo or archive). Ensure all paths work relative to the project root, and that the serving component runs locally with your `requirements.txt`.
