# Week 4 — Developing Basic Machine Learning Models for Agricultural Data

## Purpose
Complete end-to-end demonstration of a basic agricultural yield prediction workflow.

## Main model task
Predict crop yield (tonnes/hectare) using weather, soil, irrigation, crop, state, area and lagged-yield features.

## Dataset note
The included dataset is synthetic demonstration data. Public Government of India OGD sources are documented in `07_Research_Notes`.

## Run
1. Install Python packages: pandas, numpy, scikit-learn, matplotlib, python-docx.
2. Run `04_Code/week4_agricultural_ml.py`.
3. Review model metrics in `06_Model_Results`.
4. Review figures in `05_Visualizations`.

## Current holdout
Training: 2012–2022.
Testing: 2023–2025.
This avoids using future years to predict earlier years.

## Best holdout model in this run
Linear Regression

See `06_Model_Results/model_metrics.csv` for exact metrics.
