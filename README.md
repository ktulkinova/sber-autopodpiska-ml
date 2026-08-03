
# SberAutopodpiska — Lead Conversion Prediction

Predicting whether a website visitor will submit a car subscription request, based on a single session. The goal: help the business identify high-intent traffic and allocate ad budget more effectively.

**Result: ROC-AUC 0.704** — the client's target was 0.65.

Capstone project for a Data Science course. Real-world task from SberAutopodpiska, a long-term car subscription service.

---

## Problem

Highly imbalanced data: conversion rate is only ~2.7%, class ratio ~36:1. ROC-AUC was chosen as the primary metric instead of accuracy, which would be meaningless at this imbalance.

## Data

Google Analytics exports: `ga_sessions.pkl` (sessions) and `ga_hits.pkl` (events).

Not included in the repository due to size (hundreds of MB) — [download the dataset here](LINK).

## Approach

I worked from simple to complex, comparing each step honestly:

| Model | Features | ROC-AUC |
|---|---|---|
| Logistic Regression (baseline) | base | 0.604 |
| CatBoost | base | 0.643 |
| CatBoost | + traffic source and geo | **0.704** |

**Key takeaway:** adding strong features (geography, traffic source) improved the score more than switching models. Data mattered more than the algorithm.

## What drives conversion

Traffic source — the ad campaign and channel — is the strongest predictor.

A notable finding: organic traffic converts nearly twice as well as paid (~4% vs ~2.2%), suggesting the business should invest in organic acquisition rather than ads alone.

![Feature importance](feature_importance.png)

## Pipeline

1. **Cleaning** — dropped empty and irrelevant columns; found hidden missing values (`(not set)` stored as a string rather than null); converted dates to proper types.
2. **Target construction** — built the label from the events table: a session counts as `1` if it contains at least one target action from the business glossary.
3. **EDA** — feature distributions, conversion rates by device and channel.
4. **Feature engineering** — created `is_organic` to separate organic from paid traffic.
5. **Modeling** — trained and compared models, analyzed feature importance.
6. **Deployment** — packaged the final model as a prediction service.

## Stack

Python · pandas · scikit-learn · CatBoost · matplotlib · seaborn

## Repository structure

| File | Contents |
|---|---|
| `ml.ipynb` | Main notebook: cleaning, EDA, training, conclusions |
| `predict.py` | Prediction service: takes session data → returns 0 / 1 |
| `catboost_model.cbm` | Trained model |
| `feature_importance.png` | Feature importance plot |

## How to run

```bash
pip install catboost pandas
python predict.py
```

`catboost_model.cbm` must be in the same directory as `predict.py`. A sample session is hardcoded in the script. Output: `Prediction (0/1): 0` or `1`.
