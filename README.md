# double_ml
### Intro to Double ML

Predictive models e.g. XGBoost coupled with ML Interpretability models e.g. SHAP are powerful. But they are only useful for:
1. Predictions
2. Relationship between inputs and outcomes
3. Diagnosis of potential problems

Predictive models should not be used for 'decision making'. Since predictive models are **not causal**!!
Predictive models implicitly assume that everyone will keep behaving the same way in the future, and therefore correlation patterns will stay constant. But they do **not model behavior**.

## Example: Subscriber Retention
#### Predicts whether a customer will renew their product subscription

Let us assume there are the following drivers we have already found:
(1) customer discount
(2) ad spending
(3) customer’s monthly usage
(4) last upgrade
(5) bugs reported by a customer
(6) interactions with a customer
(7) sales calls with a customer
(8) macroeconomic activity.

