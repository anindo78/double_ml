# double_ml
### Intro to Double ML

### Be careful when interpreting predictive models in search of causal insights.
<br>

*A joint article about causality and interpretable machine learning with Eleanor Dillon, Jacob LaRiviere, Scott Lundberg, Jonathan Roth, and Vasilis Syrgkanis from Microsoft.*
<br>
<br>


#### Original Paper by: 
###### Double/Debiased Machine Learning for Treatment and Causal Parameters
Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, James Robins (2017)

[Link to Paper](https://arxiv.org/abs/1608.00060)


#### Additional Resources:
1. [DoubleML in Python](http://aeturrell.com/2018/02/10/econometrics-in-python-partI-ML/)

---


Predictive models e.g. XGBoost coupled with ML Interpretability models e.g. SHAP are powerful. But they are only useful for:
1. Predictions
2. Relationship between inputs and outcomes
3. Diagnosis of potential problems

Predictive models should not be used for 'decision making'. Since predictive models are **not causal**!!
Predictive models implicitly assume that everyone will keep behaving the same way in the future, and therefore correlation patterns will stay constant. But they do **not model behavior**.

## Example: Subscriber Retention
#### Predicts whether a customer will renew their product subscription

Let us assume there are the following drivers we have already found:
1. customer discount
2. ad spending
3. customer’s monthly usage
4. last upgrade
5. bugs reported by a customer
6. interactions with a customer
7. sales calls with a customer
8. macroeconomic activity.

