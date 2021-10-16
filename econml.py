from econml.dml import LinearDML
from sklearn.base import BaseEstimator, clone
import matplotlib.pyplot as plt

class RegressionWrapper(BaseEstimator):
    """ Turns a classifier into a 'regressor'.

    We use the regression formulation of double ML, so we need to approximate the classifer
    as a regression model. This treats the probabilities as just quantitative value targets
    for least squares regression, but it turns out to be a reasonable approximation.
    """
    def __init__(self, clf):
        self.clf = clf

    def fit(self, X, y, **kwargs):
        self.clf_ = clone(self.clf)
        self.clf_.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return self.clf_.predict_proba(X)[:, 1]

# Run Double ML, controlling for all the other features
def double_ml(y, causal_feature, control_features):
    """ Use doubleML from econML to estimate the slope of the causal effect of a feature.
    """
    xgb_model = xgboost.XGBClassifier(objective="binary:logistic", random_state=42)
    est = LinearDML(model_y=RegressionWrapper(xgb_model))
    est.fit(y, causal_feature, W=control_features)
    return est.effect_inference()

def plot_effect(effect, xs, true_ys, ylim=None):
    """ Plot a double ML effect estimate from econML as a line.

    Note that the effect estimate from double ML is an average effect *slope* not a full
    function. So we arbitrarily draw the slope of the line as passing through the origin.
    """
    plt.figure(figsize=(5, 3))

    pred_xs = [xs.min(), xs.max()]
    mid = (xs.min() + xs.max())/2
    pred_ys = [effect.pred[0]*(xs.min() - mid), effect.pred[0]*(xs.max() - mid)]

    plt.plot(xs, true_ys - true_ys[0], label='True causal effect', color="black", linewidth=3)
    point_pred = effect.point_estimate * pred_xs
    pred_stderr = effect.stderr * np.abs(pred_xs)
    plt.plot(pred_xs, point_pred - point_pred[0], label='Double ML slope', color=shap.plots.colors.blue_rgb, linewidth=3)
    # 99.9% CI
    plt.fill_between(pred_xs, point_pred - point_pred[0] - 3.291 * pred_stderr,
                     point_pred - point_pred[0] + 3.291 * pred_stderr, alpha=.2, color=shap.plots.colors.blue_rgb)
    plt.legend()
    plt.xlabel("Ad spend", fontsize=13)
    plt.ylabel("Zero centered effect")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.show()

# estimate the causal effect of Ad spend controlling for all the other features
causal_feature = "Ad spend"
control_features = [
    "Sales calls", "Interactions", "Economy", "Last upgrade", "Discount",
    "Monthly usage", "Bugs reported"
]
effect = double_ml(y, X[causal_feature], X.loc[:,control_features])

# plot the estimated slope against the true effect
xs, true_ys = marginal_effects(generator, 10000, X[["Ad spend"]], logit=False)[0]
plot_effect(effect, xs, true_ys, ylim=(-0.2, 0.2))