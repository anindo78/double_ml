import numpy as np
import scipy

def marginal_effects(generative_model, num_samples=100, columns=None, max_points=20, logit=True, seed=0):
    """ Helper function to compute the true marginal causal effects.
    """
    X = generative_model(num_samples)
    if columns is None:
        columns = X.columns
    ys = [[] for _ in columns]
    xs = [X[c].values for c in columns]
    xs = np.sort(xs, axis=1)
    xs = [xs[i] for i in range(len(xs))]
    for i,c in enumerate(columns):
        xs[i] = np.unique([np.nanpercentile(xs[i], v, interpolation='nearest') for v in np.linspace(0, 100, max_points)])
        for x in xs[i]:
            Xnew = generative_model(num_samples, fixed={c: x}, seed=seed)
            val = Xnew["Did renew"].mean()
            if logit:
                val = scipy.special.logit(val)
            ys[i].append(val)
        ys[i] = np.array(ys[i])
    ys = [ys[i] - ys[i].mean() for i in range(len(ys))]
    return list(zip(xs, ys))

