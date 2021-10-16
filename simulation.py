
Markov Wanderer A blog on economics, science, coding and data. Views are my own.
Econometrics in Python part I - Double machine learning
10 Feb 2018
The idea is that this will be the first in a series of posts covering econometrics in Python.

At a conference a couple of years ago, I saw Victor Chernozhukov present his paper on Double/Debiased Machine Learning for Treatment and Causal Parameters. It really stuck with me because of the way it fruitfully combines econometrics and machine learning. Machine learning is obsessed with prediction, and is getting very good at it. Econometrics is obsessed with causality and identification, and pretty good at it - especially in ‘messy’ real-world situations. Combining the two promises to provide powerful new ways to understand causal relationships.

So, in brief, what does ‘double’ machine learning do? It’s one way to bring the power of machine learning for prediction on non-linear problems into an econometric context in which the asymptotic properties of the estimates of the parameters of interest are known to behave well. The problem is that just applying machine learning to predict outcomes (Y) from a treatment or variable (D) in the presence of many controls (X) will lead to biased estimates of the model parameter (θ). The double machine learning method of Chernozhukov et al. delivers point estimators that have a N−−√ rate of convergence for N observations and are approximately unbiased and normally distributed.

The clearest example, which I reproduce here from the paper, is of partially linear regression. They take it themselves from Robinson (1988). The model is

Y=D⋅θ+g(X)+U,E[U|X,D]=0D=m(X)+V,E[V|X]=0
with X=(X1,X2,…,Xp) a vector of controls. Here η=(m,g) can be non-linear.

The naïve machine learning approach would be to estimate D⋅θ^+g^(X) using one of the standard algorithms (random forest, support vector regression, etc). The authors of the paper show that doing this means that θ^ effectively has a slower than root N rate of convergence due to the bias in estimating g^.

They suggest overcoming this bias using orthogonalisation and splitting the sample. They obtain V^=D−m^(X) using machine learning on an auxiliary sample; finding the mean of D given X. With the remaining observations, they define an estimator for θ, θˇ, which is a function of V^, D, X, and an estimate of g given by g^. As they say (with a slight change in notation),

By approximately orthogonalizing D with respect to X and approximately removing the direct effect of confounding by subtracting an estimate of g^, θˇ removes the effect of regularization bias … The formulation of θˇ also provides direct links to both the classical econometric literature, as the estimator can clearly be interpreted as a linear instrumental variable (IV) estimator, …

The double comes from estimating V^ in the auxiliary problem, as well as g^, before calculating the estimator θˇ. In their paper, Chernozhukov et al. also discuss estimating average treatment effects, local average treatment effects, and average treatment effects for the treated using a more general formulation where g is a function of both X and D. More on the technical details and other applications can be found in the paper; here we’ll look at an example estimation in the context of a model

Double machine learning in practice
So how does it work in practice? With the sample split into two sets of size n=N/2 indexed by i∈I and i∈IC, there are four steps,

Estimate V^=D−m^(X) using IC
Estimate Y=g^(X)+u^ using IC
Estimate θˇ(IC,I)=(1n∑i∈IV^iDi)−11n∑i∈IV^i(Yi−g^(Xi))
Construct the efficient, cross-fitting estimate: θˇcf=12[θˇ(IC,I)+θˇ(I,IC)]
Simulated example
This example was inspired by this great post by Gabriel Vasconcelos. To make it more exciting, I’ll use a slightly different functional form with g as sine squared and m as the wrapped Cauchy distribution:

g(x)=sin2(x)m(x;ν,γ)=12πsinh(γ)cosh(γ)−cos(x−ν)
Let’s keep it simple and set ν=0 and γ=1. The wrapped Cauchy looks like this:

The wrapped Cauchy distribution

Our model is

yi=diθ+g(x′i⋅b)+ui,di=m(x′i⋅b)+vi
xi has length K=10 and will be generated from a multivariate normal distribution, the true value of the causal parameter will be θ=0.5, and bk=1/k. The errors will be

ui,vi∼N(0,1)
and I’m going to use the scikit learn implementation of the random forest regressor to do the machine learning.

Note that I’m using a scalar D in the example below but, in the original paper, it’s a binary treatment - thanks to Kyle Carlson for pointing out that this could cause some confusion!

The code, using Python 3, is

import numpy as np
from sklearn.datasets import make_spd_matrix
import math
import statsmodels.api as sm # for OLS from sklearn.ensemble import RandomForestRegressor # Our ML algorithm # Set up the environment randomseednumber = 11022018
np.random.seed(randomseednumber)
N = 500 # No. obs k=10 # = No. variables in x_i theta=0.5 # Structural parameter b= [1/k for k in range(1,11)] # x weights sigma = make_spd_matrix(k,randomseednumber) # # NUmber of simulations MC_no = 500
def g(x):
    return np.power(np.sin(x),2)
def m(x,nu=0.,gamma=1.):
    return 0.5/math.pi*(np.sinh(gamma))/(np.cosh(gamma)-np.cos(x-nu))
# Array of estimated thetas to store results theta_est = np.zeros(shape=[MC_no,3])

for i in range(MC_no):
    # Generate data: no. obs x no. variables in x_i     X = np.random.multivariate_normal(np.ones(k),sigma,size=[N,])
    G = g(np.dot(X,b))
    M = m(np.dot(X,b))
    D = M+np.random.standard_normal(size=[500,])
    Y = np.dot(theta,D)+G+np.random.standard_normal(size=[500,])
    #     # Now run the different methods     #     # OLS --------------------------------------------------     OLS = sm.OLS(Y,D)
    results = OLS.fit()
    theta_est[i][0] = results.params[0]

    # Naive double machine Learning ------------------------     naiveDMLg =RandomForestRegressor(max_depth=2)
    # Compute ghat     naiveDMLg.fit(X,Y)
    Ghat = naiveDMLg.predict(X)
    naiveDMLm =RandomForestRegressor(max_depth=2)
    naiveDMLm.fit(X,D)
    Mhat = naiveDMLm.predict(X)
    # vhat as residual     Vhat = D-Mhat
    theta_est[i][1] = np.mean(np.dot(Vhat,Y-Ghat))/np.mean(np.dot(Vhat,D))

    # Cross-fitting DML -----------------------------------     # Split the sample     I = np.random.choice(N,np.int(N/2),replace=False)
    I_C = [x for x in np.arange(N) if x not in I]
    # Ghat for both     Ghat_1 = RandomForestRegressor(max_depth=2).fit(X[I],Y[I]).predict(X[I_C])
    Ghat_2 = RandomForestRegressor(max_depth=2).fit(X[I_C],Y[I_C]).predict(X[I])
    # Mhat and vhat for both     Mhat_1 = RandomForestRegressor(max_depth=2).fit(X[I],D[I]).predict(X[I_C])
    Mhat_2 = RandomForestRegressor(max_depth=2).fit(X[I_C],D[I_C]).predict(X[I])
    Vhat_1 = D[I_C]-Mhat_1
    Vhat_2 = D[I] - Mhat_2
    theta_1 = np.mean(np.dot(Vhat_1,(Y[I_C]-Ghat_1)))/np.mean(np.dot(Vhat_1,D[I_C]))
    theta_2 = np.mean(np.dot(Vhat_2,(Y[I]-Ghat_2)))/np.mean(np.dot(Vhat_2,D[I]))
    theta_est[i][2] = 0.5*(theta_1+theta_2)

Below is a plot of the kernel density estimates of θ using seaborn. The peak of the distributions for OLS and double ML without cross-fitting are off the true value, but the cross-fitted double ML procedure gets much closer.

The estimates of θ

So there it is: double machine learning is a useful technique at the intersection of machine learning and econometrics which can produce approximately unbiased and normally distributed point estimates in semi-parametric settings.

Related Posts
Three ways to blog with code 12 Feb 2021
10 less well-known Python packages 24 Oct 2020
Get organised 26 Jun 2019
