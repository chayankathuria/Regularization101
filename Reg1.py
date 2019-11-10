import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
%matplotlib inline

# set the seed for generating the random nos
np.random.seed(100)
x = np.linspace(1,2*np.pi,20)
y = np.cos(x)*np.log(x)+ 0.3*np.random.randn(20) # add normal-dist noise

plt.plot(x,y,'o')
plt.plot(x,np.cos(x)*np.log(x),'r-')
plt.legend(['Actual Data', 'True Function'])

# Reshaping x to make it a column vector
x = x.reshape(-1,1)
x.shape


# Making a pipeline to iterate all the regressions- Linear and Polynomial (using 2 and more features) and comparing the performance
from sklearn.pipeline import Pipeline

pipelines = []
pipelines.append(('Poly1', Pipeline([('Linear-Feat', PolynomialFeatures(1)),('LR', LinearRegression())])))
pipelines.append(('Poly2', Pipeline([('Poly2-Feat', PolynomialFeatures(2)),('PR-2', LinearRegression())])))
pipelines.append(('Poly3', Pipeline([('Poly3-Feat', PolynomialFeatures(3)),('PR-3', LinearRegression())])))
pipelines.append(('Poly4', Pipeline([('Poly4-Feat', PolynomialFeatures(4)),('PR-4', LinearRegression())])))
pipelines.append(('Poly6', Pipeline([('Poly6-Feat', PolynomialFeatures(6)),('PR-6', LinearRegression())])))
pipelines.append(('Poly10', Pipeline([('Poly10-Feat', PolynomialFeatures(10)),('PR-10', LinearRegression())])))
pipelines.append(('Poly15', Pipeline([('Poly15-Feat', PolynomialFeatures(15)),('PR-15', LinearRegression())])))
pipelines.append(('Poly20', Pipeline([('Poly20-Feat', PolynomialFeatures(20)),('PR-20', LinearRegression())])))
pipelines
