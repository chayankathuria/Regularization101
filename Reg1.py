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

# importing kfold,LOOCV for cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
import pandas as pd

scoring = 'neg_mean_squared_error'
results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=7, shuffle=True, random_state=5)
    cv_results = cross_val_score(model, x, y, cv=LeaveOneOut(), scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean()*100, cv_results.std()*100)
    print(msg)
    
n_splits=7
pd.set_option('precision',3) # to set the 3-digits for display

# To Display the result dataframe for all the CVs 0-7
results_df = pd.DataFrame(results, index=names, \
                          columns='CV1 CV2 CV3 CV4 CV5 CV6 CV7'.split())
results_df['CV Mean'] = results_df.iloc[:,0:n_splits].mean(axis=1)
results_df['CV Std Dev'] = results_df.iloc[:,0:n_splits].std(axis=1)

results_df.sort_values(by='CV Mean', ascending=False)

# Using Test Train Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, \
                                                    test_size=0.3, 
                                                    random_state=200)

scoring = 'neg_mean_squared_error'
results = []
names = []
for name, model in pipelines:
    model.fit(X_train,y_train)
    result = model.score(X_test,y_test)
    results.append(result)
    names.append(name)
    msg = "%s: %f" % (name, result)
    print(msg)

# Polynomial regression
PR = PolynomialFeatures(6)
X_poly = PR.fit_transform(x)
Poly_model = LinearRegression()
Poly_model.fit(X_poly,y)

plt.plot(X_train,y_train,'bs')
plt.plot(X_test,y_test,'ro')
plt.plot(x,Poly_model.predict(X_poly),'g-*')
plt.legend(['Train','Test Data', 'Poly4'])

for i in range(1,30):
    PR = PolynomialFeatures(i)
    X_poly = PR.fit_transform(x)
    Poly_model = LinearRegression(fit_intercept=True)
    Poly_model.fit(X_poly,y)
    print(i, Poly_model.score(X_poly,y))

# Plotting Linear regression models with polynomial features to visualize over fitting 

overall_acc = []
test_acc = []
colors = ['r','g','b','c','m']
plt.figure(figsize=(8,8))
plt.plot(x,y,'ro', label='data')

t = np.linspace(1,2*np.pi,100).reshape(-1,1)
plt.plot(t,np.cos(t)*np.log(t),'k--', lw=4, label='True')

for i in range(1,13):
    PR = PolynomialFeatures(i)
    X_poly = PR.fit_transform(x)
    Poly_model = LinearRegression(fit_intercept=True)
    Poly_model.fit(X_poly,y)
    overall_acc.append(Poly_model.score(X_poly,y))
    
    x_train, x_test, y_train, y_test = train_test_split(X_poly,y, \
                                                    test_size=0.3, 
                                                    random_state=200)
    test_acc.append(Poly_model.score(x_test,y_test))
#     print(i, Poly_model.score(X_poly,y), Poly_model.score(x_test,y_test) )
    
#     print(i, Poly_model.coef_)
#     print('')
    
    if i%3==0:
        plt.plot(t,Poly_model.predict(PR.fit_transform(t)), label=i) #, color=colors[i%5])

plt.legend()

# Changing Linear to Ridge to see the effect to Ridge on polynomial models
overall_acc = []
test_acc = []
colors = ['r','g','b','c','m']
plt.figure(figsize=(8,8))
plt.plot(x,y,'ro', label='data')

t = np.linspace(1,2*np.pi,100).reshape(-1,1)
plt.plot(t,np.cos(t)*np.log(t),'k--', lw=4, label='True')

for i in range(1,13):
    PR = PolynomialFeatures(i)
    X_poly = PR.fit_transform(x)
    Poly_model = Ridge(fit_intercept=True)
    Poly_model.fit(X_poly,y)
    overall_acc.append(Poly_model.score(X_poly,y))
    
    x_train, x_test, y_train, y_test = train_test_split(X_poly,y, \
                                                    test_size=0.3, 
                                                    random_state=200)
    test_acc.append(Poly_model.score(x_test,y_test))
#     print(i, Poly_model.score(X_poly,y), Poly_model.score(x_test,y_test) )
    
#     print(i, Poly_model.coef_)
#     print('')
    
    if i%3==0:
        plt.plot(t,Poly_model.predict(PR.fit_transform(t)), label=i) #, color=colors[i%5])

plt.legend()

# Changing Ridge to Lasso to see which one is more aggressive- clearly Lasso
overall_acc = []
test_acc = []
colors = ['r','g','b','c','m']
plt.figure(figsize=(8,8))
plt.plot(x,y,'ro', label='data')

t = np.linspace(1,2*np.pi,100).reshape(-1,1)
plt.plot(t,np.cos(t)*np.log(t),'k--', lw=4, label='True')

for i in range(1,13):
    PR = PolynomialFeatures(i)
    X_poly = PR.fit_transform(x)
    Poly_model = Lasso(fit_intercept=True)
    Poly_model.fit(X_poly,y)
    overall_acc.append(Poly_model.score(X_poly,y))
    
    x_train, x_test, y_train, y_test = train_test_split(X_poly,y, \
                                                    test_size=0.3, 
                                                    random_state=200)
    test_acc.append(Poly_model.score(x_test,y_test))
#     print(i, Poly_model.score(X_poly,y), Poly_model.score(x_test,y_test) )
    
#     print(i, Poly_model.coef_)
#     print('')
    
    if i%3==0:
        plt.plot(t,Poly_model.predict(PR.fit_transform(t)), label=i) #, color=colors[i%5])

plt.legend()
