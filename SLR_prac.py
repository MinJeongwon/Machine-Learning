import os
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.api as sm # for regression model

#>>>>> print(os.getcwd()) 
#/home/jeongwon/Documents/python/ML

# data
boston = pd.read_csv("./data/Boston_house.csv")
"""Attribute Information (in order):
    - CRIM     per capita crime rate by town
    - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
    - INDUS    proportion of non-retail business acres per town
    - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    - NOX      nitric oxides concentration (parts per 10 million)
    - RM       average number of rooms per dwelling
    - AGE      proportion of owner-occupied units built prior to 1940
    - DIS      weighted distances to five Boston employment centres
    - RAD      index of accessibility to radial highways
    - TAX      full-value property-tax rate per $10,000
    - PTRATIO  pupil-teacher ratio by town
    - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    - LSTAT    % lower status of the population
    - MEDV     Median value of owner-occupied homes in $1000's"""



# extract data w/o target column
boston_data = boston.drop(['Target'], axis = 1)

# SLR(CRIM)
target = boston[['Target']]
crim = boston[['CRIM']]

# add constant term(constant term required in SLR)
crim1 = sm.add_constant(crim, has_constant = "add")

# OLS : minimize RSS(Residual Sum of Squares)
model1 = sm.OLS(target, crim1)
fitted_model1 = model1.fit()

# get parameters
fitted_model1.params

# parameters * data
pred1 = fitted_model1.predict(crim1)

# visualize
plt.figure(figsize = (10, 8))
plt.subplot(221)
plt.scatter(crim, target, label = 'crime')
plt.plot(crim, pred1, label = 'model', color = 'r')
plt.legend()
plt.title('per capita crime rate by town')

#======================================================================
# SLR(LSTAT)
lstat = boston[['LSTAT']]
lstat1 = sm.add_constant(lstat, has_constant = "add")
model2 = sm.OLS(target, lstat1)
fitted_model2 = model2.fit()
pred2 = fitted_model2.predict(lstat1)

plt.subplot(222)
plt.scatter(lstat, target, label = 'lower status')
plt.plot(lstat, pred2, label = 'model', color = 'r')
plt.legend()
plt.title('% lower status of the population')


# SLR(RM)
rm = boston[['RM']]
rm1 = sm.add_constant(rm, has_constant = "add")
model3 = sm.OLS(target, rm1)
fitted_model3 = model3.fit()
pred3 = fitted_model3.predict(rm1)

plt.subplot(223)
plt.scatter(rm, target, label = '# room')
plt.plot(rm, pred3, label = 'model', color = 'r')
plt.legend()
plt.title('average number of rooms per dwelling')


# SLR(RM)
ptratio = boston[['PTRATIO']]
ptratio1 = sm.add_constant(ptratio, has_constant = "add")
model4 = sm.OLS(target, ptratio)
fitted_model4 = model4.fit()
pred4 = fitted_model4.predict(ptratio)

plt.subplot(224)
plt.scatter(ptratio, target, label = 'pupil-teacher')
plt.plot(ptratio, pred4, label = 'model', color = 'r')
plt.legend()
plt.title('pupil-teacher ratio by town')

plt.show()