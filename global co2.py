import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('global_co2.csv')
X=dataset.iloc[:,:1]
y=dataset.iloc[ : ,1:2]

"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5 )
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('global co')
plt.xlabel('year')
plt.ylabel('co')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('global co')
plt.xlabel('year')
plt.ylabel('co')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('global co')
plt.xlabel('year')
plt.ylabel('co')
plt.show()

# Predicting a new result with Linear Regression
a=lin_reg.predict([[2011]])
b=lin_reg.predict([[2012]])
c=lin_reg.predict([[2013]])
print('the production of co2 according to linear regresion in years 2011,2012,2103')
print(a,b,c)
# Predicting a new result with Polynomial Regression
d=lin_reg_2.predict(poly_reg.fit_transform([[2011]]))
e=lin_reg_2.predict(poly_reg.fit_transform([[2012]]))
e=lin_reg_2.predict(poly_reg.fit_transform([[2013]]))
print('the production of co2 according to polynomial regresion in years 2011,2012,2103')
print(d,e)