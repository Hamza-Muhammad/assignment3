import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('housing price.csv')
x=dataset.iloc[ : ,:1]
y=dataset.iloc[ : ,1: ]


from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=7)
x_poly=poly_reg.fit_transform(x)
poly_reg.fit(x_poly,y)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)

# Visualising the Linear Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('housing price')
plt.xlabel('id')
plt.ylabel('price')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('housing price')
plt.xlabel('id')
plt.ylabel('price')
plt.show()

b=lin_reg2.predict(poly_reg.fit_transform([[2921]]))
print('by polynomial regression,the price for the given id is',b)

#visualizing decistion tree result
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x, y)



plt.scatter(x, y, color = 'red')

plt.plot(x,regressor.predict(x), color = 'blue')
plt.title('housing price(Decision Tree Regression)')
plt.xlabel('id')
plt.ylabel('price')
plt.show()

y_pred = regressor.predict([[2921]])
print('by decision tree regressor,the price for given id is',y_pred)