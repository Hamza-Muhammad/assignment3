import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('annual_temp.csv')
y=dataset.iloc[1: :2, 2:]
x=dataset.iloc[1: :2,1:2]

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x,y)


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree =12)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

y_pred1=regressor.predict([[2016]])
y_pred2=regressor.predict([[2017]])

print('by linear regression the annual tempreture for year 2016 and 2017 are',y_pred1,y_pred2)

plt.scatter(x,y,color='blue')
plt.scatter(y_pred1,y_pred2,color='red')
plt.title('annual temperature of GISTEMP')
plt.xlabel('year')
plt.xlim(1980,2020)
plt.ylabel('mean temperature')
plt.show()


plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('annual temp gistemp')
plt.xlabel('year')
plt.ylabel('mean temp')
plt.show()


print('by polynomial regression the annual temperature for year 2016 and 2017 are',b,c)

b=lin_reg_2.predict(poly_reg.fit_transform([[2016]]))
c=lin_reg_2.predict(poly_reg.fit_transform([[2017]]))

from sklearn.tree import DecisionTreeRegressor
regression=DecisionTreeRegressor()
regression.fit(x,y)

plt.scatter(x, y, color = 'red')

plt.plot(x,regression.predict(x), color = 'blue')
plt.title('annual temperature(gistemp)')
plt.xlabel('year')
plt.ylabel('mean temp')
plt.show()

e=regression.predict([[2016]])
f=regression.predict([[2017]])
print('by decision tree regression the annual temperature for year 2016 and 2017 are',e,f)

