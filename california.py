import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('california.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 4].values

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'black')
plt.plot(X, lin_reg.predict(X), color = 'red')
plt.title('R&D vs Profit California Linear Regression')
plt.xlabel('R&D')
plt.ylabel('Profit')
plt.show()

# Visualising the Polynomial Regression results

plt.scatter(X, y, color = 'black')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'red')
plt.title('R&D vs Profit California (Polynomial Regression)')
plt.xlabel('R&D')
plt.ylabel('Profit')
plt.show()


# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'black')
plt.plot(X_grid, regressor.predict(X_grid), color = 'red')
plt.title('R&D vs Profit California (Decision Tree Regression)')
plt.xlabel('R&D')
plt.ylabel('Profit')
plt.show()

print('Profits ACCORDING TO  Polynomial Regression')
a=(lin_reg_2.predict(poly_reg.fit_transform([[9000000]])))
print(a)

print('Profits ACCORDING TO  Decision TREE')
y_pred = regressor.predict([[9000000]])
print(y_pred)
