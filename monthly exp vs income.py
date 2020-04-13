import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('monthlyexp vs incom.csv')
x=dataset.iloc[ : , :1]
y=dataset.iloc[ : ,1: ]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/4, random_state = 0)

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)


y_pred=regressor.predict(x_test)

plt.scatter(x_train, y_train, color = 'black')
plt.plot(x_train, regressor.predict(x_train), color = 'red')
plt.title('monthly expense vs income')
plt.xlabel('monthly expense')
plt.ylabel('income')
plt.show()

# Visualising the Test set results
plt.scatter(x_test, y_test, color = 'black')
plt.plot(x_train, regressor.predict(x_train), color = 'red')
plt.title('monthly expense vs income')
plt.xlabel('monthly expense')
plt.ylabel('income')
plt.show()

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x_train, y_train)

# Visualising the Decision Tree Regression results (higher resolution)

plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'BLACK')
plt.title('Months Experience VS Income (Decision Tree Regression)')
plt.xlabel('Months experience')
plt.ylabel('Income')
plt.show()