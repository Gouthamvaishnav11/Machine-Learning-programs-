import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('position_salaries.csv')
X = dataset.iloc[:, 1:2].values   # Years of Experience
y = dataset.iloc[:, 2].values     # Salary

# Splitting dataset into training & test set (optional)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_reg = PolynomialFeatures(degree=4)   # You can change degree
X_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visualizing Polynomial Regression results
def viz_polynomial():
    plt.scatter(X, y, color='red')
    plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color='blue')
    plt.title('Polynomial Regression')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.show()

viz_polynomial()
