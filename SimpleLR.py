

import pandas as pd
from pandas import DataFrame
from sklearn import linear_model


df = pd.read_csv("stock.csv")

# ---- Define variables ----
X = df[['Interest_Rate']]        # Independent variable
y = df['Stock_Index_Price']      # Dependent variable

# ---- Model fitting ----
regr = linear_model.LinearRegression()
regr.fit(X, y)

# ---- Display intercept & coefficients ----
print("Intercept:", regr.intercept_)
print("Coefficient:", regr.coef_)

# ---- Prediction for all Interest Rates ----
New_Interest_Rate = df[['Interest_Rate']]
df1 = DataFrame(regr.predict(New_Interest_Rate))
print("\nPredicted Stock Index Price:\n", df1)
