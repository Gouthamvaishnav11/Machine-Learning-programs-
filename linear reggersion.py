# Program: Linear Regression on Stock Market Data

from pandas import DataFrame
from sklearn import linear_model

# ---- Dataset ----
Stock_Market = {
    'Year': [
        2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,
        2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016
    ],
    'Month': [
        12,11,10,9,8,7,6,5,4,3,2,1,
        12,11,10,9,8,7,6,5,4,3,2,1
    ],
    'Interest_Rate': [
        2.75,2.5,2.5,2.5,2.5,2.52,2.52,2.52,2.25,2.25,2.25,2.25,
        2.2,2.2,2.1,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75
    ],
    'Unemployment_Rate': [
        5.3,5.3,5.3,5.3,5.3,5.4,5.6,5.5,5.5,5.5,5.5,5.5,
        6.1,6.2,6.1,6.1,6.1,6.1,6.2,6.2,6.2,6.1,6.1,6.1
    ],
    'Stock_Index_Price': [
        1464,1394,1357,1293,1256,1254,1234,1195,1159,1167,1130,1075,
        1047,965,943,958,971,949,884,866,876,822,704,719
    ]
}

# ---- Create DataFrame ----
df = DataFrame(
    Stock_Market,
    columns=['Year','Month','Interest_Rate','Unemployment_Rate','Stock_Index_Price']
)

# ---- Define Independent & Dependent Variables ----
X = df[['Interest_Rate']]   # Independent variable
y = df['Stock_Index_Price'] # Dependent variable

# ---- Linear Regression Model ----
regr = linear_model.LinearRegression()
regr.fit(X, y)

# ---- Display Intercept & Coefficients ----
print("Intercept (b0):", regr.intercept_)
print("Coefficient (b1):", regr.coef_)


New_Interest_Rate = 2.75
predicted_price = regr.predict([[New_Interest_Rate]])
print("Predicted Stock Index Price for Interest Rate", New_Interest_Rate, "is:", predicted_price[0])
