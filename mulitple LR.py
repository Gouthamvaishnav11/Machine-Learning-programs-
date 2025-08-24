

from pandas import DataFrame
from sklearn import linear_model
import statsmodels.api as sm

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

# ---- Define variables for multiple regression ----
X = df[['Interest_Rate','Unemployment_Rate']]
y = df['Stock_Index_Price']

# ---- Model fitting with sklearn ----
regr = linear_model.LinearRegression()
regr.fit(X, y)

print("Intercept:", regr.intercept_)
print("Coefficients:", regr.coef_)

New_Interest_Rate = 2.75
New_Unemployment_Rate = 5.3
predicted_price = regr.predict([[New_Interest_Rate, New_Unemployment_Rate]])
print("\nPredicted Stock Index Price:", predicted_price[0])


X2 = sm.add_constant(X)   
model = sm.OLS(y, X2).fit()
print("\nStatsmodels Summary:\n")
print(model.summary())
