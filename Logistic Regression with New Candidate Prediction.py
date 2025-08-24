import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Candidate dataset
candidates = {
    'gmat': [780, 750, 690, 710, 680, 730, 690, 720, 740, 690,
             590, 610, 690, 710, 680, 570, 670, 660, 580, 590],
    'gpa': [4, 3.9, 3.3, 3.7, 3.9, 3.3, 3.4, 3.5, 3.0, 3.4,
            3.2, 3.4, 3.6, 2.3, 3.7, 2.7, 3.3, 3.7, 2.9, 3.2],
    'work_experience': [3, 4, 3, 5, 4, 6, 1, 4, 5, 3,
                        6, 5, 6, 1, 2, 4, 6, 5, 1, 2],
    'admitted': [1, 1, 1, 1, 1, 1, 0, 0, 1, 0,
                 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]
}

df = pd.DataFrame(candidates, columns=['gmat', 'gpa', 'work_experience', 'admitted'])

# Splitting features and target
X = df[['gmat', 'gpa', 'work_experience']]
y = df['admitted']

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Fitting logistic regression
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)

# New candidates dataset for prediction
new_candidates = {
    'gmat': [590, 740, 680, 610, 710],
    'gpa': [2.0, 3.7, 3.3, 2.3, 3.0],
    'work_experience': [3, 4, 6, 1, 5]
}

df2 = pd.DataFrame(new_candidates, columns=['gmat', 'gpa', 'work_experience'])

# Predictions
y_pred = logistic_regression.predict(df2)

# Display new dataset & predictions
print(df2)
print(y_pred)
