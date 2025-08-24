import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

# Sample data
data = {
    'y_Predicted': [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
    'y_Actual':    [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])

# Creating confusion matrix
confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'],
                               rownames=['Actual'], colnames=['Predicted'],
                               margins=True)

# Generating heatmap and displaying it
ax = sn.heatmap(confusion_matrix, annot=True, fmt='d')
plt.show()

# Printing confusion matrix in text form
print(confusion_matrix)
