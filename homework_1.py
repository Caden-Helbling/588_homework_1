import numpy as np
import pandas as pd
import matplotlib.pyplot as plt ## needed to display plots
from sklearn.model_selection import train_test_split
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Load the Iris data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
data = pd.read_csv(url, header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'iris_class'])

# Convert 'iris_class' to ordinal variables
data_ordinal = data.replace({'iris_class': {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}})
data_ordinal.sample(5)

# Draw histograms for each attribute
attributes = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
species = data['iris_class'].unique()

# Create subplots for each
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
for i, attr in enumerate(attributes):
    row, col = i // 2, i % 2
    for sp in species:
        axs[row, col].hist(data[data['iris_class'] == sp][attr], bins=15, alpha=0.5, label=sp)
    axs[row, col].set_title(f'Histogram of {attr}')
    axs[row, col].set_xlabel(attr)
    axs[row, col].set_ylabel('Frequency')
    axs[row, col].legend()

plt.tight_layout()
plt.show()

# Define a simple classifier
def classify_iris(row):
    # Setosa has a shorter petal length than the other iris types
    if row['petal_length'] < 2.5:
        return 'Iris-setosa'
    # Versicolor falls in the middle in terms of petal length with little overlap
    elif row['petal_length'] < 5.0:
        return 'Iris-versicolor'
    # Virginica has the longest petal length
    else:
        return 'Iris-virginica'

# Apply classifier to each row
data['predicted_class'] = data.apply(classify_iris, axis=1)

# Calculate accuracy
accuracy = (data['predicted_class'] == data['iris_class']).mean()
print(f'Accuracy of the simple classifier: {accuracy * 100:.2f}%')

# Define color mapping for each species
color_map = {0: 'blue', 1: 'orange', 2: 'green'}
labels_map = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}

# Create scatter plots for each attribute pair
combinations = [('sepal_length', 'petal_length'), ('sepal_length', 'petal_width'),
                ('sepal_width', 'petal_length'), ('sepal_width', 'petal_width')]

for x_attr, y_attr in combinations:
    plt.figure(figsize=(8, 6))
    
    # Plot each class with a specific color
    for label, color in color_map.items():
        subset = data_ordinal[data_ordinal['iris_class'] == label]
        plt.scatter(subset[x_attr], subset[y_attr], c=color, label=labels_map[label], alpha=0.6)
    
    plt.xlabel(x_attr)
    plt.ylabel(y_attr)
    plt.legend(title="Species")
    plt.show()

# KNN