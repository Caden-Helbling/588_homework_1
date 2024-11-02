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

# Create scatter plots for each attribute pair & add lines to separate types
line_params = {
    ('sepal_length', 'petal_length'): [(0.5, .25), (0, 3.18)],  # slopes, intercepts
    ('sepal_length', 'petal_width'): [(0, .08),(.75, 1.18)],
    ('sepal_width', 'petal_length'): [(0.5, .2),(1, 4.1)],
    ('sepal_width', 'petal_width'): [(0.25, .15),(0, 1.25)],
    ('petal_length', 'petal_width'): [(-0.5, 0),(2, 1.65)]
}

for (x_attr, y_attr), (slopes, intercepts) in line_params.items():
    plt.figure(figsize=(8, 6))
    
    # Plot each class with a specific color
    for label, color in color_map.items():
        subset = data_ordinal[data_ordinal['iris_class'] == label]
        plt.scatter(subset[x_attr], subset[y_attr], c=color, label=labels_map[label], alpha=0.6)
    
    plt.xlabel(x_attr)
    plt.ylabel(y_attr)
    plt.xlim(0, data_ordinal[x_attr].max() + 1)  # Set x-axis limit
    plt.ylim(0, data_ordinal[y_attr].max() + 1)  # Set y-axis limit
    plt.legend(title="Species")
    for index in range(len(slopes)):
        x_vals = np.array(plt.gca().get_xlim())  # Get the x-axis limits
        y_vals = slopes[index] * x_vals + intercepts[index]
        plt.plot(x_vals, y_vals, '--', color='red', label="Decision Boundary")
    plt.show()
    # Define a function that uses line equations to classify
def classify_linear(row):
    # Example conditions based on the line equations
    if row['petal_length'] < 0.5 * row['sepal_length'] + 0:
        return 'Iris-setosa'
    elif row['petal_length'] < .25 * row['sepal_length'] + 3.18:
        return 'Iris-versicolor'
    else:
        return 'Iris-virginica'

# Apply the classifier to each row and calculate accuracy
data['linear_predicted_class'] = data.apply(classify_linear, axis=1)
accuracy = (data['linear_predicted_class'] == data['iris_class']).mean()
print(f'Accuracy of the linear classifier: {accuracy * 100:.2f}%')

# KNN

#An array used to keep track of the accuracy of each run for all 20 k's
allAcc=np.zeros(20)

#Run the code i amount of times. This is for testing your data. It is easier than rerunning the program a ton
i=100
for x in range(i):

    #Split the data into 120 training data and 30 query data
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(data_ordinal, test_size=0.2)

    #convert training data to np array
    train_np = train.to_numpy()

    #Put first four columns AKA features into a single array, and identifier on its own
    train_X = train_np[:, 0:4]
    train_Y = train_np[:, 4]

    #Run KNN with k=1 all the way to k=20 to see a pattern in how changing k will change accuracy
    for k in range(1,21):

        #variables used to keep track of correct identifications and total identifications
        correct=0 
        total=0

        #Loop through each testing value and run knn on it
        for query in test.iloc:
            #Delete the queries identifier but store it somewhere for later accuracy check
            query = query.to_numpy()
            ground_truth = query[4]
            query = np.delete(query, 4)

            #find the euclidean distance to each point, and end up in an array
            diff = np.abs(train_X - query) 
            square=np.square(diff)
            sum_square = np.sum(square, axis=-1)
            distance=np.sqrt(sum_square)

            #Find the K nearest neighbors
            idx = np.argpartition(distance, k)  #find the k smallest values in distance
            knn = train_Y[idx[:k]]

            #Predict the class based off the highest knn species
            uni, count = np.unique(knn, return_counts=True)

            #increment total and if it was correct increment correct
            total=total+1
            if uni[np.argmax(count)] == ground_truth:
                correct=correct+1

        #calculate the accuracy and add it into the array
        accuracy=correct/total
        allAcc[k-1]=allAcc[k-1]+(accuracy*100)

#print out the avg accuracy for each k value for all runs combined
k=1
for acc in allAcc:
    print(f'Accuracy of KNN with k={k} is {acc / i:.2f}%')
    k=k+1

"""
The most accurate k is always somewhere around k=13+-2
k=1 is surprisingly more accurate than most of the other lower k's
Therefore if you need to have a low k, then k=1 may be best
"""

# Calculate the average accuracy for each k value
average_accuracy = allAcc / i  # Divide by the number of runs to get the average

# Define the range of k values
k_values = range(1, 21)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(k_values, average_accuracy, marker='o', linestyle='-', color='b')
plt.title("KNN Accuracy for Different Values of k over "+ str(i) +" runs")
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("Accuracy (%)")
plt.xticks(k_values)  # Set x-axis ticks to each k value
plt.grid(True)
plt.show()
