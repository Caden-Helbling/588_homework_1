# 588_homework_1

#### Assignment
1. Write a code to read https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data file to a pandas DataFrame. Set `header=None` and `names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'iris_class']` to manually set the column names. Name your DataFrame `data`.

#### Answer
1. See homework_1.py code

#### Assignment
1. Write a code to draw the similar histogram plots for other attributes, i.e. sepal width, petal length, and petal width.
2. For each attribute, can you tell the difference between the three species? You can just "eyeball".
3. Based on your answer above, build a simple `if-else` logic to classify irises. Test your logic on the Iris data set. How accurate can you be?

#### Answer
1. See homework_1.py code
2. For each attribute you can easily tell the difference between the three species as for each histogram they tend to cluster around specific measures.
3. See homework_1.py code. By Using the information gained form the petal_length histogram which provided pretty clear separation between the types a simple if statement split along the 2.5 and 5 measurements gives a 94% accuracy.


#### Assignment
1. Implement a code to classify all the flowers in the test data set.
2. Compare the predicted result with the actual ground truth. What is the accuracy?
3. Plot the accuracy as you vary k=1, 2, 3, ..., 20. Does the accuracy changes along k? Is there any pattern you can observe?

#### Answers
1. See homework_1.py code
2. The accuracy with a KNN of 1 is around 95%
3. Plotting the accuracy as k is varied from 1 to 20 several different times a pattern quickly emerges. k=1 is fairly accurate then there is a drop with k=2, followed by an oscillation of increasing accuracy with each odd k and decreasing accuracy for each even k. They peak around k =13 before continuing an oscillation trending downward.
