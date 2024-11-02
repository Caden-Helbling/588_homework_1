# 588_homework_1
# Caden Helbling and Christopher Woodard 

#### Instructions
1. Create a virtual environment with `python -m venv venv`
2. Activate the environment with `.\venv\Scripts\activate` on Windows or `source venv/bin/activate` on MacOS/Linux
3. Install requirements with `pip install -r requirements.txt`
4. Run homework_2.py and view output


#### Assignment
1. Write a code to draw the similar histogram plots for other attributes, i.e. sepal width, petal length, and petal width.
2. For each attribute, can you tell the difference between the three species? You can just "eyeball".
3. Based on your answer above, build a simple `if-else` logic to classify irises. Test your logic on the Iris data set. How accurate can you be?

#### Answer
1. See homework_1.py code
2. For each attribute you can easily tell the difference between the three species as for each histogram they tend to cluster around specific measures.
3. See homework_1.py code. By Using the information gained form the petal_length histogram which provided pretty clear separation between the types a simple if statement split along the 2.5 and 5 measurements gives a 94.67% accuracy.

#### Assignment

1. Write a code to draw the scatter plot for all the other combinations than 'sepal length' - 'sepal width'.
2. For each plot, can you draw a straight line separating the different species? (again, eyeballing) What is the slope and the intercept of the line you came up with, roughly?
3. Implement a linear classifier using the line equations you manually came up with. What is the accuracy?

#### Answers
1. See homework_1.py code
2. We drew 2 staight lines for each of our 5 scatter plots
These are our slopes and intercepts:

'sepal_length' - 'petal_length'
slope intercept
0.5   0
0.25  3.18

sepal_length' - 'petal_width' 
slope intercept
0.0   0.75
0.08  1.18
    
'sepal_width' 'petal_length' 
slope intercept
0.5   1 
0.2   4.1

'sepal_width', 'petal_width' 
slope intercept
0.25  0 
0.15  1.25

'petal_length', 'petal_width' 
slope intercept
-0.5  2
0     1.65

3. We implemented the linear classifier using the lines for the petal_length - sepal_length
due to it having the best splitting points in the data. By doing this we got an accuracy
of 97.33%.


#### Assignment
1. Implement a code to classify all the flowers in the test data set.
2. Compare the predicted result with the actual ground truth. What is the accuracy?
3. Plot the accuracy as you vary k=1, 2, 3, ..., 20. Does the accuracy changes along k? Is there any pattern you can observe?

#### Answers
1. See homework_1.py code
2. When ran once with k=5 we tended to get an accuracy between 93% and 100%. However, after we set it up to run 100 times and take the average of all 100 runs our accuracy for k=5 tended to fall between 95.5% and 96.5%.
3. When plotting the k values from 1 to 20, we decided to run our KNN algorithm on 100 different randomly generated sets of data. This is due to each time we ran it the data being very different than the previous time (Sometimes k=1 would be the best, while others k=20 would be the best). After averaging the data between 100 runs our data was much more consistent.  We noticed an oscillating pattern where each odd number tended to be greater than the even numbers it was between. This is probably due to odd numbers being better at breaking ties between 2 different groups. We also noticed that the accuracy slowly rose until k=13 or rarely k=15 then began to drop. Meaning that for this dataset k=13 is probably the most optimal choice for k.
