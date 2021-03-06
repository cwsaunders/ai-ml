# Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow Notes

# Other resources
'''
Other resources to look into:
1. Geoffrey Hinton’s course on neural networks and Deep Learning
2. Data Science from Scratch, Joel Grus
3. kaggle.com (ML competition website)
'''

# Instances
'''
when you have the training data, you can think of it as a set of instances.
Consider the problem of email classification into spam/not spam.
So in your training data, you have a set of pairs (x, y)
where each x represents the set of values or features which correspond to a particular email, for example,
x can be a vector representation of an email which tells about the existence or non existence
of few particular words which might help for the classification of email and y tells you whether that email is spam or not spam.
Here the pair (x,y) is called an instance.
'''

# Conventions used within the book
'''
The following typographical conventions are used in this book:
1. Italic Indicates new terms, URLs, email addresses, filenames, and file extensions.

2. Constant width Used for program listings, as well as within paragraphs to refer to program 
elements such as variable or function names, databases, data types, environment variables, statements and keywords.

3. Constant width bold Shows commands or other text that should be typed literally by the user.
4. Constant width italic Shows text that should be replaced with user-supplied values or by values determined by context.

Note that when the code examples display some outputs, then these code examples are shown with Python prompts (>>> and ...),
as in a Python shell, to clearly distinguish the code from the outputs.
For example, this code defines the square() function then it computes and displays the square of 3:
>>> def square(x):
 ...     return x ** 2 
 ... 
 >>> result = square(3) 
 >>> result 
 9
'''

# Notation
'''
m == the number of instances in the dataset you are measuring the RMSE on. 
For example, if you are evaluating the RMSE on a validation set of 2,000 districts, then m = 2,000

x^(i) and y^(i) == x^i is a vector of all the feature values (excluding the label) of the ith instance in the dataset, and y(i) is its label
(the desired output value for that instance).
For example, if the first district in the dataset is located at longitude –118.29°, latitude 33.91°, and it has 1,416 inhabitants with a median income of $38,372, and the median house value is $156,400 (ignoring the other features for now), then:
x^(1) = −118.29, 33.91, 1,416, 38,372
y^(1) = 156,400

X is a matrix containing all the feature values (excluding labels) of all instances in the dataset.
There is one row per instance and the ith row is equal to the transpose of x(i), noted (x(i))T.4 —For example,
if the first district is as just described, then the matrix X looks like this:
X =
(x^(1))^T
(x^(2))^T
⋮
(x^(1999))^T
(x^(2000))^T
=
−118.29 33.91 1,416 38,372
            ⋮ ⋮ ⋮ ⋮

h == your system’s prediction function, also called a hypothesis.
When your system is given an instance’s feature vector x(i), it outputs a predicted value ŷ(i) = h(x(i)) for that instance
(ŷ is pronounced “y-hat”).
For example, if your system predicts that the median housing price in the first district is $158,400, then ŷ(1) = h(x(1)) = 158,400.
The prediction error for this district is ŷ(1) – y(1) = 2,000

RMSE(X,h) == the cost function measured on the set of examples using your hypothesis h. 

Misc:

We use lowercase italic font for scalar values (such as m or y(i)) and function names (such as h),
lowercase bold font for vectors (such as x(i)), and uppercase bold font for matrices (such as X).

'''

# Additional internal material
'''
Supplemental material (code examples, exercises, etc.) is available for download at https://github.com/ageron/handson-ml2
'''

# Decreasing performance over time
'''
If an algorithm is learning live (e.g google) it may be susceptible to decreasing in performance. This may be due to
a malicious user feeding bad data into the algorithm. e.g someone spamming google.com with searches for their site to increase
their ranking. A decrease in performance may also simply be due to bad data. Actions that may be taken to mitigate this:
1. implementing an anomoly-detection algorithm
2. temporarily turning off learning 
3. reverting to a prior state
'''

# Example code to create GDP to happiness model -- linear regression
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model

# Load the data
oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("gdp_per_capita.csv",thousands=',',delimiter='\t',
encoding='latin1', na_values="n/a")

# Prepare data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# Visualize data
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.show()

# Select linear model
model = sklean.linear_model.LinearRegression()

# Train the model
model.fit(X,y)

# Make a prediction for Cyprus
X_new = [[22587]] # Cyprus GDP / capita
print(model.predict(X_new)) # Output 
'''

# k-Nearest Neighbors regression
'''
The prior example code would create a k-nearest neighbors regression model simply by replacing these two lines:
import sklearn.linear_model
model = sklearn.linear_model.LinearRegression() 

with these two lines:

import sklearn.neighbors
model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
'''

# Regularization
'''
Constraining a model to make it simpler and reduce the risk of overfitting is called regularization
'''

# Hyperparameter
'''
A hyperparameter is a parameter of a learning algorithm (not of the model). 
As such, it is not affected by the learning algorithm itself; it must be set prior to training and remains constant during training.
If you set the regularization hyperparameter to a very large value, you will get an almost flat model (a slope close to zero);
the learning algorithm will almost certainly not overfit the training data, but it will be less likely to find a good solution.
Tuning hyperparameters is an important part of building a Machine Learning system 
(you will see a detailed example in the next chapter). 

Notes on tuning p. 58

holdout validation is a good solution for tuning. (p. 58)
e.g holdout validation is when you hold out a part of the training set to evaluate several candidate models and select the best one.
the new heldout set is called the validation set. (sometimes development set/dev set)
'''

# Testing and Validating
'''
Split your data into two sets:
the training set and the test set.
As these names imply, you train your model using the training set, and you test it using the test set.


The error rate on new cases is called the generalization error (or out-ofsample error).
By evaluating your model on the test set, you get an estimate of this error.


If the training error is low (i.e., your model makes few mistakes on the training set) but the generalization error is high,
it means that your model is overfitting the training data.
'''

# Train-Dev Set
'''
Popularized by Andrew Ng, described on p. 59

Method for testing if your model is properly assessing data that will affect your use-case or data you collected from another source.
'''

# Dataset locations for practice
'''
 Popular open data repositories:
 —UC Irvine Machine Learning Repository
 —Kaggle datasets
 —Amazon’s AWS datasets

 • Meta portals (they list open data repositories):
 —http://dataportals.org/
 —http://opendatamonitor.eu/
 —http://quandl.com/

 • Other pages listing many popular open data repositories:
 —Wikipedia’s list of Machine Learning datasets
 —https://www.quora.com/Where-can-I-find-large-datasets-open-to-the-public
 —Datasets subreddit 
'''

# ***************************
# End-To-End Project

# Picking a model for your project
'''
Real estate example:
Okay, with all this information you are now ready to start designing your system.
First, you need to frame the problem: is it supervised, unsupervised, or Reinforcement Learning?
Is it a classification task, a regression task, or something else? Should you use batch learning or online learning techniques?
Before you read on, pause and try to answer these questions for yourself. Have you found the answers?
Let’s see: it is clearly a typical supervised learning task since you are given labeled training examples
(each instance comes with the expected output, i.e., the district’s median housing price).
Moreover, it is also a typical regression task, since you are asked to predict a value.
More specifically, this is a multiple regression problem since the system will use multiple features to make a prediction
(it will use the district’s population, the median income, etc.).
It is also a univariate regression problem since we are only trying to predict a single value for each district.
If we were trying to predict multiple values per district, it would be a multivariate regression problem.
Finally, there is no continuous flow of data coming in the system, there is no particular need to adjust to changing data rapidly,
and the data is small enough to fit in memory, so plain batch learning should do just fine.

'''

# Selecting a performance measure
'''
Root Mean Square Error (RMSE):
Your next step is to select a performance measure.
A typical performance measure for regression problems is the Root Mean Square Error (RMSE).
It gives an idea of how much error the system typically makes in its predictions, with a higher weight for large errors.
Equation 2-1 shows the mathematical formula to compute the RMSE. (P. 68)
Computing the root of a sum of squares (RMSE) corresponds to the Euclidean norm: it is the notion of distance you are familiar with.
It is also called the ℓ2 norm, noted ∥ · ∥2 (or just ∥ · ∥). 

Mean Absolute Error (MAE):
Even though the RMSE is generally the preferred performance measure for regression tasks,
in some contexts you may prefer to use another function. For example, suppose that there are many outlier districts.
In that case, you may consider using the Mean Absolute Error (also called the Average Absolute Deviation; see Equation 2-2). (P. 70)
Computing the sum of absolutes (MAE) corresponds to the ℓ1 norm, noted ∥ · ∥1.
It is sometimes called the Manhattan norm because it measures the distance between two points in a city if you can only
travel along orthogonal city blocks. 

Extra info:
Both the RMSE and the MAE are ways to measure the distance between two vectors:
the vector of predictions and the vector of target values. Various distance measures, or norms, are possible

The higher the norm index, the more it focuses on large values and neglects small ones.
This is why the RMSE is more sensitive to outliers than the MAE.
'''

# Checking Assumptions
'''
See page 71
'''

# Initial Technical Steps for Chp 2 Project
'''
1. Get the data
2. Create the workspace
Open a terminal and type the following commands (after the $ prompts):
$ export ML_PATH="$HOME/ml"      # You can change the path if you prefer
$ mkdir -p $ML_PATH 
You will need a number of Python modules: Jupyter, NumPy, Pandas, Matplotlib, and Scikit-Learn. 
'''

# Creating a test set
'''
pick some instances randomly, typically 20% of the dataset (or less if your dataset is very large), and set them aside

Code available in Chp_2_ML_Proj.ipynb

Different implementations P. 81-82
'''

# Dealing with attributes that have missing inputs (i.g unknown)
'''
housing.dropna(subset=["total_bedrooms"])    # option 1
housing.drop("total_bedrooms", axis=1)       # option 2
median = housing["total_bedrooms"].median()  # option 3
housing["total_bedrooms"].fillna(median, inplace=True) # option 3 continued

sklearn module available (shown in proj 2)
'''

# Transforming into numerical data
'''
The following paragraph describes a situation where categories are not implicitly more different based on increasing or decreasing
numbers:
One issue with this representation is that ML algorithms will assume that two nearby values are more similar than two distant values.
This may be fine in some cases (e.g., for ordered categories such as “bad”, “average”, “good”, “excellent”),
but it is obviously not the case for the ocean_proximity column
(for example, categories 0 and 4 are clearly more similar than categories 0 and 1).
To fix this issue, a common solution is to create one binary attribute per category:
one attribute equal to 1 when the category is “<1H OCEAN” (and 0 otherwise),
another attribute equal to 1 when the category is “INLAND” (and 0 otherwise),
and so on. This is called one-hot encoding, because only one attribute will be equal to 1 (hot),
while the others will be 0 (cold). The new attributes are sometimes called dummy attributes.
Scikit-Learn provides a OneHotEn coder class to convert categorical values into one-hot vectors20

Transforming class example available in sklearn-transformer-class.py in the hands_on_ml folder
Useful for hyperparameters

transforming vectorization tools also shown in use in Chp_2_ML_Proj.ipynb in a more simplistic form. -- MORE INFO ON P. 98 OF TEXT
'''

# Min-Max Scaling (also called normalization) and/or standardization
'''
ML models tend not to perform well when different attributes have significantly different scale. i.g one attribute ranges from
1-100,000 and another attribute ranges from 0-10. The two common ways to solve this are min-max scaling and standardization

Min-Max:
Values are shifted and rescaled so that they end up ranging from 0 to 1.
This is done by subtracting the min value and dividing by the max minus the min.
Scikit-Learn provides a transformer called MinMaxScaler for this. It has a feature_range hyperparameter that lets you change the
range if you don't want 0-1 for some reason.

Standardization:
first it subtracts the mean value (so standardized values always have a zero mean),
and then it divides by the standard deviation so that the resulting distribution has unit variance.
Unlike min-max scaling, standardization does not bound values to a specific range,
which may be a problem for some algorithms (e.g., neural networks often expect an input value ranging from 0 to 1)
However, standardization is much less affected by outliers. For example, suppose a district had a median income equal to 100
(by mistake). Min-max scaling would then crush all the other values from 0–15 down to 0–0.15,
whereas standardization would not be much affected. Scikit-Learn provides a transformer called StandardScaler for standardization.
'''

# Scikit-Learn Pipieline module (More transformation of data)
'''
Scikit-Learn provides the Pipeline class to help with sequences of transformations. 


'''

# Fine tuning the model
'''
Grid Search: Code available in project. Best used on smaller projects, as it is resource intensive
Randomized Search: Best used in larger projects, uses estimation and randomization. Code available in project
'''

# Additional notes available in chapter 2 project (jupyter notebook)

# End
# ***********************

# Chapter 3

# MNIST Dataset
'''
Often referred to as the "hello world" of machine learning. Probably the most popular dataset of all time.

70,000 small images of digits written by high school students and US Census Bereau workers. Each image is labeled with the digit 
it represents.

'''

# ROC Curve (reciever operating characteristic curve)
'''
A common tool used with binary classifiers. It is similar to precision/recall curve, but instead of plotting precision vs recall
the ROC curve plots the true positive rate (recall) against the false positive rate. The FPR (False positive rate) is the ratio of
negative instances that are incorrectly classified as positive.
It is equal to one minus the true negative rate, which is the ratio of negative instances that are correctly classified as negative.
The TNR is also called specificity. 
'''

# confusion matrix / how to work on misclassified data
'''
Now you can clearly see the kinds of errors the classifier makes. Remember that rows
represent actual classes, while columns represent predicted classes. The column for
class 8 is quite bright, which tells you that many images get misclassified as 8s. How‐
ever, the row for class 8 is not that bad, telling you that actual 8s in general get prop‐
erly classified as 8s. As you can see, the confusion matrix is not necessarily
symmetrical. You can also see that 3s and 5s often get confused (in both directions).
Analyzing the confusion matrix can often give you insights on ways to improve your
classifier. Looking at this plot, it seems that your efforts should be spent on reducing
the false 8s. For example, you could try to gather more training data for digits that
look like 8s (but are not) so the classifier can learn to distinguish them from real 8s.
Or you could engineer new features that would help the classifier—for example, writ‐
ing an algorithm to count the number of closed loops (e.g., 8 has two, 6 has one, 5 has
none). Or you could preprocess the images (e.g., using Scikit-Image, Pillow, or
OpenCV) to make some patterns stand out more, such as closed loops.
'''

# Linear Regression (Math)
'''
(((
    PAGE 141:
This can be written much more concisely using a vectorized form, as shown in Equa‐
tion 4-2.
Equation 4-2. Linear Regression model prediction (vectorized form)
y = hθx = θ · x
• θ is the model’s parameter vector, containing the bias term θ0 and the feature weights θ1 to θn.
• x is the instance’s feature vector, containing x0 to xn, with x0 always equal to 1.
• θ · x is the dot product of the vectors θ and x, which is of course equal to θ0x0 + θ1 x1+ θ2x2+ ⋯ + θnxn.
• hθ
 is the hypothesis function, using the model parameters θ.
)))
 '''

 # Pseudoinverse computation
'''
The pseudoinverse itself is computed using a standard matrix factorization technique
called Singular Value Decomposition (SVD) that can decompose the training set
matrix X into the matrix multiplication of three matrices U Σ VT
(see numpy.linalg.svd()). The pseudoinverse is computed as X+= VΣ+UT. To computethe matrix Σ+
, the algorithm takes Σ and sets to zero all values smaller than a tiny
threshold value, then it replaces all the non-zero values with their inverse, and finally
it transposes the resulting matrix. This approach is more efficient than computing the
Normal Equation, plus it handles edge cases nicely: indeed, the Normal Equation may
not work if the matrix XTX is not invertible (i.e., singular), such as if m < n or if some
features are redundant, but the pseudoinverse is always defined.
'''

# Normal equation / computational complexity (big O)
'''
The Normal Equation computes the inverse of XT X, which is an (n + 1) × (n + 1)
matrix (where n is the number of features). The computational complexity of inverting
such a matrix is typically about O(n2.4) to O(n3) (depending on the implementation).In other words, if you double the number of features,
you multiply the computationtime by roughly 22.4 = 5.3 to 23 = 8.
The SVD approach used by Scikit-Learn’s LinearRegression class is about O(n2). If you double the number of features,
you multiply the computation time by roughly 4.
'''