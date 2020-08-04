# Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow Notes

# Other resources
'''
Other resources to look into:
1. Geoffrey Hinton’s course on neural networks and Deep Learning
2. Data Science from Scratch, Joel Grus
3. kaggle.com (ML competition website)
'''

# Conventions used within the book
'''
The following typographical conventions are used in this book:
1. Italic Indicates new terms, URLs, email addresses, filenames, and file extensions.

2. Constant width Used for program listings, as well as within paragraphs to refer to program 
elements such as variable or function names, databases, data types, environment variables, statements and keywords.

3. Constant width bold Shows commands or other text that should be typed literally by the user.
4. Constant width italic Shows text that should be replaced with user-supplied values or by values determined by context.

Note that when the code examples display some outputs, then these code examples are shown with Python prompts (>>> and ...), as in a Python shell, to clearly distinguish the code from the outputs. For example, this code defines the square() function then it computes and displays the square of 3:
>>> def square(x):
 ...     return x ** 2 
 ... 
 >>> result = square(3) 
 >>> result 
 9
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

# 