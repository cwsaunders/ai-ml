import numpy as np

X = 2*np.random.rand(100,1) # generate random data
y = 4 + 3 * X + np.random.rand(100,1) # generate random data

X_b = np.c_[np.ones((100,1)), X] # add xTHETA = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) # calculate theta hat using normal equation

