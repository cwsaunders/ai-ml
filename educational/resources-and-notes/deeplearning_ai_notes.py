# Deeplearning.ai Coursera Notes

# Note resource
'''
https://www.slideshare.net/TessFerrandez/notes-from-coursera-deep-learning-courses-by-andrew-ng?trk=v-feed&lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_recent_activity_details_shares%3Ba1Vq7AYlTOmwFz%2FVvzR2wA%3D%3D&licu=urn%3Ali%3Acontrol%3Ad_flagship3_profile_view_base_recent_activity_details_shares-object
'''

# Logistic Regression
'''
An algorithm for binary classification (1 represents that an item is x and 0 represents that it is not x)
C1W2L01 explains binary classification in detail
Used when Y = 0 or Y = 1 (e.g binary classification problems)


'''

# Notation for course
'''
Video C1W2L01 explains. Also available online.
w = n dimensional vector (weights)
b = real number (regression coefficients)
In code, when computing " dFindOutputVar / dvar " for derivatives. -- represents the derivative of the final output variable
you care about with respect to different intermediaries in your code.
this is represented by "dvar"
'''

# Computing derivatives
'''
When solving for derivatives in practice generally the most important variable is your output variable. (C1W2L08 -- info)
Derivative information C1W2L08 especially around 6-7 minutes.
'''

# Gradient descent using derivatives (for logistic regression)
'''
Python variable names available in C1W2L09

code:

j=0
dw1=0
dw2=0
db=0

for i in m:
    z^i = w^tx^i+b
    a^i=sigmoid(z^i)
    j+= -[y^iloga^i+(1-y^i)log(1-a^i)]
    dz^i = a^i-y^i
    dw1+=x1^idz^i
    dw2+=x2^idz^i
    db+=dz^i
j/=m
dw1 /= m
dw2 /= m
db /= m

However, the weakness with this method is that you need to add one line in for each additional value of x. This can be fixed
with vectorization. (Getting rid of for loops)

'''