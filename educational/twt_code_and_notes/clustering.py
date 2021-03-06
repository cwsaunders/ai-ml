# Clustering
'''
Algorithm finds clusters within dataset
Clustering
Now that we've covered regression and classification it's time to talk about clustering data! 

Clustering is a Machine Learning technique that involves the grouping of data points. In theory, data points that are in the same group should have similar properties and/or features, while data points in different groups should have highly dissimilar properties and/or features. (https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68)

Unfortunalty there are issues with the current version of TensorFlow and the implementation for KMeans. This means we cannot use KMeans without writing the algorithm from scratch. We aren't quite at that level yet, so we'll just explain the basics of clustering for now.

Basic Algorithm for K-Means.
- Step 1: Randomly pick K points to place K centroids
- Step 2: Assign all the data points to the centroids by distance. The closest centroid to a point is the one it is assigned to.
- Step 3: Average all the points belonging to each centroid to find the middle of those clusters (center of mass). Place the corresponding centroids into that position.
- Step 4: Reassign every point once again to the closest centroid.
- Step 5: Repeat steps 3-4 until no point changes which centroid it belongs to.

Sources:
https://colab.research.google.com/drive/15Cyy2H7nT40sGR7TBN5wBvgTd57mVKay#forceEdit=true&sandboxMode=true&scrollTo=d0dfaT4esRh3
https://www.youtube.com/watch?v=tPYj3fFJGjk (2:18:40 for sketch)

'''
# Centroid
'''
Where our current cluster is defined.
the center of mass of a geometric object of uniform density.

Using Euclidean distance (defined below) we can find all the dots (data points) proximity to different centroid points (in video
example 3 points are used) and those points are assigned to those centroids based on proximity. Essentially assigning them
into centroid groups
THEN
all the centroids are moved to the middle of all their graphed data points (center of mass)
THEN
the process repeats and the data points are re-assigned to their closest centroid

Once the process ceases we have the correct clusters. Therefore, when the new clusters are assigned it is easy to find where additions
to the graph would fit. i.g any new additions to the graph are added to their closest centroid/cluster

'''
# Euclidean / Manhattan distance
'''
In mathematics, the Euclidean distance or Euclidean metric is the "ordinary" straight-line distance between two points
 in Euclidean space. 
With this distance, Euclidean space becomes a metric space. The associated norm is called the Euclidean norm. 
Older literature refers to the metric as the Pythagorean metric.
'''

# Determining K (number of centroids/clusters)
'''
Source:
https://www.geeksforgeeks.org/ml-determine-the-optimal-value-of-k-in-k-means-clustering/#:~:text=There%20is%20a%20popular%20method,fewer%20elements%20in%20the%20cluster.
'''