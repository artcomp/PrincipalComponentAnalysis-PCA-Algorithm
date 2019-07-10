from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import math
from itertools import combinations
import statistics

# a = (1, 2, 3)
# b = (4, 5, 6)
# dst = distance.euclidean(a, b)


def interCluterDistance(centers):
	distances = [distance.euclidean(p1, p2) for p1, p2 in combinations(centers, 2)]
	avg_distance = sum(distances) / len(distances)
	st_dev = statistics.stdev(distances)
	return  st_dev / avg_distance


def euclidianDistance(point_a, point_b):
	return distance.euclidean(point_a,point_b)

def intraClusterDistance(center, points):
	
	sum_distance_of_point_and_its_centroid = []
	for i in range(len(center)):
		result = 0
		for j in points[i]:
			result = result + euclidianDistance(center[i], j)
		sum_distance_of_point_and_its_centroid.append(result)

	average_intra_distance = sum(sum_distance_of_point_and_its_centroid)/len(center)
	st_dev = (statistics.stdev(sum_distance_of_point_and_its_centroid))
	
	return st_dev / average_intra_distance 


def plotData(X,centers):
	plt.scatter(X[:,0],X[:,1], s=50, cmap='viridis')
	plt.scatter(centers[:,0], centers[:,1], c='black', s=200, alpha=0.5)
	plt.show()



def clusterization(X, num_clusters):

	kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
	kmeans.predict(X)

	# plotData(X, kmeans.cluster_centers_)

	#list of lists, the number os lists depends on
	#the amount of clusters possible
	l = [ [] for i in range(num_clusters) ]

	# group points by its labels
	for i in range(len(X)):
	    l[kmeans.labels_[i]].append(X[i])

	# get average and std deviation for inter and intra clusters
	cv_intra_cluster = intraClusterDistance(kmeans.cluster_centers_, l)
	cv_inter_cluster = interCluterDistance(kmeans.cluster_centers_)
	beta_cv = cv_intra_cluster / cv_inter_cluster
	return beta_cv

def main():
	X = np.array([
			[1, 2], [1, 4], [1, 0],
			[10, 2], [10, 4], [10, 0]
		])

	num_clusters = [x for x in range(3,14)]

	cv_s = list( map( lambda x: clusterization(X,x) , num_clusters) ) 
	plt.plot(num_clusters,cv_s)
	plt.xlabel('Número de clusters')
	plt.ylabel('Coeficiente de Variação (CV)')
	plt.title('Número ideal de clusters (K)')
	plt.show()
	
if __name__ == '__main__':
	main()