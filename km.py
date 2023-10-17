import random
import numpy as np
import sys
import csv

DEFAULT_MAX=64000
SCALER=0.8

class KMeans():
	def __init__(self,data: list, k: int, cluster_size: int=None):
		'''
		Data : inputs from the dataset
		k : number of clusters/categories
		'''
		self.data = data
		self.covariance = np.cov(data[0:len(data[0])])
		self.k = k
		self.clusters = []
		self.data_points = []
		self.max_cluster_size = cluster_size
		# initialize the clusters
		for r in range(self.k):
			tmp_cluster = Cluster(points=[],centroid=np.random.random(size=len(data[0])), dimensions=len(data[0]), max_points=self.max_cluster_size)
			self.clusters.append(tmp_cluster)
		for dp in self.data:
			tmp_dp = DataPoint(value=dp,distance=sys.float_info.max)
			self.data_points.append(tmp_dp)
		self.set_starting()
			

	def distance(self, m, x, C=None):
		# C must be a positive diagonal matrix with a rank equal to n in the data dimension
		if not C:
			C = self.covariance
		x_minus_mu = x - m
		C_inv = np.linalg.pinv(C)
		left_term = np.dot(x_minus_mu, C_inv)
		distance = np.dot(left_term, x_minus_mu)
		return distance

	def k_means(self):
		converged = False
		rounds = 0
		while not converged:
			previous_clusters = self.clusters.copy()
			print(f"Round {rounds}")
			for dp in self.data_points:
				for c in self.clusters:
					if len(c.points) < c.max_points if c.max_points else DEFAULT_MAX:
						dist = self.distance(c.centroid, dp.value)
						if dist < dp.distance:
							dp.update_distance(dist)
							dp.set_cluster(c)
			for c in self.clusters:
				c.update_centroid()
				print(f"Cluster: \n  Centroid: {c.centroid}\nPoints: {[str(x) for _, x in enumerate(c.points)]}\n")
			all_same = 0
			for i in range(len(previous_clusters)):
				if np.array_equal(previous_clusters[i].points,self.clusters[i].points):
					all_same += 1 
			if rounds > 0:
				if all_same >= len(self.clusters):
					converged = True
					print("Converged!")
			rounds += 1 
		return self.clusters

					
	def set_starting(self):
		for i in range(len(self.data_points)):
			selection = i % self.k
			self.data_points[i].set_cluster(self.clusters[selection])
		for c in self.clusters:
			c.update_centroid()
			for p in c.points:
				p.update_distance(self.distance(c.centroid, p.value))

	def assign_cluster(self, data_point):
		for i in range(len(self.clusters)):
			dist = self.distance(self.clusters[i]['centroid'], data_point,C=self.covariance)
			if dist < data_point.distance:
				data_point.update_distance(dist)
				data_point.set_cluster(self.clusters[i])


class DataPoint():
	def __init__(self,cluster=None,value=None, distance=sys.float_info.max):
		self.cluster = cluster
		self.value = value
		self.distance = distance

	def update_distance(self,dist):
		self.distance = dist 

	def set_cluster(self,cluster):
		if self in self.cluster.points if self.cluster else []:
			self.cluster.remove_point(self)
		cluster.add_point(self)
		self.cluster = cluster

	def __str__(self):
		return str(self.value)


class Cluster():
	def __init__(self,points=[],centroid=None,dimensions=2, max_points=None):
		self.points = points
		self.centroid = centroid
		self.dimensions = dimensions
		self.max_points = max_points
		if self.centroid is None:
			self.centroid = np.random.random(size=self.dimensions)

	def remove_point(self,point):
		self.points.remove(point)

	def add_point(self,point):
		if len(self.points) < self.max_points if self.max_points else DEFAULT_MAX:
			self.points.append(point)
			return self.points
		else:
			return None

	def update_centroid(self):
		for v in range(len(self.centroid)):
			tmp_sum = 0.0
			for p in self.points:
				tmp_sum += p.value[v]
			if len(self.points) > 0:
				mean = tmp_sum / float(len(self.points))
			else:
				mean = self.centroid[v]
			self.centroid[v] = mean

	def get_accuracy(self,points):
		score = 0.0
		for p in points:
			if any(p == x.value for x in self.points):
				score += 1.0
		return score / float(len(points))


def verify(clusters,labels,data):
	data_map = {}
	total_accuracy = 0.0
	for i in list(set(labels)):
		data_map[i] = []
	for i in range(len(labels)):
		data_map[labels[i]].append(data[i])
	for group in data_map:
		current_accuracy = 0.0
		for c in clusters:
			tmp_cluster_accuracy = c.get_accuracy(data_map[group])
			if tmp_cluster_accuracy == 0.0:
				continue
			if tmp_cluster_accuracy > current_accuracy:
				current_accuracy = tmp_cluster_accuracy
			else:
				current_accuracy = (current_accuracy + tmp_cluster_accuracy) / 2.0 
		total_accuracy += current_accuracy
	return total_accuracy / len(data_map)


def main():
	np.random.seed(1337)
	## Read in data 
	dataset_file = input("Path to dataset: ")
	passes = int(input("Number of passes (integer): "))
	labels = []
	data = []
	with open(dataset_file,newline='') as csvfile:
		spamreader = csv.reader(csvfile)
		for row in spamreader:
			if "wine" in dataset_file:
				labels.append(int(row[0]))
			elif "iris" in dataset_file:
				lab = "".join(row[-1:])
				labels.append(lab)
			tmp_data = []
			if "wine" in dataset_file:
				for item in row[1:]:
					if item[0] == '.':
						item = '0'+item
					tmp_data.append(float(item))	
				data.append(tmp_data)
			elif "iris" in dataset_file:
				for item in row[:-1]:
					if item[0] == '.':
						item = '0'+item
					tmp_data.append(float(item))	
				data.append(tmp_data)
	labels = [x for x in labels if x]
	data = [x for x in data if x]
	num_clusters = len(set(labels))
	
	# Run K-Means setup and algorithm until converged
	km = KMeans(data,num_clusters,cluster_size=(len(data)/num_clusters)+(len(data)/(num_clusters+SCALER)))
	clusters = km.k_means()
	
	# Verify the results against the labels
	accuracy = verify(clusters,labels,data)
	print(f"Accuracy: {accuracy*100}%")
	count = 0
	while accuracy < 0.9 and count < passes:
		# Run K-Means setup and algorithm until converged
		km.set_starting()
		clusters = km.k_means()
		
		# Verify the results against the labels
		accuracy = verify(clusters,labels,data)
		print(f"Accuracy: {accuracy*100}%")
		count+=1


if __name__ == "__main__":
	try:
		main()
	finally:
		print("Goodbye")
