from __future__ import print_function
import numpy as np

class CreateBM:
	def __init__(self, num_visible, num_hidden):
		self.num_hidden = num_hidden
		self.num_visible = num_visible
		self.num_total = num_hidden+num_visible
		self.b = 2 * np.random.random_sample((self.num_total, 1)) - 1
		self.w = 2 * np.random.random_sample((self.num_total, self.num_total)) - 1
		self.w = 0.5*(self.w + self.w.transpose()) 		# Make w symmetric
		for i in range(0,self.w.shape[0]):
			self.w[i,i]=0
		# self.w = np.insert(self.w, 0, 0, axis = 0)
		# self.w = np.insert(self.w, 0, 0, axis = 1)
		print(self.w)
		# print(self.b)

	def getWeights(self):
		return self.w

	def getBiasWeights(self):
		return self.b

	def GenerateSamples(self, num_samples, max_epochs=1000):
		# Set all units to random binary states
		samples = np.random.random_integers(0,1,size=(num_samples,self.num_total))
		# print(samples)
		# Update all units until we reach thermal equilibrium
		for i in range(1,num_samples):
			nodes = samples[i-1,:] # pick each starting random configuration
			for epoch in range(max_epochs):
				node = np.random.random_integers(0,self.num_total-1,1)
				node_activation = self.b[node,0] + np.dot(nodes, self.w[:,node])
				node_probability = self._logistic(node_activation)
				node_state = node_probability > np.random.rand()
				nodes[node] = node_state
			# Once thermal equilibrium is reached, put back the configuration
			samples[i-1,:] = nodes
			print('.', end="")
		return samples[:,0:self.num_visible]

	def _logistic(self, x):
		return 1.0 / (1 + np.exp(-x))

class BM:
	def __init__(self, num_visible, num_hidden, learning_rate = 0.1):
		self.num_hidden = num_hidden
		self.num_visible = num_visible
		self.num_total = num_hidden+num_visible
		self.learning_rate = learning_rate
		# Initialize weights
		self.w = 0.1 * np.random.randn(self.num_total, self.num_total)
		self.w = 0.5*(self.w + self.w.T) 		# Make w symmetric
		for i in range(0,self.w.shape[0]):
			self.w[i,i]=0
		self.b = 0.1 * np.random.randn(self.num_total, 1)
		self.bias = np.ones((self.num_total, self.num_total))
		# print(self.w)
		# print(self.b)

	def train(self, data, orig_w, orig_bw, max_epochs = 100):
		data_size = data.shape[0]
		error = np.sum((orig_w - self.w) ** 2)
		print("error is %s" % (error))

		for epoch in range(max_epochs):

			# Positive Phase

			# For each training pattern
			s = np.zeros((self.num_total, self.num_total))
			db = np.zeros((self.num_total, 1))
			nodes = np.zeros((1,self.num_total))
			for i in range(1,data_size+1):
				nodes = data[i-1,:]
				nodes = nodes[None,:]
				s += np.dot(nodes.T, nodes)
				db += np.dot(self.bias, nodes.T)
			pplus = s/data_size
			bplus = db/data_size
			# print(pplus)

			# Negative Phase

			# Randomly set all the units to binary states
			daydream = np.zeros((data_size, 6))
			s = np.zeros((self.num_total, self.num_total))
			db = np.zeros((self.num_total, 1))
			for i in range(1,data_size+1):
				nodes = np.random.random_integers(0,1,6)
				for inner_epoch in range(10):
					node = np.random.random_integers(0,5,1)
					node_activation = self.b[node,0] + np.dot(nodes, self.w[:,node])
					node_probability = self._logistic(node_activation)
					node_state = node_probability > np.random.rand()
					nodes[node] = node_state
				nodes = nodes[None,:]
				daydream[i-1,:] = nodes
				s += np.dot(nodes.T, nodes)
				db += np.dot(self.bias, nodes.T)
			pminus = s/data_size
			bminus = db/data_size
			# print(pminus)
			
			# Update Weights

			self.w += self.learning_rate * (pplus - pminus)
			self.b += self.learning_rate * (bplus - bminus)
			for i in range(0,self.w.shape[0]):
				self.w[i,i]=0
			error1 = np.sum((orig_w - self.w) ** 2)
			error2 = np.sum((orig_bw - self.b) ** 2)
			error = error1 + error2
			print("Epoch %s: error is %s" % (epoch, error))
			derror = np.sum((data - daydream) ** 2)
			print("Epoch %s: derror is %s" % (epoch, derror))

	def trainWithHidden(self, data, orig_w, orig_bw, max_epochs = 100):
		data_size = data.shape[0]
		# error = np.sum((orig_w - self.w) ** 2)
		# print("error is %s" % (error))

		for epoch in range(max_epochs):

			# Positive Phase

			# For each training pattern
			s = np.zeros((self.num_total, self.num_total))
			db = np.zeros((self.num_total, 1))
			for i in range(1,data_size+1):
				# clamp data vector and randomly initialize hidden nodes
				nodes = np.random.random_integers(0,1,self.num_total)
				for j in range(1,self.num_visible):
					nodes[j-1] = data[i-1,j]
				for inner_epoch in range(1000):
					node = np.random.random_integers(0,self.num_total-1,1)
					node_activation = self.b[node,0] + np.dot(nodes, self.w[:,node])
					node_probability = self._logistic(node_activation)
					node_state = node_probability > np.random.rand()
					nodes[node] = node_state
				nodes = nodes[None,:]
				s += np.dot(nodes.T, nodes)
				db += np.dot(self.bias, nodes.T)
			pplus = s/data_size
			bplus = db/data_size
			# print(pplus)

			# Negative Phase

			# Randomly set all the units to binary states
			daydream = np.zeros((data_size, self.num_visible))
			s = np.zeros((self.num_total, self.num_total))
			db = np.zeros((self.num_total, 1))
			for i in range(1,data_size+1):
				nodes = np.random.random_integers(0,1,self.num_total)
				for inner_epoch in range(10):
					node = np.random.random_integers(0,self.num_total-1,1)
					node_activation = self.b[node,0] + np.dot(nodes, self.w[:,node])
					node_probability = self._logistic(node_activation)
					node_state = node_probability > np.random.rand()
					nodes[node] = node_state
				# nodes = nodes[None,:]
				for j in range(1,self.num_visible):
					daydream[i-1,j-1] = nodes[j-1]
				nodes = nodes[None,:]
				s += np.dot(nodes.T, nodes)
				db += np.dot(self.bias, nodes.T)
			pminus = s/data_size
			bminus = db/data_size
			# print(pminus)
			
			# Update Weights

			self.w += self.learning_rate * (pplus - pminus)
			self.b += self.learning_rate * (bplus - bminus)
			for i in range(0,self.w.shape[0]):
				self.w[i,i]=0
			# error1 = np.sum((orig_w - self.w) ** 2)
			# error2 = np.sum((orig_bw - self.b) ** 2)
			# error = error1 + error2
			# print("Epoch %s: error is %s" % (epoch, error))
			derror = np.sum((data - daydream) ** 2)
			print("Epoch %s: derror is %s" % (epoch, derror))

	def _logistic(self, x):
		return 1.0 / (1 + np.exp(-x))

if __name__ == '__main__':
	c = CreateBM(num_visible = 3, num_hidden = 3) # Generate Data
	orig_w = c.getWeights()
	orig_bw = c.getBiasWeights()
	samples = c.GenerateSamples(num_samples = 100, max_epochs = 1000)
	print(samples)
	r = BM(num_visible = 3, num_hidden = 3, learning_rate = 1)
	r.trainWithHidden(samples, orig_w, orig_bw, max_epochs = 50)
	print(r.w)
	print(orig_w)																																							