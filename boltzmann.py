from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

class CreateBM:
	def __init__(self, num_total):
		self.num_total = num_total
		self.b = 2 * np.random.random_sample((self.num_total, 1)) - 1
		self.w = 2 * np.random.random_sample((self.num_total, self.num_total)) - 1
		self.w = 0.5*(self.w + self.w.transpose()) 		# Make w symmetric
		for i in range(0,self.w.shape[0]):
			self.w[i,i]=0
		print(self.w)
		# print(self.b)

	def getWeights(self):
		return self.w

	def getBiasWeights(self):
		return self.b

	def getSamples(self, visible):
		return self.samples[:,0:visible]

	def GenerateSamples(self, num_samples, max_epochs=1000):
		# Set all units to random binary states
		self.samples = np.random.random_integers(0,1,size=(num_samples,self.num_total))
		# print(samples)
		# Update all units until we reach thermal equilibrium
		for i in range(1,num_samples):
			nodes = self.samples[i-1,:] # pick each starting random configuration
			for epoch in range(max_epochs):
				node = np.random.random_integers(0,self.num_total-1,1)
				node_activation = self.b[node,0] + np.dot(nodes, self.w[:,node])
				node_probability = self._logistic(node_activation)
				node_state = node_probability > np.random.rand()
				nodes[node] = node_state
			# Once thermal equilibrium is reached, put back the configuration
			self.samples[i-1,:] = nodes
			# print('.', end="")
		# return self.samples[:,0:self.num_visible]

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

	def dayDream(self, data_size):
		# Randomly set all the units to binary states
		daydream = np.zeros((data_size, self.num_visible))
		s = np.zeros((self.num_total, self.num_total))
		db = np.zeros((self.num_total, 1))
		for i in range(1,data_size+1):
			nodes = np.random.random_integers(0,1,self.num_total)
			for inner_epoch in range(1000):
				node = np.random.random_integers(0,self.num_total-1,1)
				node_activation = self.b[node,0] + np.dot(nodes, self.w[:,node])
				node_probability = self._logistic(node_activation)
				node_state = node_probability > np.random.rand()
				nodes[node] = node_state
			# nodes = nodes[None,:]
			for j in range(1,self.num_visible):
				daydream[i-1,j-1] = nodes[j-1]
		return daydream

	def train(self, data, orig_w, orig_bw, max_epochs = 100):
		data_size = data.shape[0]
		data_prob = self.find_prob(data)

		daydream = self.dayDream(data_size)
		model_prob = self.find_prob(daydream)
		derror = np.sum((data_prob - model_prob) ** 2)
		# derror = np.sum((data - daydream) ** 2)
		print("derror is %s" % (derror))
		errors = np.zeros((1, max_epochs+1))
		errors[0] = derror
		x = np.zeros((1,max_epochs+1))
		for epoch in range(1,max_epochs+1):

			# Positive Phase

			# For each training pattern
			if self.num_visible == 0:
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

			# For each training pattern
			if self.num_visible != 0:
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

			# Negative Phase

			# Randomly set all the units to binary states
			s = np.zeros((self.num_total, self.num_total))
			db = np.zeros((self.num_total, 1))
			for i in range(1,data_size+1):
				nodes = np.random.random_integers(0,1,self.num_total)
				for inner_epoch in range(max_epochs-epoch):
					node = np.random.random_integers(0,self.num_total-1,1)
					node_activation = self.b[node,0] + np.dot(nodes, self.w[:,node])
					node_probability = self._logistic(node_activation)
					node_state = node_probability > np.random.rand()
					nodes[node] = node_state
				# nodes = nodes[None,:]
				# for j in range(1,self.num_visible):
				# 	daydream[i-1,j-1] = nodes[j-1]
				nodes = nodes[None,:]
				s += np.dot(nodes.T, nodes)
				db += np.dot(self.bias, nodes.T)
			pminus = s/data_size
			bminus = db/data_size
			
			# Update Weights

			self.w += self.learning_rate * (pplus - pminus)
			self.b += self.learning_rate * (bplus - bminus)
			for i in range(0,self.w.shape[0]):
				self.w[i,i]=0
			# error1 = np.sum((orig_w - self.w) ** 2)
			# error2 = np.sum((orig_bw - self.b) ** 2)
			# error = error1 + error2
			# print("Epoch %s: error is %s" % (epoch, error))
			daydream = self.dayDream(data_size)
			model_prob = self.find_prob(daydream)
			derror = np.sum((data_prob - model_prob) ** 2)
			# derror = np.sum((data - daydream) ** 2)
			errors[0,epoch] = derror
			x[0,epoch] = epoch
			print("Epoch %s: derror is %s" % (epoch, derror))
		#plot the graph
		plt.plot(x, errors, 'ro')
		# plt.axis([-1, max_epochs+2, 0, 100])
		plt.show()

	def find_prob(self, data):
		size = data.shape[0]
		s = np.zeros((self.num_visible, self.num_visible))
		t = np.ones((self.num_visible, self.num_visible))
		for i in range(1,size+1):
			nodes = data[i-1,:]
			s += np.dot(nodes.T,nodes)
		return s/size

	def _logistic(self, x):
		return 1.0 / (1 + np.exp(-x))

if __name__ == '__main__':
	# Generate Data
	c = CreateBM(num_total = 6) 
	orig_w = c.getWeights()
	orig_bw = c.getBiasWeights()
	c.GenerateSamples(num_samples = 200, max_epochs = 1000)

	# Train the model assuming we got data from BM with all visible nodes
	samples = c.getSamples(visible = 6)
	r = BM(num_visible = 6, num_hidden = 0, learning_rate = .1)
	r.train(samples, orig_w, orig_bw, max_epochs = 100)
	print(r.w)
	print(orig_w)				

	# Train the model assuming we got data from BM 3 hidden and 3 visisble
	# samples = c.getSamples(visible = 3)
	# r = BM(num_visible = 3, num_hidden = 3, learning_rate = .1)
	# r.train(samples, orig_w, orig_bw, max_epochs = 100)
	# print(r.w)
	# print(orig_w)		

	# # Train the model assuming we got data from BM 2 hidden and 3 visisble
	# samples = c.getSamples(visible = 3)
	# r = BM(num_visible = 3, num_hidden = 2, learning_rate = .1)
	# r.train(samples, orig_w, orig_bw, max_epochs = 10)
	# print(r.w)
	# print(orig_w)

	# # Train the model assuming we got data from BM 4 hidden and 3 visisble
	# samples = c.getSamples(visible = 3)
	# r = BM(num_visible = 3, num_hidden = 4, learning_rate = .1)
	# r.train(samples, orig_w, orig_bw, max_epochs = 10)
	# print(r.w)
	# print(orig_w)																																		