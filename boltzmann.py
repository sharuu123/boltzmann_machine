from __future__ import print_function
import numpy as np

class CreateBM:
	def __init__(self):
		self.b = 2 * np.random.random_sample((6, 1)) - 1
		self.w = 2 * np.random.random_sample((6, 6)) - 1
		self.w = 0.5*(self.w + self.w.transpose()) 		# Make w symmetric
		for i in range(0,self.w.shape[0]):
			self.w[i,i]=0
		print(self.w)
		# print(self.b)

	def getWeights(self):
		return self.w

	def GenerateSamples(self, num_samples, max_epochs=1000):
		# Set all units to random binary states
		samples = np.random.random_integers(0,1,size=(num_samples,6))
		# print(samples)
		# Update all units until we reach thermal equilibrium
		for i in range(1,num_samples):
			nodes = samples[i-1,:] # pick each starting random configuration
			for epoch in range(max_epochs):
				node = np.random.random_integers(0,5,1)
				node_activation = self.b[node,0] + np.dot(nodes, self.w[:,node])
				node_probability = self._logistic(node_activation)
				node_state = node_probability > np.random.rand()
				nodes[node] = node_state
			# Once thermal equilibrium is reached, put back the configuration
			samples[i-1,:] = nodes
		return samples

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
		# print(self.w)
		# print(self.b)

	def train(self, data, orig_w, max_epochs = 100):
		data_size = data.shape[0]

		for epoch in range(max_epochs):

			# Positive Phase

			# For each training pattern
			s = np.zeros((self.num_total, self.num_total))
			nodes = np.zeros((1,self.num_total))
			for i in range(1,data_size+1):
				nodes = data[i-1,:]
				nodes = nodes[None,:]
				s += np.dot(nodes.T, nodes)
			pplus = s/data_size
			print(pplus)

			# Negative Phase

			# Randomly set all the units to binary states
			s = np.zeros((self.num_total, self.num_total))
			for i in range(1,data_size+1):
				nodes = np.random.random_integers(0,1,6)
				for inner_epoch in range(1000):
					node = np.random.random_integers(0,5,1)
					node_activation = self.b[node,0] + np.dot(nodes, self.w[:,node])
					node_probability = self._logistic(node_activation)
					node_state = node_probability > np.random.rand()
					nodes[node] = node_state
				s += np.dot(nodes.T, nodes)
				# print(nodes)
			pminus = s/data_size
			print(pminus)
			
			# Update Weights

			self.w += self.learning_rate * (pplus - pminus)
			for i in range(0,self.w.shape[0]):
				self.w[i,i]=0
			error = np.sum((orig_w - self.w) ** 2)
			print("Epoch %s: error is %s" % (epoch, error))

	def _logistic(self, x):
		return 1.0 / (1 + np.exp(-x))

if __name__ == '__main__':
	c = CreateBM() # Generate Data
	orig_w = c.getWeights()
	samples = c.GenerateSamples(num_samples = 20, max_epochs = 5000)
	print(samples)
	r = BM(num_visible = 6, num_hidden = 0)
	r.train(samples, orig_w, max_epochs = 10)
	print(r.w)
	# user = np.array([[0,0,0,1,1,0]])
	# print(r.run_visible(user))																																									