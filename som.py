import numpy as np

class SOM:
    """
    Self-Organizing Map (SOM) implementation using NumPy.
    """
    def __init__(self, x, y, input_len, learning_rate=0.5, radius=None, num_epochs=100):
        self.x = x  # width of SOM grid
        self.y = y  # height of SOM grid
        self.input_len = input_len  # number of input features
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Initial neighborhood radius
        self.radius = radius if radius else max(x, y) / 2

        # Time constant for decay of radius
        self.time_constant = self.num_epochs / np.log(self.radius)

        # Initialize weights randomly
        self.weights = np.random.rand(x, y, input_len)

        # For mapping data to neurons
        self.neuron_locations = np.array(list(self._neuron_locations()))
        self.errors = []

    def _neuron_locations(self):
        """
        Generator for coordinates of each neuron in the SOM grid.
        """
        for i in range(self.x):
            for j in range(self.y):
                yield np.array([i, j])

    def _find_bmu(self, sample):
        """
        Find the Best Matching Unit (BMU) for a given input sample.
        """
        bmu_idx = None
        min_dist = np.inf

        for loc in self.neuron_locations:
            w = self.weights[loc[0], loc[1], :]
            dist = np.linalg.norm(sample - w)
            if dist < min_dist:
                min_dist = dist
                bmu_idx = loc

        return bmu_idx, min_dist

    def _decay_radius(self, epoch):
        return self.radius * np.exp(-epoch / self.time_constant)

    def _decay_learning_rate(self, epoch):
        return self.learning_rate * np.exp(-epoch / self.num_epochs)

    def train(self, data):
        """
        Train the SOM on the input data.
        """
        for epoch in range(self.num_epochs):
            total_error = 0
            for sample in data:
                bmu, error = self._find_bmu(sample)
                total_error += error
                radius = self._decay_radius(epoch)
                lr = self._decay_learning_rate(epoch)
                for loc in self.neuron_locations:
                    dist_to_bmu = np.linalg.norm(loc - bmu)
                    if dist_to_bmu <= radius:
                        influence = np.exp(-(dist_to_bmu ** 2) / (2 * (radius ** 2)))
                        delta = lr * influence * (sample - self.weights[loc[0], loc[1], :])
                        self.weights[loc[0], loc[1], :] += delta
            self.errors.append(total_error)

    def map_input(self, data):
        """
        Map each input sample to its BMU.
        """
        return [self._find_bmu(x)[0] for x in data]
