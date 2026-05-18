import numpy as np

class HopfieldNetwork():
    
    def __init__(self, n):
        self.W = np.zeros(n, n)
        self.x = np.zeros(n)
        self.ths = np.zeros(n)
        self.n_cells = n
        self.n_patterns = 0
        self.rng = np.random.default_rng()
    
    def sign(x):
        if (x < 0):
            return -1
        if (x >= 0):
            return 1
    
    def learn(self, x):
        for x_ in x:
            self.n_patterns += 1
            n = self.n_patterns
            self.W = self.W * (n - 1) / n + np.dot(x_.T, x_) / n
        for i in range(self.n_cells):
            self.W[i][i] = 0
            
    def reset(self):
        self.n_patterns = 0
        self.W = np.zeros(self.n_cells, self.n_cells)
    
    def set(self, x):
        self.x = x
    
    def recall(self, n=1):
        for i in range(n):
            id = self.rng.integers(n)
            self.x[id] = self.sign(np.dot(self.x, self.W[:][id]) - self.ths[i])
        return self.x
            
    def show(self):
        return self.x