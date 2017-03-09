import numpy as np

class Robot():
    def __init__(self):
        self.position = np.array([0,0,0])
        self.direction = np.array([0,0,0])

    def update(self):
        self.position = self.position + self.direction
        self.direction = np.array([0,0,0])

    def forward(self):
        self.direction = np.array([0,1,0])

    def backward(self):
        self.direction = np.array([0,-1,0])

    def left(self):
        self.direction = np.array([-1,0,0])

    def right(self):
        self.direction = np.array([1,0,0])
