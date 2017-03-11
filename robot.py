import numpy as np

class Robot():
    def __init__(self, world_shape):
        self.position = np.array([0,0,0])
        self.direction = np.array([0,0,0])
        self.world_shape = world_shape

    def reset(self):
        self.position = np.array([0,0,0])
        self.direction = np.array([0,0,0])

    def update(self):
        inX = (self.position[0] > -self.world_shape[0] and self.position[0] < self.world_shape[0])
        inY = (self.position[1] > -self.world_shape[1] and self.position[0] < self.world_shape[1])
        if (inX and inY):
            self.position = self.position + self.direction

        self.direction = np.array([0,0,0])

    def executeAction(self, action):
        if (action == 1):
            self.forward()
        elif (action == 2):
            self.backward()
        elif (action == 3):
            self.left()
        elif (action == 4):
            self.right()

    def forward(self):
        self.direction = np.array([0,1,0])

    def backward(self):
        self.direction = np.array([0,-1,0])

    def left(self):
        self.direction = np.array([-1,0,0])

    def right(self):
        self.direction = np.array([1,0,0])
