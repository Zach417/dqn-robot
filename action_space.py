import numpy as np
from space import Space

class ActionSpace(Space):
    def __init__(self):
        self.n = 5
        Space.__init__(self,0,4)
