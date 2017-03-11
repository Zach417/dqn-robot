import numpy as np
from space import Space

class ObservationSpace(Space):
    def __init__(self):
        self.n = 84*84*3
        Space.__init__(self,0,84*84*3)
