import numpy as np

class Space(object):
    """Abstract model for a space that is used for the state and action spaces. This class has the
    exact same API that OpenAI Gym uses so that integrating with it is trivial.
    """

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def sample(self, seed=None):
        """Uniformly randomly sample a random element of this space.
        """
        return np.random.uniform(self.low, self.high)

    def contains(self, x):
        """Return boolean specifying if x is a valid member of this space
        """
        return x.shape == self.shape and (x >= self.low).all() and (x <= self.high).all()
