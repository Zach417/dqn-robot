class Space(object):
    """Abstract model for a space that is used for the state and action spaces. This class has the
    exact same API that OpenAI Gym uses so that integrating with it is trivial.
    """

    def sample(self, seed=None):
        """Uniformly randomly sample a random element of this space.
        """
        return 1

    def contains(self, x):
        """Return boolean specifying if x is a valid member of this space
        """
        return True
