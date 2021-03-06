import sys
import os
import numpy as np
from robot import Robot
from space import Space
from action_space import ActionSpace
from observation_space import ObservationSpace

def getObservation(robot, token):
    pos = robot.position
    observation = np.zeros((84,84,3), dtype=np.uint8)
    observation[pos[0]+42][pos[1]+42] = [255, 255, 255] # robot
    observation[token[0]+42][token[1]+42] = [100, 100, 100] # token
    return observation

class Env(object):
    """The abstract environment class that is used by all agents. This class has the exact
    same API that OpenAI Gym uses so that integrating with it is trivial. In contrast to the
    OpenAI Gym implementation, this class only defines the abstract methods without any actual
    implementation.
    """
    def __init__(self):
        self.robot = Robot((84, 84))
        self.token = np.array([14, -5, 0])
        self.previousAction = 0
        self.iteration = 0

    reward_range = (-1, 1)
    action_space = ActionSpace()
    observation_space = ObservationSpace()

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        self.previousAction = action
        pos1 = self.robot.position
        self.robot.executeAction(action)
        pos2 = self.robot.position + self.robot.direction

        self.iteration += 1
        isDone = self.iteration > 50 or np.array_equal(self.robot.position, self.token)

        reward = 0
        if self.iteration > 50:
            reward = -1
        elif isDone == True:
            reward = 1
        elif (np.linalg.norm(self.token - pos1) > np.linalg.norm(self.token - pos2)):
            reward = 0.25
        else:
            reward = -0.25

        self.robot.update()

        observation = getObservation(self.robot, self.token)
        return observation, reward, isDone, {'info':'test'}

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        Returns:
            observation (object): the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self.robot.reset()
        self.iteration = 0
        return np.zeros((84,84,3), dtype=np.uint8)

    def render(self, mode='human', close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
            close (bool): close all open renderings
        """
        text = str(self.iteration) + " "
        text += "Robot: " + str(self.robot.position) + "; "
        text += "Token " + str(self.token) + "; "
        text += "Action " + str(self.previousAction) + ";"
        sys.stdout.write('\r' + str(text) + ' ' * 20)
        sys.stdout.flush()

    def close(self):
        """Override in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        return 1

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        return [1]

    def configure(self, *args, **kwargs):
        """Provides runtime configuration to the environment.
        This configuration should consist of data that tells your
        environment how to run (such as an address of a remote server,
        or path to your ImageNet data). It should not affect the
        semantics of the environment.
        """
        return 1

    def __del__(self):
        self.close()

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)
