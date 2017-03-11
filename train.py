import model
import arguments
import numpy as np
import keras.backend as K
from env import Env
from envProcessor import EnvProcessor
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

args = arguments.getArgs()
env = Env()
np.random.seed(123)
env.seed(123)

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
processor = EnvProcessor(INPUT_SHAPE)
model = model.getModel(INPUT_SHAPE, WINDOW_LENGTH, env.action_space.n)

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=1000000)
dqn = DQNAgent(model=model, nb_actions=env.action_space.n, policy=policy, memory=memory, processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=1000, train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=.001), metrics=['mae'])

if args.mode == 'train':
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    checkpoint_weights_filename = 'dqn_' + args.env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(args.env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    #dqn.load_weights(weights_filename)
    dqn.fit(env, callbacks=callbacks, nb_steps=1000, log_interval=100)
    dqn.save_weights(weights_filename, overwrite=True)
    dqn.test(env, nb_episodes=10, visualize=True)
elif args.mode == 'test':
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True)
