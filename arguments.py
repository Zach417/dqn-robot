import argparse

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--env-name', type=str, default='DqnRobot')
    parser.add_argument('--weights', type=str, default=None)
    return parser.parse_args()
