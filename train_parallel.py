import argparse
import os
import shutil
import threading
import time

from master import SimpleMaster
import environments


def start(env, gpu):
    env_name = env

    if not os.path.exists('logs'):
        os.mkdir('logs')

    if not os.path.exists('models'):
        os.mkdir('models')

    try:
        shutil.rmtree("logs/" + env_name)
    except:
        pass

    env_producer = environments.EnvironmentProducer(env_name, gpu)
    master = SimpleMaster(env_producer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Parallel PPO'))
    parser.add_argument('-env', type=str, help='Env name')
    parser.add_argument('-gpu', action='store_true')
    args = parser.parse_args()

    start(**vars(args))
