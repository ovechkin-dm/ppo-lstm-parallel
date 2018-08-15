import argparse
import os
import shutil
import threading
import time

import gym
from master import SimpleMaster
import environments


def start(env, gpu):
    env = gym.make(env)

    env_name = env.spec.id

    if not os.path.exists('logs'):
        os.mkdir('logs')

    if not os.path.exists('models'):
        os.mkdir('models')

    try:
        shutil.rmtree("logs/" + env_name)
    except:
        pass

    env_producer = environments.EnvironmentProducer(env.spec.id)
    env_opts = environments.get_env_options(env, gpu)
    worker_num = env_opts["worker_num"]
    gather_per_worker = env_opts["gather_per_worker"]
    master = SimpleMaster(worker_num, gather_per_worker, env_opts, env_producer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Parallel PPO'))
    parser.add_argument('-env', type=str, help='Env name')
    parser.add_argument('-gpu', action='store_true')
    args = parser.parse_args()

    start(**vars(args))
