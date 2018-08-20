import tensorflow as tf
import argparse

import tensorflow as tf

import environments
from agent import PPOAgent
from policy import *


def print_summary(ep_count, rew):
    print("Episode: %s. Reward: %s" % (ep_count, rew))


def start(env):
    MASTER_NAME = "master-0"

    tf.reset_default_graph()

    with tf.Session() as session:
        with tf.variable_scope(MASTER_NAME) as scope:
            env_opts = environments.get_env_options(env, False)
            policy = get_policy(env_opts, session)
            master_agent = PPOAgent(policy, session, MASTER_NAME, env_opts)
        saver = tf.train.Saver(max_to_keep=1)
        saver = tf.train.import_meta_graph(tf.train.latest_checkpoint("models/%s/" % env) + ".meta")
        saver.restore(session, tf.train.latest_checkpoint("models/%s/" % env))
        try:
            pass
        except:
            print("Failed to restore model, starting from scratch")
            session.run(tf.global_variables_initializer())

        producer = environments.EnvironmentProducer(env, False)
        env = producer.get_new_environment()
        while True:
            terminal = False
            s0 = env.reset()
            cum_rew = 0
            cur_hidden_state = master_agent.get_init_hidden_state()
            episode_count = 0
            while not terminal:
                episode_count += 1
                env.render()
                action, h_out = master_agent.get_strict_sample(s0, cur_hidden_state)
                cur_hidden_state = h_out
                s0, r, terminal, _ = env.step(action)
                cum_rew += r
            print(episode_count, cum_rew)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Parallel PPO'))
    parser.add_argument('-env', type=str, help='Env name')
    args = parser.parse_args()

    start(**vars(args))
