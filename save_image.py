import gym
import tensorflow as tf
import argparse
import environments
from agent import PPOAgent
import imageio
from policy import *

def print_summary(ep_count, rew):
    print("Episode: %s. Reward: %s" % (ep_count, rew))


def start(env):
    env = gym.make(env)
    frames = []
    MASTER_NAME = "master-0"
    IMAGE_PATH = "images/%s.gif" % env.spec.id
    tf.reset_default_graph()

    with tf.Session() as session:
        with tf.variable_scope(MASTER_NAME) as scope:
            env_opts = environments.get_env_options(env, False)
            policy = get_policy(env_opts, session)
            master_agent = PPOAgent(policy, session, MASTER_NAME, env_opts)

        saver = tf.train.Saver(max_to_keep=1)
        saver = tf.train.import_meta_graph(tf.train.latest_checkpoint("models/%s/" % env.spec.id) + ".meta")
        saver.restore(session, tf.train.latest_checkpoint("models/%s/" % env.spec.id))
        try:
            pass
        except:
            print("Failed to restore model, starting from scratch")
            session.run(tf.global_variables_initializer())

        global_step = 0
        while global_step < 1000:
            terminal = False
            s0 = env.reset()
            cum_rew = 0
            cur_hidden_state = master_agent.get_init_hidden_state()
            episode_count = 0
            while not terminal:
                episode_count += 1
                frames.append(env.render(mode='rgb_array'))
                action, h_out = master_agent.get_strict_sample(s0, cur_hidden_state)
                cur_hidden_state = h_out
                s0, r, terminal, _ = env.step(action)
                cum_rew += r
                global_step += 1
            print(episode_count, cum_rew)
        imageio.mimsave(IMAGE_PATH, frames, duration=1.0 / 60.0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Parallel PPO'))
    parser.add_argument('-env', type=str, help='Env name')
    args = parser.parse_args()

    start(**vars(args))
