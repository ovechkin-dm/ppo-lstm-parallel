import numpy as np

from agent import PPOAgent
from policy import get_policy
import utils


class GatheringWorker:
    def __init__(self, idx, env_producer, env_opts, rollout_size,
                 worker_queue, weights_queue):
        self.session = None
        self.idx = idx
        self.env_producer = env_producer
        self.env = None
        self.s0 = None
        self.trainable_vars = None
        self.agent = None
        self.env_opts = env_opts
        self.cur_hidden_state = None
        self.episode = None
        self.episodes = []
        self.batch_size = env_opts["batch_size"]
        self.terminal = False
        self.recurrent_policy = env_opts["recurrent"]
        self.timestep_size = env_opts["timestep_size"]
        if not self.recurrent_policy:
            self.timestep_size = 1
        self.discount_factor = env_opts["discount_factor"]
        self.gae_factor = env_opts["gae_factor"]
        self.max_episode_steps = env_opts["max_episode_steps"]
        self.rollout_size = rollout_size
        self.discrete_env = env_opts["discrete"]
        self.ep_count = 0
        self.episode_step = 0
        self.cum_rew = 0
        self.global_step = 0
        self.sampled_action = None
        self.sampled_a_prob = None
        self.accum_vars = None
        self.assign_op = None
        self.worker_queue = worker_queue
        self.weights_queue = weights_queue
        self.stats = []
        self.get_experience()

    def get_experience(self):
        self.init()
        action, a_prob, h_out, v_out = self.agent.get_sample(self.s0, self.cur_hidden_state)
        self.sampled_action = action
        self.sampled_a_prob = a_prob
        while True:
            self.stats = []
            self.apply_weights()
            self.episodes = []
            for i in range(self.rollout_size):
                if self.terminal:
                    if self.episode_step == self.max_episode_steps and len(self.episode[1]) > 0:
                        self.episode[4][-1] = False
                    self.episode_step = 0
                    self.s0 = self.env.reset()
                    self.episodes.append(self.episode)
                    self.cur_hidden_state = self.agent.get_init_hidden_state()
                    self.episode = [self.s0], [], [], [], [], [self.cur_hidden_state], []
                    self.stats.append({
                        "reward": self.cum_rew,
                        "step": self.ep_count,
                        "a_probs": self.sampled_a_prob,
                        "picked_a": self.sampled_action,
                        "a_dim": self.env_opts["action_dim"],
                        "discrete": self.env_opts["discrete"]
                    })
                    self.terminal = False
                    self.ep_count += 1
                    self.cum_rew = 0

                action, a_prob, h_out, v_out = self.agent.get_sample(self.s0, self.cur_hidden_state)
                self.episode_step += 1
                self.global_step += 1
                if np.random.random() > 0.99:
                    self.sampled_action = action
                    self.sampled_a_prob = a_prob
                self.cur_hidden_state = h_out
                self.s0, r, self.terminal, _ = self.env.step(action)
                self.cum_rew += r
                self.episode[0].append(self.s0)
                self.episode[1].append(self.agent.transform_reward(r))
                self.episode[2].append(action)
                self.episode[3].append(a_prob)
                self.episode[4].append(self.terminal)
                self.episode[5].append(h_out)
                self.episode[6].append(v_out)
            self.episodes.append(self.episode)
            self.episode = [self.s0], [], [], [], [], [self.cur_hidden_state], []
            result = self.process_episodes(self.episodes)
            self.worker_queue.put(result)

    def apply_weights(self):
        weights = self.weights_queue.get()
        feed_dict = {}
        for i, t in enumerate(self.accum_vars):
            feed_dict[t] = weights[i]
        self.session.run(self.assign_op, feed_dict=feed_dict)

    def init(self):
        import tensorflow as tf
        self.env = self.env_producer.get_new_environment()
        self.s0 = self.env.reset()
        self.session = utils.create_session(self.env_opts, False)
        with tf.device("/cpu:0"):
            with tf.variable_scope("gather-%s" % self.idx):
                pol = get_policy(self.env_opts, self.session)
                self.agent = PPOAgent(pol, self.session, "gather-%s" % self.idx, self.env_opts)
                self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "gather-%s" % self.idx)
                self.accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in
                                   self.trainable_vars]
                assign_ops = [self.trainable_vars[i].assign(self.accum_vars[i]) for i in
                              range(len(self.trainable_vars))]
                self.assign_op = tf.group(assign_ops)
            self.session.run(tf.global_variables_initializer())
            self.cur_hidden_state = self.agent.get_init_hidden_state()
            self.episode = [self.s0], [], [], [], [], [self.cur_hidden_state], []

    def process_episodes(self, episodes):
        all_states = []
        all_advantages = []
        all_returns = []
        all_picked_actions = []
        all_old_actions_probs = []
        all_pred_values = []
        all_hidden_states = []

        for episode in episodes:
            st, rewards, picked_actions, old_action_probs, terminals, hidden_states, values = episode
            if len(rewards) == 0:
                continue
            states = np.asarray(st)
            pred_values = np.zeros(len(values) + 1)
            pred_values[:-1] = np.array(values)
            episode_len = len(rewards)
            advantages = np.zeros((episode_len,))
            returns = np.zeros((episode_len + 1,))
            if terminals[-1]:
                pred_values[-1] = 0
            else:
                _, _, _, v_out = self.agent.get_sample(states[-1], hidden_states[-1])
                pred_values[-1] = v_out
            returns[-1] = pred_values[-1]
            for i in reversed(range(episode_len)):
                r = rewards[i]
                next_v = pred_values[i + 1]
                cur_v = pred_values[i]
                diff = r + self.discount_factor * next_v - cur_v
                if i == episode_len - 1:
                    advantages[i] = diff
                else:
                    advantages[i] = diff + self.discount_factor * self.gae_factor * advantages[i + 1]
                returns[i] = r + self.discount_factor * returns[i + 1]
            returns = returns[:-1]

            ep_states = states[:-1]
            ep_advantages = advantages
            ep_returns = returns
            ep_picked_actions = np.array(picked_actions)
            ep_old_action_probs = np.array(old_action_probs)
            ep_all_pred_values = pred_values
            ep_hidden_state = np.array(hidden_states[:-1])
            splitted = utils.split_episode(ep_states, ep_advantages, ep_returns, ep_picked_actions, ep_old_action_probs,
                                     ep_all_pred_values, ep_hidden_state, self.timestep_size)
            for b_states, b_hidden_state, b_advantages, b_returns, b_picked_actions, b_old_action_probs, b_all_pred_values in splitted:
                all_states.append(b_states)
                all_advantages.append(b_advantages)
                all_returns.append(b_returns)
                all_picked_actions.append(b_picked_actions)
                all_old_actions_probs.append(b_old_action_probs)
                all_pred_values.append(b_all_pred_values)
                all_hidden_states.append(b_hidden_state)

        all_states = np.array(all_states)
        all_advantages = np.array(all_advantages)
        all_picked_actions = np.array(all_picked_actions)
        all_returns = np.array(all_returns)
        all_old_actions_probs = np.array(all_old_actions_probs)
        all_pred_values = np.array(all_pred_values)
        all_hidden_states = np.array(all_hidden_states)

        return [
            all_states,
            all_advantages,
            all_picked_actions,
            all_returns,
            all_old_actions_probs,
            all_pred_values,
            all_hidden_states,
            self.ep_count,
            self.stats,
            self.idx
        ]
