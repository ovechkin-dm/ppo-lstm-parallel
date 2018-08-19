from multiprocessing import Queue, Process

import numpy as np
from threading import Thread
from agent import PPOAgent
from gather import GatheringWorker
from policy import get_policy
import utils
import environments


class Worker:
    def __init__(self, env_producer, idx, master_weights_in_queue, master_weights_out_queue):
        self.env_name = env_producer.get_env_name()
        self.config = environments.get_config(self.env_name)
        self.num_gather_workers = self.config["gather_per_worker"]
        self.env_producer = env_producer
        self.batch_size = self.config["batch_size"]
        self.clip_eps = self.config["clip_eps"]
        self.grad_step = self.config["grad_step"]
        self.epochs = self.config["epochs"]
        self.entropy_coef = self.config["entropy_coef"]
        self.idx = idx
        self.session = None
        self.episode_step = 0
        self.initialized = False
        self.beta = self.config["init_beta"]
        self.eta = self.config["eta"]
        self.kl_target = self.config["kl_target"]
        self.use_kl_loss = self.config["use_kl_loss"]
        self.lr_multiplier = 1.0
        self.prev_batch = None
        self.variables_file_path = "models/%s/variables.txt" % self.env_name
        self.worker_queue = Queue()
        self.weights_queues = [Queue() for _ in range(self.num_gather_workers)]
        self.master_weights_in_queue = master_weights_in_queue
        self.master_weights_out_queue = master_weights_out_queue
        self.init_workers()
        self.agent = None
        self.trainable_vars = None
        self.accum_vars = None
        self.assign_op = None
        self.p_opt_vars = None
        self.v_opt_vars = None
        self.init_agent()

    def init_agent(self):
        import tensorflow as tf
        env_opts = environments.get_env_options(self.env_name, self.env_producer.get_use_gpu())
        self.session = utils.create_session(env_opts, True)
        with tf.variable_scope("worker-%s" % self.idx):
            pol = get_policy(env_opts, self.session)
            self.agent = PPOAgent(pol, self.session, "worker-%s" % self.idx, env_opts)
            self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "worker-%s" % self.idx)
            self.accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in
                               self.trainable_vars]
            p_vars = self.agent.p_opt.variables()
            v_vars = self.agent.v_opt.variables()
            self.p_opt_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in p_vars]
            self.v_opt_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in v_vars]
            p_assign_ops = [p_vars[i].assign(self.p_opt_vars[i]) for i in range(len(p_vars))]
            v_assign_ops = [v_vars[i].assign(self.v_opt_vars[i]) for i in range(len(v_vars))]

            assign_ops = [self.trainable_vars[i].assign(self.accum_vars[i]) for i in
                          range(len(self.trainable_vars))]
            self.assign_op = tf.group(assign_ops + p_assign_ops + v_assign_ops)

        self.session.run(tf.global_variables_initializer())
        self.run()

    def init_workers(self):
        for i in range(self.num_gather_workers):
            rollout_size = self.config["rollout_size"] // self.num_gather_workers
            t = Process(target=make_worker, args=(i, self.env_producer,
                                                  self.worker_queue,
                                                  self.weights_queues[i],
                                                  rollout_size))
            t.start()

    def run(self):
        while True:
            self.apply_shared_variables()
            self.apply_weights_to_gather_workers()
            stats = self.compute_grads_and_stats()
            self.send_to_master(stats)

    def send_to_master(self, stats):
        weights, p_opt_weights, v_opt_weights = self.session.run([self.trainable_vars,
                                                                  self.agent.p_opt.variables(),
                                                                  self.agent.v_opt.variables()])
        arr = [self.beta, self.lr_multiplier, p_opt_weights, v_opt_weights, weights, stats]
        self.master_weights_out_queue.put(arr)

    def apply_weights_to_gather_workers(self):
        weights = self.session.run(self.trainable_vars)
        for q in self.weights_queues:
            q.put(weights)

    def apply_shared_variables(self):
        beta, lr_multiplier, p_opt_weights, v_opt_weights, weights = self.master_weights_in_queue.get()
        self.beta = beta
        self.lr_multiplier = lr_multiplier
        fd = {}
        for i, t in enumerate(self.accum_vars):
            fd[t] = weights[i]
        for i, t in enumerate(self.p_opt_vars):
            fd[t] = p_opt_weights[i]
        for i, t in enumerate(self.v_opt_vars):
            fd[t] = v_opt_weights[i]
        self.session.run(self.assign_op, feed_dict=fd)

    def compute_grads_and_stats(self):
        results = []
        for i in range(self.num_gather_workers):
            results.append(self.worker_queue.get())
        w_idx = list(range(self.num_gather_workers))
        cur_all_states = np.concatenate([results[i][0] for i in w_idx], axis=0)
        cur_all_advantages = np.concatenate([results[i][1] for i in w_idx], axis=0)
        cur_all_picked_actions = np.concatenate([results[i][2] for i in w_idx], axis=0)
        cur_all_returns = np.concatenate([results[i][3] for i in w_idx], axis=0)
        cur_all_old_actions_probs = np.concatenate([results[i][4] for i in w_idx], axis=0)
        cur_all_pred_values = np.concatenate([results[i][5] for i in w_idx], axis=0)
        cur_all_hidden_states = np.concatenate([results[i][6] for i in w_idx], axis=0)

        if self.prev_batch is not None:
            prev_all_states, prev_all_advantages, prev_all_picked_actions, prev_all_returns, \
            prev_all_old_actions_probs, prev_all_pred_values, prev_all_hidden_states = self.prev_batch
            all_states = np.concatenate([cur_all_states, prev_all_states], axis=0)
            all_advantages = np.concatenate([cur_all_advantages, prev_all_advantages], axis=0)
            all_picked_actions = np.concatenate([cur_all_picked_actions, prev_all_picked_actions], axis=0)
            all_returns = np.concatenate([cur_all_returns, prev_all_returns], axis=0)
            all_old_actions_probs = np.concatenate([cur_all_old_actions_probs, prev_all_old_actions_probs], axis=0)
            all_pred_values = np.concatenate([cur_all_pred_values, prev_all_pred_values], axis=0)
            all_hidden_states = np.concatenate([cur_all_hidden_states, prev_all_hidden_states], axis=0)
        else:
            all_states = cur_all_states
            all_advantages = cur_all_advantages
            all_picked_actions = cur_all_picked_actions
            all_returns = cur_all_returns
            all_old_actions_probs = cur_all_old_actions_probs
            all_pred_values = cur_all_pred_values
            all_hidden_states = cur_all_hidden_states

        self.prev_batch = [cur_all_states, cur_all_advantages, cur_all_picked_actions, cur_all_returns,
                           cur_all_old_actions_probs, cur_all_pred_values, cur_all_hidden_states]

        all_advantages = (all_advantages - all_advantages.mean()) / (max(all_advantages.std(), 1e-4))

        first_gather = [x for x in results if x[9] == 0][0]

        self.episode_step = first_gather[7]
        stats = first_gather[8]

        sz = len(all_states)
        n_batches = (sz - 1) // self.batch_size + 1
        steps = 0
        cur_kl = 0
        entropy = 0
        hinge = 0
        src_policy_loss = 0
        vloss = 0
        ploss = 0
        for cur_epoch in range(self.epochs):
            idx = np.arange(len(all_states))
            np.random.shuffle(idx)
            all_states = all_states[idx]
            all_returns = all_returns[idx]
            all_picked_actions = all_picked_actions[idx]
            all_old_actions_probs = all_old_actions_probs[idx]
            all_advantages = all_advantages[idx]
            all_pred_values = all_pred_values[idx]
            all_hidden_states = all_hidden_states[idx]
            for b in range(n_batches):
                start = b * self.batch_size
                end = min(sz, (b + 1) * self.batch_size)
                states_b = all_states[start:end]
                returns_b = all_returns[start:end]
                picked_actions_b = all_picked_actions[start:end]
                old_action_probs_b = all_old_actions_probs[start:end]
                advantages_b = all_advantages[start:end]
                hidden_states_b = all_hidden_states[start:end]
                old_values_b = all_pred_values[start:end]
                cur_kl, entropy, hinge, src_policy_loss, vloss, ploss = \
                    self.agent.train(states_b,
                                     advantages_b,
                                     returns_b,
                                     picked_actions_b,
                                     old_action_probs_b,
                                     hidden_states_b,
                                     old_values_b,
                                     self.clip_eps,
                                     self.beta,
                                     self.eta,
                                     self.grad_step * self.lr_multiplier)
                steps += 1
            if cur_kl > self.kl_target * 4 and self.use_kl_loss:
                break

        if self.use_kl_loss:
            if cur_kl > self.kl_target * 2:
                self.beta = np.minimum(35, 1.5 * self.beta)
                if self.beta > 30.0:
                    self.lr_multiplier /= 1.5
            elif cur_kl < self.kl_target / 2:
                self.beta = np.maximum(1 / 35, self.beta / 1.5)
                if self.beta <= 1 / 30.0:
                    self.lr_multiplier *= 1.5
            self.lr_multiplier = max(min(self.lr_multiplier, 3.0), 0.1)

        train_stats = {
            "stats": stats,
            "kl": cur_kl,
            "entropy": entropy,
            "hinge": hinge,
            "src_policy_loss": src_policy_loss,
            "vloss": vloss,
            "ploss": ploss,
            "lr_multiplier": self.lr_multiplier,
            "beta": self.beta,
            "step": self.episode_step,
            "idx": self.idx
        }
        return train_stats


def make_worker(i, env_producer, worker_queue, weights_queue, rollout_size):
    return GatheringWorker(i, env_producer, rollout_size, worker_queue, weights_queue)
