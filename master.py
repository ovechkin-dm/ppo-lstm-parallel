from multiprocessing import Queue, Process

from threading import Thread
import numpy as np

import utils
from agent import PPOAgent
from policy import get_policy
from worker import Worker
import environments


class SimpleMaster:
    def __init__(self, env_producer):
        self.env_name = env_producer.get_env_name()
        self.config = environments.get_config(self.env_name)
        self.worker_size = self.config["worker_num"]
        self.env_producer = env_producer
        self.queues = []
        self.w_in_queue = Queue()
        self.init_workers()
        self.session = None
        self.trainable_vars = None
        self.accum_vars = None
        self.p_opt_vars = None
        self.v_opt_vars = None
        self.assign_op = None
        self.agent = None
        self.saver = None
        self.summary_writer = None
        self.beta = 1
        self.lr_multiplier = 1.0
        self.iter_count = 1
        self.variables_file_path = "models/%s/variables.txt" % self.env_name
        self.model_path = "models/%s/model" % self.env_name
        self.initialized = False
        self.cur_step = -1
        self.start()

    def init_workers(self):
        for i in range(self.worker_size):
            q = Queue()
            self.queues.append(q)
            t = Process(target=make_worker, args=(self.env_producer, i, q, self.w_in_queue))
            t.start()

    def start(self):
        import tensorflow as tf
        env_opts = environments.get_env_options(self.env_name, self.env_producer.get_use_gpu())
        self.summary_writer = tf.summary.FileWriter("logs/%s" % self.env_name)
        self.session = utils.create_session(env_opts, True)
        with tf.variable_scope("master-0"):
            pol = get_policy(env_opts, self.session)
            self.agent = PPOAgent(pol, self.session, "master-0", env_opts)
            self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "master-0")
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

        self.restore_variables()
        self.saver = tf.train.Saver(max_to_keep=1)
        self.session.run(tf.global_variables_initializer())
        try:
            self.saver = tf.train.import_meta_graph(
                tf.train.latest_checkpoint("models/%s/" % env_opts["env_name"]) + ".meta")
            self.saver.restore(self.session,
                               tf.train.latest_checkpoint("models/%s/" % env_opts["env_name"]))
        except:
            print("failed to restore model")

        while True:
            if self.iter_count % 10 == 0:
                print("Saving model...")
                self.save_variables()
                self.saver.save(self.session, self.model_path, self.iter_count)
                print("Model saved")
            self.broadcast_weights()
            self.merge_weights()
            self.iter_count += 1

    def restore_variables(self):
        try:
            lines = open(self.variables_file_path).readlines()
            result = {}
            for l in lines:
                a, b = l.split("=")
                b = b.strip()
                result[a] = b
            self.iter_count = int(result["global_step"]) + 1
            self.beta = float(result["beta"])
            self.lr_multiplier = float(result["lr_multiplier"])
        except:
            print("failed to restore variables")

    def save_variables(self):
        f = open(self.variables_file_path, "w")
        lines = []
        lines.append("global_step=%s\n" % self.iter_count)
        lines.append("beta=%s\n" % self.beta)
        lines.append("lr_multiplier=%s\n" % self.lr_multiplier)
        f.writelines(lines)
        f.close()

    def broadcast_weights(self):
        weights, p_opt_weights, v_opt_weights = self.session.run([self.trainable_vars,
                                                                  self.agent.p_opt.variables(),
                                                                  self.agent.v_opt.variables()])
        arr = [self.beta, self.lr_multiplier, p_opt_weights, v_opt_weights, weights]
        for q in self.queues:
            q.put(arr)

    def merge_weights(self):
        results = []
        for i in range(self.worker_size):
            results.append(self.w_in_queue.get())
        self.beta = np.mean([x[0] for x in results])
        self.lr_multiplier = np.mean([x[1] for x in results])
        p_opt_weights = self.make_means([x[2] for x in results])
        v_opt_weights = self.make_means([x[3] for x in results])
        weights = self.make_means([x[4] for x in results])
        first_worker = [x for x in results if x[5]["idx"] == 0][0]
        self.record_stats(first_worker[5])
        fd = {}
        for i, t in enumerate(self.accum_vars):
            fd[t] = weights[i]
        for i, t in enumerate(self.p_opt_vars):
            fd[t] = p_opt_weights[i]
        for i, t in enumerate(self.v_opt_vars):
            fd[t] = v_opt_weights[i]
        self.session.run(self.assign_op, feed_dict=fd)

    def make_means(self, weights):
        result = []
        for i in range(len(weights[0])):
            acc = []
            for j in range(len(weights)):
                acc.append(weights[j][i])
            acc = np.mean(acc, axis=0)
            result.append(acc)
        return result

    def record_stats(self, stats):
        if self.cur_step == stats["step"]:
            return
        self.cur_step = stats["step"]
        self.record_losses(stats["kl"], stats["entropy"], stats["hinge"], stats["src_policy_loss"],
                           stats["vloss"], stats["ploss"], stats["step"])
        cum_rew = 0
        for s in stats["stats"]:
            self.log_summary(s["reward"], s["step"], s["a_probs"], s["picked_a"], s["a_dim"], s["discrete"])
            cum_rew += s["reward"]
        cum_rew /= max(1, len(stats["stats"]))
        print("Average reward: %s" % cum_rew)

    def record_losses(self, cur_kl, entropy, hinge, src_policy_loss, vloss, ploss, step):
        import tensorflow as tf
        summary = tf.Summary()
        summary.value.add(tag='Losses/value_loss', simple_value=vloss)
        summary.value.add(tag='Losses/policy_loss', simple_value=ploss)
        summary.value.add(tag='Losses/kl_divergence', simple_value=cur_kl)
        summary.value.add(tag='Losses/entropy', simple_value=entropy)
        summary.value.add(tag='Losses/src_policy_loss', simple_value=src_policy_loss)
        summary.value.add(tag='Losses/hinge', simple_value=hinge)
        summary.value.add(tag='Vars/beta', simple_value=self.beta)
        summary.value.add(tag='Vars/lr_multiplier', simple_value=self.lr_multiplier)
        self.summary_writer.add_summary(summary, step)
        self.summary_writer.flush()

    def log_summary(self, reward, step, a_probs, picked_a, a_dim, discrete):
        import tensorflow as tf
        summary = tf.Summary()
        summary.value.add(tag='Reward/per_episode', simple_value=float(reward))
        if not discrete:
            for i in range(a_dim):
                prefix = "Action" + str(i)
                summary.value.add(tag=prefix + '/mean', simple_value=float(a_probs[i]))
                summary.value.add(tag=prefix + "/std", simple_value=float(a_probs[i + a_dim]))
                summary.value.add(tag=prefix + '/picked', simple_value=float(picked_a[i]))
        else:
            for i in range(a_dim):
                prefix = "Action" + str(i)
                summary.value.add(tag=prefix + '/prob', simple_value=float(a_probs[i]))
            summary.value.add(tag='Action/picked', simple_value=float(picked_a))
        self.summary_writer.add_summary(summary, step)
        self.summary_writer.flush()


def make_worker(env_producer, i, q, w_in_queue):
    return Worker(env_producer, i, q, w_in_queue)
