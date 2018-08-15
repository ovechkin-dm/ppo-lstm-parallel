from reward import *
from utils import BATCH_SIZE, MAX_SEQ_LEN


class PPOAgent:
    def __init__(self, policy, session, worker_name, env_opts):
        import tensorflow as tf
        self.env_opts = env_opts
        self.policy = policy
        self.session = session
        self.reward_transformer = ScalingRewardTransformer(env_opts)
        self.worker_name = worker_name
        self.advantages = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_SEQ_LEN])
        self.returns = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_SEQ_LEN])
        self.picked_actions = self.policy.get_picked_actions_input()
        self.old_actions = self.policy.get_old_actions_input()
        self.old_values = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_SEQ_LEN])
        self.policy_inputs = self.policy.get_state_inputs()
        self.hidden_inputs = self.policy.get_hidden_inputs()
        self.entropy_coef = env_opts["entropy_coef"]
        self.kl_target = env_opts["kl_target"]
        self.use_kl_loss = env_opts["use_kl_loss"]
        self.advantages_flat = tf.reshape(self.advantages, [-1])
        returns_flat = tf.reshape(self.returns, [-1])
        states_flat = tf.reshape(self.policy_inputs, [-1, policy.state_size])
        old_values_flat = tf.reshape(self.old_values, [-1])
        self.value_outputs = self.policy.value_outputs()[:, 0]

        self.clip_eps = tf.placeholder(tf.float32, ())
        self.kl_beta = tf.placeholder(tf.float32, ())
        self.eta = tf.placeholder(tf.float32, ())
        self.grad_step = tf.placeholder(tf.float32, ())

        _, old_values_std = tf.nn.moments(old_values_flat, axes=[0])
        self.mask = tf.sign(tf.reduce_max(tf.abs(states_flat), axis=1))
        self.seq_len = tf.reduce_sum(self.mask)
        self.value_loss = self.reduce_mean(tf.square(self.value_outputs - returns_flat))

        ratio = self.policy.get_current_prob() / (self.policy.get_old_prob() + 1e-4)
        clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        policy_loss_left = self.advantages_flat * clipped_ratio
        policy_loss_right = ratio * self.advantages_flat
        policy_loss_batched = tf.minimum(policy_loss_left, policy_loss_right)
        self.src_policy_loss = -self.reduce_mean(policy_loss_right)
        self.policy_loss_mean = -self.reduce_mean(policy_loss_batched)
        self.entropy = self.reduce_mean(self.policy.entropy())
        self.kl = 0.5 * self.reduce_mean(self.policy.kl())
        self.hinge = self.eta * tf.square(tf.maximum(0.0, self.kl - 2.0 * self.kl_target))

        if self.use_kl_loss:
            self.policy_loss = self.src_policy_loss + self.kl_beta * self.kl + self.hinge - self.entropy_coef * self.entropy
        else:
            self.policy_loss = self.policy_loss_mean - self.entropy_coef * self.entropy

        v_train_ops = self.get_train_op(self.value_loss, 5.0, True, self.grad_step)
        p_train_ops = self.get_train_op(self.policy_loss, 5.0, True, self.grad_step)
        self.v_opt, self.v_optimize, self.v_grads = v_train_ops
        self.p_opt, self.p_optimize, self.p_grads = p_train_ops

    def reduce_mean(self, t):
        import tensorflow as tf
        if self.env_opts["recurrent"]:
            return tf.reduce_sum(t * self.mask) / self.seq_len
        else:
            return tf.reduce_mean(t)

    def get_train_op(self, loss, clip_factor, clip, step):
        import tensorflow as tf
        optimizer = tf.train.AdamOptimizer(learning_rate=step)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        filtered_grads = []
        filtered_vars = []
        for i in range(len(gradients)):
            if gradients[i] is not None:
                filtered_grads.append(gradients[i])
                filtered_vars.append(variables[i])
        gradients = filtered_grads
        variables = filtered_vars
        if clip:
            gradients, _ = tf.clip_by_global_norm(gradients, clip_factor)
        grad_norm = tf.reduce_sum([tf.norm(grad) for grad in gradients])
        train_op = optimizer.apply_gradients(zip(gradients, variables))
        return optimizer, train_op, grad_norm

    def train(self, states, advantages, returns, picked_actions, action_probs, hidden_states,
              old_values, clip_eps, beta, eta, grad_step):
        action_probs = action_probs.reshape([-1, action_probs.shape[-1]])
        picked_actions = picked_actions.reshape([-1, picked_actions.shape[-1]])
        _, _, kl, entropy, hinge, src_policy_loss, \
        vloss, ploss = self.session.run([self.v_optimize,
                                         self.p_optimize,
                                         self.kl,
                                         self.entropy,
                                         self.hinge,
                                         self.src_policy_loss,
                                         self.value_loss,
                                         self.policy_loss], feed_dict={
            self.old_values: old_values,
            self.policy_inputs: states,
            self.advantages: advantages,
            self.returns: returns,
            self.picked_actions: picked_actions,
            self.old_actions: action_probs,
            self.hidden_inputs: hidden_states,
            self.clip_eps: clip_eps,
            self.grad_step: grad_step,
            self.kl_beta: beta,
            self.eta: eta
        })
        return kl, entropy, hinge, src_policy_loss, vloss, ploss

    def get_sample(self, state, hidden_state):
        return self.policy.sample(state, hidden_state)

    def get_strict_sample(self, state, hidden_state):
        return self.policy.strict_sample(state, hidden_state)

    def get_init_hidden_state(self):
        return self.policy.get_initial_state()

    def transform_reward(self, r):
        return self.reward_transformer.transform_reward(r)
