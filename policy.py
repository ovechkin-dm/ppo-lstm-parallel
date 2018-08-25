from utils import *


class Policy:
    def __init__(self, action_size, state_size):
        self.action_size = action_size
        self.state_size = state_size

    def get_state_inputs(self):
        pass

    def get_hidden_inputs(self):
        pass

    def value_outputs(self):
        pass

    def sample(self, state, hidden_state):
        pass

    def get_old_actions_input(self):
        pass

    def get_picked_actions_input(self):
        pass

    def get_old_prob(self):
        pass

    def get_current_prob(self):
        pass

    def entropy(self):
        pass

    def strict_sample(self, state, hidden_state):
        pass

    def kl(self):
        pass

    def get_initial_state(self):
        pass

    def get_prob_ratio(self):
        return (self.get_current_prob() + 1e-8) / (self.get_old_prob() + 1e-8)


class LstmContinousPolicy(Policy):
    def __init__(self, session, env_opts):
        import tensorflow as tf
        super().__init__(env_opts["action_dim"], env_opts["state_dim"])
        state_size = env_opts["state_dim"]
        hidden_layer_size = env_opts["hidden_layer_size"]
        self.state_inputs = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_SEQ_LEN, state_size])

        p_lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size, state_is_tuple=True, name="policy_rnn")
        v_lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size, state_is_tuple=True, name="value_rnn")
        p_cell_size = p_lstm_cell.state_size.c + p_lstm_cell.state_size.h
        v_cell_size = v_lstm_cell.state_size.c + v_lstm_cell.state_size.h
        self.init_hidden_state = np.zeros((p_cell_size + v_cell_size), np.float32)
        self.hidden_state_in = tf.placeholder(tf.float32, [BATCH_SIZE, p_cell_size + v_cell_size])

        p_c_in = self.hidden_state_in[:, 0:p_lstm_cell.state_size.c]
        p_h_in = self.hidden_state_in[:, p_lstm_cell.state_size.c:p_cell_size]

        v_c_in = self.hidden_state_in[:, p_cell_size:(p_cell_size + v_lstm_cell.state_size.c)]
        v_h_in = self.hidden_state_in[:, (p_cell_size + v_lstm_cell.state_size.c):]

        p_state_tuple = tf.contrib.rnn.LSTMStateTuple(p_c_in, p_h_in)
        v_state_tuple = tf.contrib.rnn.LSTMStateTuple(v_c_in, v_h_in)

        self.seq_len = get_rnn_length(self.state_inputs)
        p_lstm_outputs_src, p_lstm_state = tf.nn.dynamic_rnn(
            p_lstm_cell, self.state_inputs, initial_state=p_state_tuple, sequence_length=self.seq_len,
            time_major=False)

        v_lstm_outputs_src, v_lstm_state = tf.nn.dynamic_rnn(
            v_lstm_cell, self.state_inputs, initial_state=v_state_tuple, sequence_length=self.seq_len,
            time_major=False)

        p_flat_rnn_out = tf.reshape(p_lstm_outputs_src, (-1, hidden_layer_size))
        p_lstm_c, p_lstm_h = p_lstm_state

        v_flat_rnn_out = tf.reshape(v_lstm_outputs_src, (-1, hidden_layer_size))
        v_lstm_c, v_lstm_h = v_lstm_state

        self.hidden_state_out = tf.concat([p_lstm_c, p_lstm_h, v_lstm_c, v_lstm_h], axis=1)
        w_a, b_a = dense([hidden_layer_size, hidden_layer_size], "actor_layer")

        w_mu_1, b_mu_1 = dense([hidden_layer_size, self.action_size], "actor_layer_mu_1")

        w_sigma_1, b_sigma_1 = dense([hidden_layer_size, self.action_size], "actor_layer_sigma_1")

        w_v, b_v = dense([hidden_layer_size, hidden_layer_size], "critic_layer")
        w4, b4 = dense([hidden_layer_size, 1], 'value_l3')

        l_a = tf.nn.tanh(tf.matmul(p_flat_rnn_out, w_a) + b_a)
        mu_logits = tf.matmul(l_a, w_mu_1) + b_mu_1

        if env_opts["nn_std"]:
            sigma_logits = tf.matmul(l_a, w_sigma_1) + b_sigma_1
        else:
            std_dim = hidden_layer_size // 3
            std_vars = tf.Variable(tf.random_normal([self.action_size, std_dim], stddev=0.005), name="std_vars")
            sigma_logits = tf.reduce_mean(std_vars, axis=1)
            sigma_logits = tf.zeros_like(mu_logits) + sigma_logits

        l_v = tf.nn.tanh(tf.matmul(v_flat_rnn_out, w_v) + b_v)

        self.mu_outputs = mu_logits
        self.sigma_outputs = tf.nn.softplus(sigma_logits) + 1e-3
        self.action_outputs = tf.concat([self.mu_outputs, self.sigma_outputs], axis=1)
        self.dist = get_distribution(self.action_size, self.mu_outputs, self.sigma_outputs)
        self.v_outputs = tf.identity(tf.matmul(l_v, w4) + b4)

        self.old_actions = tf.placeholder(tf.float32, [None, self.action_size * 2])
        old_mu = self.old_actions[:, 0:self.action_size]
        old_sigma = self.old_actions[:, self.action_size:]
        self.old_dist = get_distribution(self.action_size, old_mu, old_sigma)
        self.picked_actions = tf.placeholder(tf.float32, [None, self.action_size])
        self.session = session

    def get_state_inputs(self):
        return self.state_inputs

    def get_hidden_inputs(self):
        return self.hidden_state_in

    def value_outputs(self):
        return self.v_outputs

    def sample(self, state, hidden_state):
        a_out, h_out, v_out = self.session.run([self.action_outputs,
                                                self.hidden_state_out,
                                                self.v_outputs], feed_dict={
            self.state_inputs: np.array(state).reshape((1, 1, -1)),
            self.hidden_state_in: np.array([hidden_state])
        })
        a_out = a_out[0]
        h_out = h_out[0]
        v_out = v_out[0, 0]

        picked_actions = np.random.normal(a_out[0:self.action_size], a_out[self.action_size:])
        return picked_actions, a_out, h_out, v_out

    def get_old_actions_input(self):
        return self.old_actions

    def get_picked_actions_input(self):
        return self.picked_actions

    def get_old_prob(self):
        import tensorflow as tf
        prob = self.old_dist.prob(self.picked_actions)
        prod = tf.squeeze(prob)
        return prod

    def get_current_prob(self):
        import tensorflow as tf
        prob = self.dist.prob(self.picked_actions)
        prod = tf.squeeze(prob)
        return prod

    def entropy(self):
        import tensorflow as tf
        ent = self.dist.entropy()
        if self.action_size == 1:
            ent = tf.reduce_mean(ent, axis=1)
        return ent

    def strict_sample(self, state, hidden_state):
        a_outs, h_out = self.session.run([self.action_outputs,
                                          self.hidden_state_out], feed_dict={
            self.state_inputs: np.array(state).reshape((1, 1, -1)),
            self.hidden_state_in: np.array([hidden_state])
        })
        a_outs = a_outs[0]
        h_out = h_out[0]
        return a_outs[0:self.action_size], h_out

    def kl(self):
        import tensorflow as tf
        kl = self.old_dist.kl_divergence(self.dist)
        if self.action_size == 1:
            kl = tf.reduce_mean(kl, axis=1)
        return kl

    def get_initial_state(self):
        return self.init_hidden_state


class MlpContinousPolicy(Policy):
    def __init__(self, session, env_opts):
        super().__init__(env_opts["action_dim"], env_opts["state_dim"])
        import tensorflow as tf
        state_size = env_opts["state_dim"]
        hidden_layer_size = env_opts["hidden_layer_size"]
        self.state_inputs_timestep = tf.placeholder(tf.float32, [BATCH_SIZE, 1, state_size])
        self.state_inputs = tf.reshape(self.state_inputs_timestep, [-1, state_size])
        a_w1, a_b1 = dense([state_size, hidden_layer_size], "actor_l1")
        a_w2, a_b2 = dense([hidden_layer_size, hidden_layer_size], "actor_l2")
        a_w3, a_b3 = dense([hidden_layer_size, hidden_layer_size], "actor_l2")
        l1_a = tf.nn.tanh(tf.matmul(self.state_inputs, a_w1) + a_b1)
        l2_a = tf.nn.tanh(tf.matmul(l1_a, a_w2) + a_b2)
        l3_a = tf.nn.tanh(tf.matmul(l2_a, a_w3) + a_b3)

        a_mean_w1, a_mean_b1 = dense([hidden_layer_size, self.action_size], "actor_mean_l3")

        a_std_w1, a_std_b1 = dense([hidden_layer_size, self.action_size], "actor_std_l3")

        mu_logits = tf.matmul(l3_a, a_mean_w1) + a_mean_b1

        if env_opts["nn_std"]:
            sigma_logits = tf.matmul(l3_a, a_std_w1) + a_std_b1
        else:
            std_dim = hidden_layer_size // 3
            std_vars = tf.Variable(tf.random_normal([self.action_size, std_dim], stddev=0.005), name="std_vars")
            sigma_logits = tf.reduce_mean(std_vars, axis=1)
            sigma_logits = tf.zeros_like(mu_logits) + sigma_logits

        self.mu_outputs = mu_logits
        self.sigma_outputs = tf.nn.softplus(sigma_logits) + 1e-3
        self.action_outputs = tf.concat([self.mu_outputs, self.sigma_outputs], axis=1)
        self.dist = get_distribution(self.action_size, self.mu_outputs, self.sigma_outputs)
        self.old_actions = tf.placeholder(tf.float32, [None, self.action_size * 2])
        old_mu = self.old_actions[:, 0:self.action_size]
        old_sigma = self.old_actions[:, self.action_size:]
        self.old_dist = get_distribution(self.action_size, old_mu, old_sigma)
        self.picked_actions = tf.placeholder(tf.float32, [None, self.action_size])
        self.session = session
        self.hidden_inputs = tf.placeholder(tf.float32, [None, None])

        v_w1, v_b1 = dense([state_size, hidden_layer_size], "value_l1")
        v_w2, v_b2 = dense([hidden_layer_size, hidden_layer_size], "value_l2")
        v_w3, v_b3 = dense([hidden_layer_size, hidden_layer_size], "value_l3")
        v_w4, v_b4 = dense([hidden_layer_size, 1], "value_l4")
        l1_v = tf.nn.tanh(tf.matmul(self.state_inputs, v_w1) + v_b1)
        l2_v = tf.nn.tanh(tf.matmul(l1_v, v_w2) + v_b2)
        l3_v = tf.nn.tanh(tf.matmul(l2_v, v_w3) + v_b3)
        self.v_outputs = tf.identity(tf.matmul(l3_v, v_w4) + v_b4)

    def get_state_inputs(self):
        return self.state_inputs_timestep

    def get_hidden_inputs(self):
        return self.hidden_inputs

    def value_outputs(self):
        return self.v_outputs

    def sample(self, state, hidden_state):
        a_out, v_out = self.session.run([self.action_outputs,
                                         self.v_outputs], feed_dict={
            self.state_inputs: np.array(state).reshape((1, -1))
        })
        a_out = a_out[0]
        v_out = v_out[0, 0]
        h_out = np.array([0.0])
        picked_actions = np.random.normal(a_out[0:self.action_size], a_out[self.action_size:])
        return picked_actions, a_out, h_out, v_out

    def get_old_actions_input(self):
        return self.old_actions

    def get_picked_actions_input(self):
        return self.picked_actions

    def get_old_prob(self):
        import tensorflow as tf
        prob = self.old_dist.prob(self.picked_actions)
        prod = tf.squeeze(prob)
        return prod

    def get_current_prob(self):
        import tensorflow as tf
        prob = self.dist.prob(self.picked_actions)
        prod = tf.squeeze(prob)
        return prod

    def entropy(self):
        import tensorflow as tf
        ent = self.dist.entropy()
        if self.action_size == 1:
            ent = tf.reduce_mean(ent, axis=1)
        return ent

    def strict_sample(self, state, hidden_state):
        a_outs = self.session.run(self.action_outputs, feed_dict={
            self.state_inputs: np.array(state).reshape((1, -1))
        })
        a_outs = a_outs[0]
        h_out = np.array([0.0])
        return a_outs[0:self.action_size], h_out

    def kl(self):
        import tensorflow as tf
        kl = self.old_dist.kl_divergence(self.dist)
        if self.action_size == 1:
            kl = tf.reduce_mean(kl, axis=1)
        return kl

    def get_initial_state(self):
        return np.array([0.0])


class MlpDiscretePolicy(Policy):
    def __init__(self, session, env_opts):
        super().__init__(env_opts["action_dim"], env_opts["state_dim"])
        import tensorflow as tf
        state_size = env_opts["state_dim"]
        hidden_layer_size = env_opts["hidden_layer_size"]
        self.state_inputs_timestep = tf.placeholder(tf.float32, [BATCH_SIZE, 1, state_size])
        self.state_inputs = tf.reshape(self.state_inputs_timestep, [-1, state_size])
        a_w1, a_b1 = dense([state_size, hidden_layer_size], "actor_l1")
        a_w2, a_b2 = dense([hidden_layer_size, hidden_layer_size], "actor_l2")
        a_w3, a_b3 = dense([hidden_layer_size, hidden_layer_size], "actor_l3")
        l1_a = tf.nn.tanh(tf.matmul(self.state_inputs, a_w1) + a_b1)
        l2_a = tf.nn.tanh(tf.matmul(l1_a, a_w2) + a_b2)
        l3_a = tf.nn.tanh(tf.matmul(l2_a, a_w3) + a_b3)
        a_mean_w3, a_mean_b3 = dense([hidden_layer_size, self.action_size], "actor_mean_l3")
        logits = tf.matmul(l3_a, a_mean_w3) + a_mean_b3
        self.action_outputs = tf.nn.softmax(logits)
        self.old_actions = tf.placeholder(tf.float32, [None, self.action_size])
        self.picked_actions = tf.placeholder(tf.int32, [None, 1])
        picked_actions_squeeze = self.picked_actions[:, 0]
        self.picked_actions_ohe = tf.one_hot(picked_actions_squeeze, self.action_size)
        self.session = session
        self.hidden_inputs = tf.placeholder(tf.float32, [None, None])

        v_w1, v_b1 = dense([state_size, hidden_layer_size], "value_l1")
        v_w2, v_b2 = dense([hidden_layer_size, hidden_layer_size], "value_l2")
        v_w3, v_b3 = dense([hidden_layer_size, hidden_layer_size], "value_l3")
        v_w4, v_b4 = dense([hidden_layer_size, 1], "value_l4")
        l1_v = tf.nn.tanh(tf.matmul(self.state_inputs, v_w1) + v_b1)
        l2_v = tf.nn.tanh(tf.matmul(l1_v, v_w2) + v_b2)
        l3_v = tf.nn.tanh(tf.matmul(l2_v, v_w3) + v_b3)
        self.v_outputs = tf.identity(tf.matmul(l3_v, v_w4) + v_b4)

    def get_state_inputs(self):
        return self.state_inputs_timestep

    def get_hidden_inputs(self):
        return self.hidden_inputs

    def value_outputs(self):
        return self.v_outputs

    def sample(self, state, hidden_state):
        a_out, v_out = self.session.run([self.action_outputs,
                                         self.v_outputs], feed_dict={
            self.state_inputs: np.array(state).reshape((1, -1))
        })
        a_out = a_out[0]
        v_out = v_out[0, 0]
        h_out = np.array([0.0])
        picked_actions = np.random.choice(np.arange(0, self.action_size), p=a_out)
        return picked_actions, a_out, h_out, v_out

    def get_old_actions_input(self):
        return self.old_actions

    def get_picked_actions_input(self):
        return self.picked_actions

    def get_old_prob(self):
        import tensorflow as tf
        result = tf.reduce_sum(self.picked_actions_ohe * self.old_actions, axis=1)
        return result

    def get_current_prob(self):
        import tensorflow as tf
        result = tf.reduce_sum(self.picked_actions_ohe * self.action_outputs, axis=1)
        return result

    def entropy(self):
        import tensorflow as tf
        result = -tf.reduce_sum(self.action_outputs * tf.log(self.action_outputs), axis=1)
        return result

    def strict_sample(self, state, hidden_state):
        a_outs = self.session.run(self.action_outputs, feed_dict={
            self.state_inputs: np.array(state).reshape((1, -1))
        })
        a_outs = a_outs[0]
        h_out = np.array([0.0])
        picked_actions = np.argmax(a_outs)
        return picked_actions, h_out

    def kl(self):
        import tensorflow as tf
        x = self.old_actions * tf.log(self.old_actions / (self.action_outputs + 1e-4))
        result = tf.reduce_sum(x, axis=1)
        return result

    def get_initial_state(self):
        return np.array([0.0])


class LstmDiscretePolicy(Policy):
    def __init__(self, session, env_opts):
        import tensorflow as tf
        super().__init__(env_opts["action_dim"], env_opts["state_dim"])
        state_size = env_opts["state_dim"]
        hidden_layer_size = env_opts["hidden_layer_size"]
        self.state_inputs = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_SEQ_LEN, state_size])
        p_lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size, state_is_tuple=True, name="policy_rnn")
        v_lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size, state_is_tuple=True, name="value_rnn")
        p_cell_size = p_lstm_cell.state_size.c + p_lstm_cell.state_size.h
        v_cell_size = v_lstm_cell.state_size.c + v_lstm_cell.state_size.h
        self.init_hidden_state = np.zeros((p_cell_size + v_cell_size), np.float32)
        self.hidden_state_in = tf.placeholder(tf.float32, [BATCH_SIZE, p_cell_size + v_cell_size])

        p_c_in = self.hidden_state_in[:, 0:p_lstm_cell.state_size.c]
        p_h_in = self.hidden_state_in[:, p_lstm_cell.state_size.c:p_cell_size]

        v_c_in = self.hidden_state_in[:, p_cell_size:(p_cell_size + v_lstm_cell.state_size.c)]
        v_h_in = self.hidden_state_in[:, (p_cell_size + v_lstm_cell.state_size.c):]

        p_state_tuple = tf.contrib.rnn.LSTMStateTuple(p_c_in, p_h_in)
        v_state_tuple = tf.contrib.rnn.LSTMStateTuple(v_c_in, v_h_in)

        self.seq_len = get_rnn_length(self.state_inputs)
        p_lstm_outputs_src, p_lstm_state = tf.nn.dynamic_rnn(
            p_lstm_cell, self.state_inputs, initial_state=p_state_tuple, sequence_length=self.seq_len,
            time_major=False)

        v_lstm_outputs_src, v_lstm_state = tf.nn.dynamic_rnn(
            v_lstm_cell, self.state_inputs, initial_state=v_state_tuple, sequence_length=self.seq_len,
            time_major=False)

        p_flat_rnn_out = tf.reshape(p_lstm_outputs_src, (-1, hidden_layer_size))
        p_lstm_c, p_lstm_h = p_lstm_state

        v_flat_rnn_out = tf.reshape(v_lstm_outputs_src, (-1, hidden_layer_size))
        v_lstm_c, v_lstm_h = v_lstm_state

        self.hidden_state_out = tf.concat([p_lstm_c, p_lstm_h, v_lstm_c, v_lstm_h], axis=1)

        a_w1, a_b1 = dense([hidden_layer_size, hidden_layer_size], "actor_l1")
        l1_a = tf.nn.tanh(tf.matmul(p_flat_rnn_out, a_w1) + a_b1)
        a_mean_w3, a_mean_b3 = dense([hidden_layer_size, self.action_size], "actor_mean_l3")
        logits = tf.matmul(l1_a, a_mean_w3) + a_mean_b3
        self.action_outputs = tf.nn.softmax(logits)
        self.old_actions = tf.placeholder(tf.float32, [None, self.action_size])
        self.picked_actions = tf.placeholder(tf.int32, [None, 1])
        picked_actions_squeeze = self.picked_actions[:, 0]
        self.picked_actions_ohe = tf.one_hot(picked_actions_squeeze, self.action_size)
        self.session = session
        self.hidden_inputs = tf.placeholder(tf.float32, [None, None])

        v_w1, v_b1 = dense([hidden_layer_size, hidden_layer_size], "value_l1")
        v_w4, v_b4 = dense([hidden_layer_size, 1], "value_l4")
        l1_v = tf.nn.tanh(tf.matmul(v_flat_rnn_out, v_w1) + v_b1)
        self.v_outputs = tf.identity(tf.matmul(l1_v, v_w4) + v_b4)

    def get_state_inputs(self):
        return self.state_inputs

    def get_hidden_inputs(self):
        return self.hidden_state_in

    def value_outputs(self):
        return self.v_outputs

    def sample(self, state, hidden_state):
        a_out, h_out, v_out = self.session.run([self.action_outputs,
                                                self.hidden_state_out,
                                                self.v_outputs], feed_dict={
            self.state_inputs: np.array(state).reshape((1, 1, -1)),
            self.hidden_state_in: np.array([hidden_state])
        })
        a_out = a_out[0]
        v_out = v_out[0, 0]
        h_out = h_out[0]
        picked_actions = np.random.choice(np.arange(0, self.action_size), p=a_out)
        return picked_actions, a_out, h_out, v_out

    def get_old_actions_input(self):
        return self.old_actions

    def get_picked_actions_input(self):
        return self.picked_actions

    def get_old_prob(self):
        import tensorflow as tf
        result = tf.reduce_sum(self.picked_actions_ohe * self.old_actions, axis=1)
        return result

    def get_current_prob(self):
        import tensorflow as tf
        result = tf.reduce_sum(self.picked_actions_ohe * self.action_outputs, axis=1)
        return result

    def entropy(self):
        import tensorflow as tf
        result = -tf.reduce_mean(self.action_outputs * tf.log(self.action_outputs), axis=1)
        return result

    def strict_sample(self, state, hidden_state):
        a_outs, h_out = self.session.run([self.action_outputs,
                                          self.hidden_state_out], feed_dict={
            self.state_inputs: np.array(state).reshape((1, 1, -1)),
            self.hidden_state_in: np.array([hidden_state])
        })
        a_outs = a_outs[0]
        h_out = h_out[0]
        picked_actions = np.argmax(a_outs)
        return picked_actions, h_out

    def kl(self):
        import tensorflow as tf
        x = self.old_actions * tf.log(self.old_actions / (self.action_outputs + 1e-4))
        result = tf.reduce_sum(x, axis=1)
        return result

    def get_initial_state(self):
        return self.init_hidden_state


class DiscretizeContinousPolicy(Policy):
    def __init__(self, session, env_opts, discrete_step):
        super().__init__(env_opts["action_dim"], env_opts["state_dim"])
        import tensorflow as tf
        state_size = env_opts["state_dim"]
        self.scales_lo = env_opts["scales_lo"]
        self.scales_hi = env_opts["scales_hi"]
        self.discrete_step = discrete_step

        hidden_layer_size = env_opts["hidden_layer_size"]
        self.state_inputs_timestep = tf.placeholder(tf.float32, [BATCH_SIZE, 1, state_size])
        self.state_inputs = tf.reshape(self.state_inputs_timestep, [-1, state_size])
        a_w1, a_b1 = dense([state_size, hidden_layer_size], "actor_l1")
        a_w2, a_b2 = dense([hidden_layer_size, hidden_layer_size], "actor_l2")
        a_w3, a_b3 = dense([hidden_layer_size, hidden_layer_size], "actor_l3")
        l1_a = tf.nn.tanh(tf.matmul(self.state_inputs, a_w1) + a_b1)
        l2_a = tf.nn.tanh(tf.matmul(l1_a, a_w2) + a_b2)
        l3_a = tf.nn.tanh(tf.matmul(l2_a, a_w3) + a_b3)
        self.picked_actions = tf.placeholder(tf.float32, [None, self.action_size])
        self.old_actions = tf.placeholder(tf.float32, [None, self.discrete_step * self.action_size])
        all_action_outputs = []
        for i in range(self.action_size):
            a_mean_w3, a_mean_b3 = dense([hidden_layer_size, self.discrete_step], "actor_mean_l3_%s" % i)
            logits = tf.matmul(l3_a, a_mean_w3) + a_mean_b3
            action_outputs = tf.nn.softmax(logits)
            all_action_outputs.append(action_outputs)
        self.action_outputs = tf.concat(all_action_outputs, axis=1)

        pick_mask = (self.picked_actions - self.scales_lo) / (self.scales_hi - self.scales_lo) * (
                self.discrete_step - 1)
        self.picked_actions_int = tf.cast(pick_mask + 0.00001, tf.int32)
        self.picked_actions_ohe = tf.one_hot(self.picked_actions_int, self.discrete_step)

        self.picked_actions_ohe = tf.reshape(self.picked_actions_ohe, [-1, self.discrete_step * self.action_size])

        self.session = session
        self.hidden_inputs = tf.placeholder(tf.float32, [None, None])

        v_w1, v_b1 = dense([state_size, hidden_layer_size], "value_l1")
        v_w2, v_b2 = dense([hidden_layer_size, hidden_layer_size], "value_l2")
        v_w3, v_b3 = dense([hidden_layer_size, hidden_layer_size], "value_l3")
        v_w4, v_b4 = dense([hidden_layer_size, 1], "value_l4")
        l1_v = tf.nn.tanh(tf.matmul(self.state_inputs, v_w1) + v_b1)
        l2_v = tf.nn.tanh(tf.matmul(l1_v, v_w2) + v_b2)
        l3_v = tf.nn.tanh(tf.matmul(l2_v, v_w3) + v_b3)
        self.v_outputs = tf.identity(tf.matmul(l3_v, v_w4) + v_b4)

    def get_state_inputs(self):
        return self.state_inputs_timestep

    def get_hidden_inputs(self):
        return self.hidden_inputs

    def value_outputs(self):
        return self.v_outputs

    def sample(self, state, hidden_state):
        a_out, v_out = self.session.run([self.action_outputs,
                                         self.v_outputs], feed_dict={
            self.state_inputs: np.array(state).reshape((1, -1))
        })
        a_out = a_out[0]
        v_out = v_out[0, 0]
        h_out = np.array([0.0])
        picked_actions = np.reshape(a_out, [self.action_size, self.discrete_step])
        acts = []
        for i in range(self.action_size):
            action_probs = picked_actions[i]
            chosen_action = np.random.choice(np.arange(0, self.discrete_step), p=action_probs)
            chosen_action = chosen_action / (self.discrete_step - 1)
            chosen_action = chosen_action * (self.scales_hi[i] - self.scales_lo[i]) + self.scales_lo[i]
            acts.append(chosen_action)
        picked_actions = np.array(acts)
        return picked_actions, a_out, h_out, v_out

    def get_old_actions_input(self):
        return self.old_actions

    def get_picked_actions_input(self):
        return self.picked_actions

    def get_old_prob(self):
        import tensorflow as tf
        result = self.picked_actions_ohe * self.old_actions
        result = tf.reshape(result, [-1, self.action_size, self.discrete_step])
        result = tf.reduce_sum(result, axis=2)
        result = tf.reduce_prod(result, axis=1)
        return result

    def get_current_prob(self):
        import tensorflow as tf
        result = self.picked_actions_ohe * self.action_outputs
        result = tf.reshape(result, [-1, self.action_size, self.discrete_step])
        result = tf.reduce_sum(result, axis=2)
        result = tf.reduce_prod(result, axis=1)
        return result

    def entropy(self):
        import tensorflow as tf
        result = -tf.reduce_sum(self.action_outputs * tf.log(self.action_outputs), axis=1)
        return result / self.discrete_step

    def strict_sample(self, state, hidden_state):
        a_out = self.session.run(self.action_outputs, feed_dict={
            self.state_inputs: np.array(state).reshape((1, -1))
        })
        a_out = a_out[0]
        h_out = np.array([0.0])
        picked_actions = np.reshape(a_out, [self.action_size, self.discrete_step])
        acts = []
        for i in range(self.action_size):
            action_probs = picked_actions[i]
            chosen_action = np.argmax(action_probs)
            chosen_action = chosen_action / (self.discrete_step - 1)
            chosen_action = chosen_action * (self.scales_hi[i] - self.scales_lo[i]) + self.scales_lo[i]
            acts.append(chosen_action)
        picked_actions = np.array(acts)
        return picked_actions, h_out

    def kl(self):
        import tensorflow as tf
        x = self.old_actions * tf.log(self.old_actions / (self.action_outputs + 1e-9))
        result = tf.reduce_mean(x, axis=1)
        return result

    def get_initial_state(self):
        return np.array([0.0])

    def get_prob_ratio(self):
        import tensorflow as tf
        new_result = self.picked_actions_ohe * self.action_outputs
        new_result = tf.reshape(new_result, [-1, self.action_size, self.discrete_step])
        new_result = tf.reduce_sum(new_result, axis=2)

        old_result = self.picked_actions_ohe * self.old_actions
        old_result = tf.reshape(old_result, [-1, self.action_size, self.discrete_step])
        old_result = tf.reduce_sum(old_result, axis=2)
        result = tf.reduce_prod(new_result / old_result, axis=1)
        return result


def get_policy(env_opts, session):
    if env_opts["discrete"]:
        if env_opts["recurrent"]:
            return LstmDiscretePolicy(session, env_opts)
        else:
            return MlpDiscretePolicy(session, env_opts)
    else:
        if env_opts["recurrent"]:
            return LstmContinousPolicy(session, env_opts)
        else:
            if env_opts["discretize_space"]:
                return DiscretizeContinousPolicy(session, env_opts, env_opts["discrete_step"])
            else:
                return MlpContinousPolicy(session, env_opts)


def get_distribution(action_dim, loc, scale):
    import tensorflow as tf
    if action_dim == 1:
        return tf.distributions.Normal(loc, scale)
    else:
        return tf.contrib.distributions.MultivariateNormalDiag(loc, scale)
