import numpy as np

MAX_SEQ_LEN = None
BATCH_SIZE = None


def dense(shape, name):
    import tensorflow as tf
    cur_w = tf.Variable(tf.random_normal(shape, stddev=0.005), name=name + "_weights")
    cur_b = tf.Variable(tf.zeros([shape[1]], dtype=tf.float32), name=name + "_biases")
    return cur_w, cur_b


def get_rnn_length(sequence):
    import tensorflow as tf
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


def last_relevant_rnn_outputs(output, length):
    import tensorflow as tf
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant


def split_episode(states, advantages, returns, picked_actions, old_action_probs, all_pred_values,
                  hidden_states, max_timestep):
    state_size = len(states[0])
    batch_elements = []
    if len(picked_actions.shape) == 1:
        picked_actions = np.expand_dims(picked_actions, axis=1)
    for i in range((len(states) - 1) // max_timestep + 1):
        start = i * max_timestep
        end = min((i + 1) * max_timestep, len(states))
        l = end - start
        b_states = np.zeros((max_timestep, state_size))
        b_states[0:l] = states[start:end]
        b_hidden_state = hidden_states[start]
        b_advantages = np.zeros(max_timestep, )
        b_advantages[0:l] = advantages[start:end]
        b_returns = np.zeros(max_timestep, )
        b_returns[0:l] = returns[start:end]
        b_picked_actions = np.ones((max_timestep, picked_actions.shape[-1]))
        b_picked_actions[0:l] = picked_actions[start:end]
        b_old_action_probs = np.zeros((max_timestep, old_action_probs.shape[-1])) + 0.5
        b_old_action_probs[0:l] = old_action_probs[start:end]
        b_all_pred_values = np.zeros(max_timestep, )
        b_all_pred_values[0:l] = all_pred_values[start:end]
        res = b_states, b_hidden_state, b_advantages, b_returns, b_picked_actions, b_old_action_probs, b_all_pred_values
        batch_elements.append(res)
    return batch_elements


def create_session(env_opts, on_gpu):
    import tensorflow as tf
    if env_opts["use_gpu"] and on_gpu:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=env_opts["mem_fraction"])
        session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    else:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
        session = tf.Session(config=tf.ConfigProto(device_count={'CPU': 1, 'GPU': 0}, gpu_options=gpu_options))
    return session
