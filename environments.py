def parse_properties(file_name):
    lines = open(file_name).readlines()
    result = {}
    for l in lines:
        a, b = l.split("=")
        b = b.strip()
        if b == "True":
            result[a] = True
        elif b == "False":
            result[a] = False
        elif "." in b:
            result[a] = float(b)
        elif b.isdigit():
            result[a] = int(b)
        else:
            result[a] = b
    return result


def get_env_options(env_name, use_gpu):
    import gym
    env = gym.make(env_name)
    max_episode_steps = env.spec.max_episode_steps
    if max_episode_steps is None:
        max_episode_steps = 1e+8
    state_dim = env.observation_space.shape[0]
    discrete = False
    action_dim = None
    scales_lo = None
    scales_hi = None
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
        discrete = True
    elif isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]
        scales_lo = env.action_space.low
        scales_hi = env.action_space.high
    basic_opts = {
        "action_dim": action_dim,
        "env_name": env_name,
        "state_dim": state_dim,
        "discrete": discrete,
        "scales_lo": scales_lo,
        "scales_hi": scales_hi,
        "max_episode_steps": max_episode_steps,
        "use_gpu": use_gpu,

    }
    file_props = get_config(env_name)
    for k, v in file_props.items():
        basic_opts[k] = v
    result = basic_opts
    for k, v in result.items():
        print("%s : %s" % (k, v))
    return result


def get_config(env_name):
    try:
        file_props = parse_properties("props/%s.properties" % env_name)
    except:
        print("Failed to load custom properties for env. Using default")
        file_props = {}
    default_props = parse_properties("props/default.properties")
    result = {}
    for k, v in default_props.items():
        result[k] = v
    for k, v in file_props.items():
        result[k] = v
    mem_fraction = 0.98 / (result["worker_num"] + 2)
    result["mem_fraction"] = mem_fraction
    return result


class EnvironmentProducer:
    def __init__(self, env_name, use_gpu):
        self.env_name = env_name
        self.use_gpu = use_gpu

    def get_new_environment(self):
        import gym
        env = gym.make(self.env_name)
        return env

    def get_env_name(self):
        return self.env_name

    def get_use_gpu(self):
        return self.use_gpu
