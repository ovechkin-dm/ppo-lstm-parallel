class ScalingRewardTransformer:
    def __init__(self, env_opts):
        self.env_opts = env_opts

    def transform_reward(self, r):
        return max(-0.001, r / 100.0)


class PositiveRewardTransformer:
    def __init__(self, env_opts):
        self.env_opts = env_opts

    def transform_reward(self, r):
        return max(-0.001, r / 100.0)


class IdentityRewardTransformer:
    def __init__(self, env_opts):
        self.env_opts = env_opts

    def transform_reward(self, r):
        return r


def get_reward_transformer(env_ops):
    name = env_ops["reward_transform"]
    if name == "scale":
        return ScalingRewardTransformer(env_ops)
    elif name == "positive":
        return PositiveRewardTransformer(env_ops)
    else:
        return IdentityRewardTransformer(env_ops)