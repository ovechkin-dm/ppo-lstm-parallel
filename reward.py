class ScalingRewardTransformer:
    def __init__(self, env_opts):
        self.env_opts = env_opts
        self.discount_factor = env_opts["discount_factor"]

    def transform_reward(self, r):
        return r / 100.0
