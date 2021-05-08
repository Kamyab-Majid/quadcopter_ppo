from .env_wrapper import EnvWrapper


def create_env_wrapper(config):
    env_name = config['env']
    return EnvWrapper(env_name)