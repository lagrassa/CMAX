from odium.utils.env_utils.wrapper import get_wrapper_env


def make_env(env_name, env_id=None, discrete=None, reward_type='sparse',cfg=None):

    assert isinstance(env_name, str), "Env name arg should be string"

    if env_name.startswith('Residual'):
        # Residual environment
        raise NotImplementedError
    else:
        # Plain environment
        return get_wrapper_env(env_name,
                               env_id=env_id,
                               discrete=discrete,
                               reward_type=reward_type,
                               cfg=cfg
        )
