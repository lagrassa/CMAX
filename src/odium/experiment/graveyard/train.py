import os
import random
from mpi4py import MPI
import numpy as np
import torch

from odium.agents.her_agent.arguments import get_args
from odium.agents.her_agent.her_agent import her_agent
import odium.utils.logger as logger
from odium.utils.env_utils.make_env import make_env

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""


def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
              'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0],
              'action_max': env.action_space.high[0],
              'num_features': env.num_features,
              }
    params['max_timesteps'] = env._max_episode_steps
    return params


def launch(args):
    # create the ddpg_agent
    env = make_env(args.env_name, env_id=args.env_id,
                   reward_type=args.reward_type)
    # set random seeds for reproducibility
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # Configure logger
    if MPI.COMM_WORLD.Get_rank() == 0 and args.log_dir:
        logger.configure(dir=os.path.join(
            'logs', 'her', args.log_dir), format_strs=['tensorboard', 'log', 'csv', 'json', 'stdout'])
    args.log_dir = logger.get_dir()
    assert args.log_dir is not None
    os.makedirs(args.log_dir, exist_ok=True)
    # TODO: Write code for loading and saving params from/to json files
    # get the environment parameters
    env_params = get_env_params(env)
    # create the ddpg agent to interact with the environment
    her_trainer = her_agent(args, env, env_params)
    her_trainer.learn()


if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    launch(args)
