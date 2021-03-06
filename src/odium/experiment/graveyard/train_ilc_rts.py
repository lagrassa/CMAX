import os
import random
from mpi4py import MPI
import numpy as np
import torch
import ray

from odium.agents.ilc_rts_agent.arguments import get_args
from odium.agents.ilc_rts_agent.ilc_rts_agent import ilc_rts_agent
import odium.utils.logger as logger
from odium.utils.logger import DEBUG
from odium.utils.env_utils.make_env import make_env
from odium.utils.controller_utils.get_controller import get_controller


def get_env_params(env):
    obs = env.reset()

    params = {'obs': obs['observation'].shape[0],
              'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0],
              'num_actions': env.num_discrete_actions,
              'qpos': obs['sim_state'].qpos.shape[0],
              'qvel': obs['sim_state'].qvel.shape[0],
              'num_features': env.num_features,
              # 'action_max': env.action_space.high[0],
              }
    params['max_timesteps'] = env._max_episode_steps
    return params


def launch(args):
    ray.init()
    env = make_env(args.env_name, env_id=args.env_id,
                   discrete=True, reward_type=args.reward_type)
    planning_env = make_env(args.env_name, env_id=args.planning_env_id,
                            discrete=True, reward_type=args.reward_type)
    # set random seeds for reproducibility
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # NOTE: Not using the same seed as the real env
    planning_env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.deterministic:
        env.make_deterministic()
        planning_env.make_deterministic()
    if args.debug:
        logger.set_level(DEBUG)
    controller = get_controller(
        args.env_name,
        env_id=args.planning_env_id,
        discrete=True,
        num_expansions=args.num_expansions,
        reward_type=args.reward_type,
    )

    # Configure logger
    if MPI.COMM_WORLD.Get_rank() == 0 and args.log_dir:
        logger.configure(dir=os.path.join(
            'logs', 'ilc', args.log_dir), format_strs=['tensorboard', 'log', 'csv', 'json', 'stdout'])
    args.log_dir = logger.get_dir()
    assert args.log_dir is not None
    os.makedirs(args.log_dir, exist_ok=True)

    env_params = get_env_params(env)
    ilc_rts_trainer = ilc_rts_agent(
        args, env, planning_env, env_params, controller)
    ilc_rts_trainer.learn()


if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    launch(args)
