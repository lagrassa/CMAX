import os
from datetime import datetime
import torch
import numpy as np
from mpi4py import MPI
from odium.utils.mpi_utils.mpi_utils import sync_networks, sync_grads
from odium.agents.her_switch_residual_agent.replay_buffer import replay_buffer
from odium.agents.her_switch_residual_agent.models import actor, critic_with_switch
from odium.utils.mpi_utils.normalizer import normalizer
from odium.agents.her_switch_residual_agent.her_sampler import her_sampler
import odium.utils.logger as logger

"""
ddpg with HER (MPI-version)

"""


class her_switch_residual_agent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params
        # create the network
        self.actor_network = actor(env_params, residual=True)
        self.critic_switch_network = critic_with_switch(env_params)
        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_switch_network)
        # build up the target network
        self.actor_target_network = actor(env_params, residual=True)
        self.critic_switch_target_network = critic_with_switch(env_params)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(
            self.actor_network.state_dict())
        self.critic_switch_target_network.load_state_dict(
            self.critic_switch_network.state_dict())
        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_switch_network.cuda()
            self.actor_target_network.cuda()
            self.critic_switch_target_network.cuda()
        # create the optimizer
        self.actor_optim = torch.optim.Adam(
            self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_switch_optim = torch.optim.Adam(
            self.critic_switch_network.parameters(), lr=self.args.lr_critic)
        # her sampler
        self.her_module = her_sampler(
            self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)
        # create the replay buffer
        self.buffer = replay_buffer(
            self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        # create the normalizer
        self.o_norm = normalizer(
            size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(
            size=env_params['goal'], default_clip_range=self.args.clip_range)
        # create the dict for store the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = os.path.join(
                self.args.save_dir, self.args.env_name)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
        logger.info("initialized agent")

    def learn(self):
        """
        train the network

        """
        logger.info("Training..")
        n_steps = 0
        best_success_rate = 0.
        prev_actor_losses = [0.0]
        actor_losses = [0.0]
        coin_flipping = False
        # start to collect samples
        for epoch in range(self.args.n_epochs):
            if (epoch == 0 or abs(np.mean(actor_losses) - np.mean(prev_actor_losses)) > self.args.threshold):
                logger.info('Only training critic')
                self.change_actor_lr(0.0)
                coin_flipping = True
            else:
                self.change_actor_lr(self.args.lr_actor)
                coin_flipping = False
            prev_actor_losses = actor_losses
            actor_losses = []
            critic_losses = []
            switch_losses = []
            for _ in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions, mb_switch_actions = [], [], [], [], []
                for _ in range(self.args.num_rollouts_per_mpi):
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions, ep_switch_actions = [], [], [], [], []
                    # reset the environment
                    observation = self.env.reset()
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    # start to collect samples
                    for _ in range(self.env_params['max_timesteps']):
                        n_steps += 1
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(obs, g)
                            pi = self.actor_network(input_tensor)
                            _, switch_actions_q_values = self.critic_switch_network(
                                input_tensor, pi)
                            switch_action = self._select_switch_actions(
                                switch_actions_q_values)
                        # feed the actions into the environment
                        if switch_action == 0:
                            # Hardcoded action
                            # self.controller.act(observation)
                            # Zero action corresponds to controller action
                            action = np.zeros(self.env_params['action'])
                        else:
                            # Learned policy action
                            action = self._select_actions(pi, coin_flipping)
                        observation_new, _, _, info = self.env.step(action)
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())
                        ep_switch_actions.append(switch_action)
                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new
                        observation = observation_new
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                    mb_switch_actions.append(ep_switch_actions)
                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                mb_switch_actions = np.array(mb_switch_actions)
                # store the episodes
                self.buffer.store_episode(
                    [mb_obs, mb_ag, mb_g, mb_actions, mb_switch_actions])
                self._update_normalizer(
                    [mb_obs, mb_ag, mb_g, mb_actions, mb_switch_actions])
                for _ in range(self.args.n_batches):
                    # train the network
                    critic_loss, actor_loss, switch_loss = self._update_network()
                    actor_losses.append(actor_loss.detach().numpy())
                    critic_losses.append(critic_loss.detach().numpy())
                    switch_losses.append(switch_loss.detach().numpy())
                # soft update
                self._soft_update_target_network(
                    self.actor_target_network, self.actor_network)
                self._soft_update_target_network(
                    self.critic_switch_target_network, self.critic_switch_network)
            # start to do the evaluation
            success_rate, prop_hardcoded = self._eval_agent()
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[{}] epoch is: {}, Num steps: {}, eval success rate is: {:.3f}'.format(
                    datetime.now(), epoch, n_steps, success_rate))
                logger.record_tabular('epoch', epoch)
                logger.record_tabular('n_steps', n_steps)
                logger.record_tabular('success_rate', success_rate)
                logger.record_tabular('prop_hardcoded', prop_hardcoded)
                logger.record_tabular('actor_loss', np.mean(actor_losses))
                logger.record_tabular('critic_loss', np.mean(critic_losses))
                logger.record_tabular('switch_loss', np.mean(switch_losses))
                logger.dump_tabular()
                if success_rate > best_success_rate:
                    logger.info("Better success rate... Saving policy")
                    torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict()],
                               self.model_path + '/model.pt')
                    best_success_rate = success_rate

    def change_actor_lr(self, new_lr):
        for param_group in self.actor_optim.param_groups:
            param_group['lr'] = new_lr

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs

    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi, coin_flipping=False):
        action = pi.cpu().numpy().squeeze()
        noise_eps = self.args.noise_eps
        random_eps = self.args.random_eps
        if coin_flipping:
            deterministic_actions = np.random.random() < 0.5
            if deterministic_actions:
                noise_eps = 0.
                random_eps = 0.
        # add the gaussian
        action += noise_eps * \
            self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(
            action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'],
                                           size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, random_eps,
                                     1)[0] * (random_actions - action)
        return action

    # this function will choose the switch action and do the exploration
    def _select_switch_actions(self, q_values):
        switch_action = torch.argmax(q_values.squeeze())
        # Exploration
        if np.random.random() <= self.args.switch_noise_eps:
            # Random action
            switch_action = np.random.randint(2)

        return switch_action

    # update the normalizer

    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions, mb_switch_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs,
                       'ag': mb_ag,
                       'g': mb_g,
                       'actions': mb_actions,
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       'switch_actions': mb_switch_actions,
                       }
        transitions = self.her_module.sample_her_transitions(
            buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                (1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(
            o_next, g)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(
            inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(
            transitions['actions'], dtype=torch.float32)
        switch_actions_tensor = torch.tensor(
            transitions['switch_actions'], dtype=torch.int64)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            switch_actions_tensor = switch_actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value, switch_q_target_value = self.critic_switch_target_network(
                inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)

            # Switch network training
            # Double network training
            # NOTE: Double network training enabled by default
            _, switch_q_next_value = self.critic_switch_network(
                inputs_next_norm_tensor, actions_next)
            switch_actions_next = torch.argmax(
                switch_q_next_value, dim=1, keepdim=True)
            switch_q_target_max_value = switch_q_target_value.gather(
                1, switch_actions_next)

            switch_q_target_max_value = switch_q_target_max_value.detach()
            switch_target_q_value = r_tensor + self.args.gamma * switch_q_target_max_value
            switch_target_q_value = torch.clamp(
                switch_target_q_value, -clip_return, 0)
        # the q loss
        q_value, switch_q_value = self.critic_switch_network(
            inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - q_value).pow(2).mean()
        # the switch loss
        # switch_real_q_values = self.switch_network(inputs_next_norm_tensor)
        switch_q_value = switch_q_value.gather(
            1, switch_actions_tensor.unsqueeze(-1))
        switch_loss = (switch_target_q_value -
                       switch_q_value).pow(2).mean()
        # Critic switch loss
        critic_switch_loss = critic_loss + switch_loss
        # the actor loss
        actions_real = self.actor_network(inputs_norm_tensor)
        actor_loss, _, values = self.critic_switch_network(
            inputs_norm_tensor, actions_real, value=True)
        # NOTE: Using advantage
        actor_loss = actor_loss - values
        actor_loss = -actor_loss.mean()
        actor_loss += self.args.action_l2 * \
            (actions_real / self.env_params['action_max']).pow(2).mean()
        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()
        # update the critic_switch_network
        self.critic_switch_optim.zero_grad()
        critic_switch_loss.backward()
        sync_grads(self.critic_switch_network)
        self.critic_switch_optim.step()

        return critic_loss, actor_loss, switch_loss

    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        total_steps = 0
        total_hardcoded_steps = 0
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                total_steps += 1
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    # convert the actions
                    _, switch_q_values = self.critic_switch_network(
                        input_tensor, pi)
                    switch_actions = torch.argmax(
                        switch_q_values.squeeze())
                    if switch_actions == 0:
                        # Hardcoded controller
                        total_hardcoded_steps += 1
                        # self.controller.act(observation)
                        actions = np.zeros(self.env_params['action'])
                    else:
                        # Learned controller
                        actions = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = self.env.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                observation = observation_new
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(
            local_success_rate, op=MPI.SUM)
        global_steps = MPI.COMM_WORLD.allreduce(total_steps, op=MPI.SUM)
        global_hardcoded_steps = MPI.COMM_WORLD.allreduce(
            total_hardcoded_steps, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size(), global_hardcoded_steps / global_steps