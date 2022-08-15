import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from utils.networks import MLPNetwork
from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from utils.agents import DDPGAgent
from utils.consensus_builder import ConsensusBuilder

MSELoss = torch.nn.MSELoss()

class MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params, alg_types,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                 discrete_action=False, ema_tau=0.996, consensus_builder_hidden_dim=32, consensus_builder_dim=4,
                 online_temp=0.1, target_temp=0.04, center_tau=0.9, consensus=True):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.agents = [DDPGAgent(lr=lr, discrete_action=discrete_action,
                                 hidden_dim=hidden_dim, consensus_builder_dim=consensus_builder_dim, consensus=consensus,
                                 **params)
                       for params in agent_init_params]

        self.consensus = consensus
        if self.consensus:
            self.consensus_builder = ConsensusBuilder(agent_init_params[0]['num_in_pol'], consensus_builder_hidden_dim, consensus_builder_dim, ema_tau)
            self.consensus_builder_optimizer = torch.optim.Adam(self.consensus_builder.update_parameters(), lr=lr)
            self.obs_center = torch.zeros(1, consensus_builder_dim)
            self.online_temp = online_temp
            self.target_temp = target_temp
            self.consensus_builder_params = self.consensus_builder.update_parameters()
            self.center_tau = center_tau

        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        if self.consensus:
            self.consensus_builder_dev = 'cpu'  # device for consensus_builder
        self.niter = 0

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        if self.consensus:
            obs_tensor = torch.stack(observations, dim=-2)
            student_obs = self.consensus_builder.calc_student(obs_tensor)
            obs_prob = F.softmax(student_obs / self.online_temp, dim=-1)
            obs_id = obs_prob.max(-1)[1]
            obs_id_onehot = F.one_hot(obs_id, self.consensus_builder.consensus_builder_dim).float().squeeze(0).unsqueeze(1)
            return [a.step(obs, ids, explore=explore) for a, obs, ids in zip(self.agents,
                                                                    observations, obs_id_onehot)]
        else:
            return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                                    observations)]


    def update_consensus_builder(self, sample, logger):
        obs, acs, rews, next_obs, dones = sample

        obs = torch.stack(obs, dim=-2)
        teacher_obs = self.consensus_builder.calc_teacher(obs)
        student_obs = self.consensus_builder.calc_student(obs)

        teacher_obs_centering = teacher_obs - self.obs_center.unsqueeze(0)
        student_obs_sharp = student_obs / self.online_temp
        teacher_obs_centering_z = F.softmax(teacher_obs_centering / self.target_temp, dim=-1)
        contrastive_loss = - torch.bmm(teacher_obs_centering_z.detach(), F.log_softmax(student_obs_sharp, dim=-1).transpose(1, 2))
        contrastive_mask = torch.ones_like(obs[:, :, 0]).unsqueeze(-1)
        contrastive_mask = torch.bmm(contrastive_mask, contrastive_mask.transpose(1, 2))
        contrastive_mask = contrastive_mask * (1 - torch.diag_embed(torch.ones(obs.size(1)))).unsqueeze(0)
        contrastive_loss = (contrastive_loss * contrastive_mask).sum() / contrastive_mask.sum()
        self.consensus_builder_optimizer.zero_grad()
        contrastive_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.consensus_builder_params, 0.5)
        self.consensus_builder_optimizer.step()
        logger.add_scalars('contrastive_loss',
                               {'c_loss': contrastive_loss},
                               self.niter)
        self.obs_center = (self.center_tau * self.obs_center + (1 - self.center_tau) * teacher_obs.mean(0, keepdim=True).mean(1)).detach()
        self.consensus_builder.update()


    def update(self, sample, agent_i, parallel=False, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        obs, acs, rews, next_obs, dones = sample

        if self.consensus:
            obs_tensor = torch.stack(obs, dim=-2)
            student_obs = self.consensus_builder.calc_student(obs_tensor)
            obs_prob = F.softmax(student_obs / self.online_temp, dim=-1)
            obs_id = obs_prob.max(-1)[1]
            obs_id_onehot = F.one_hot(obs_id, self.consensus_builder.consensus_builder_dim).float().permute(1, 0, 2)
            id_count = (obs_id_onehot.sum(0).sum(0) > 0).sum()

            next_obs_tensor = torch.stack(next_obs, dim=-2)
            next_student_obs = self.consensus_builder.calc_student(next_obs_tensor)
            next_obs_prob = F.softmax(next_student_obs / self.online_temp, dim=-1)
            next_obs_id = next_obs_prob.max(-1)[1]
            next_obs_id_onehot = F.one_hot(next_obs_id, self.consensus_builder.consensus_builder_dim).float().permute(1, 0, 2)

        curr_agent = self.agents[agent_i]

        curr_agent.critic_optimizer.zero_grad()
        if self.consensus:
            if self.alg_types[agent_i] == 'MADDPG':
                if self.discrete_action: # one-hot encode action
                    all_trgt_acs = [onehot_from_logits(pi(nobs, nids)) for pi, nobs, nids in
                                    zip(self.target_policies, next_obs, next_obs_id_onehot)]
                else:
                    all_trgt_acs = [pi(nobs, nids) for pi, nobs, nids in zip(self.target_policies,
                                                                 next_obs, next_obs_id_onehot)]
                trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs, *next_obs_id_onehot), dim=1)
            else:  # DDPG
                if self.discrete_action:
                    trgt_vf_in = torch.cat((next_obs[agent_i],
                                            onehot_from_logits(
                                                curr_agent.target_policy(
                                                    next_obs[agent_i], next_obs_id_onehot[agent_i]))),
                                           dim=1)
                else:
                    trgt_vf_in = torch.cat((next_obs[agent_i],
                                            curr_agent.target_policy(next_obs[agent_i], next_obs_id_onehot[agent_i]), next_obs_id_onehot[agent_i]),
                                           dim=1)
            target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                            curr_agent.target_critic(trgt_vf_in) *
                            (1 - dones[agent_i].view(-1, 1)))

            if self.alg_types[agent_i] == 'MADDPG':
                vf_in = torch.cat((*obs, *acs, *obs_id_onehot), dim=1)
            else:  # DDPG
                vf_in = torch.cat((obs[agent_i], acs[agent_i], obs_id_onehot[agent_i]), dim=1)
            actual_value = curr_agent.critic(vf_in)
            vf_loss = MSELoss(actual_value, target_value.detach())
            vf_loss.backward()
            if parallel:
                average_gradients(curr_agent.critic)
            torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)
            curr_agent.critic_optimizer.step()

            curr_agent.policy_optimizer.zero_grad()

            if self.discrete_action:
                # Forward pass as if onehot (hard=True) but backprop through a differentiable
                # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
                # through discrete categorical samples, but I'm not sure if that is
                # correct since it removes the assumption of a deterministic policy for
                # DDPG. Regardless, discrete policies don't seem to learn properly without it.
                curr_pol_out = curr_agent.policy(obs[agent_i], obs_id_onehot[agent_i])
                curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
            else:
                curr_pol_out = curr_agent.policy(obs[agent_i], obs_id_onehot[agent_i])
                curr_pol_vf_in = curr_pol_out
            if self.alg_types[agent_i] == 'MADDPG':
                all_pol_acs = []
                for i, pi, ob, id in zip(range(self.nagents), self.policies, obs, obs_id_onehot):
                    if i == agent_i:
                        all_pol_acs.append(curr_pol_vf_in)
                    elif self.discrete_action:
                        all_pol_acs.append(onehot_from_logits(pi(ob, id)))
                    else:
                        all_pol_acs.append(pi(ob, id))
                vf_in = torch.cat((*obs, *all_pol_acs, *obs_id_onehot), dim=1)
            else:  # DDPG
                vf_in = torch.cat((obs[agent_i], curr_pol_vf_in, obs_id_onehot[agent_i]),
                                  dim=1)
            pol_loss = -curr_agent.critic(vf_in).mean()
            pol_loss += (curr_pol_out**2).mean() * 1e-3
            pol_loss.backward()
            if parallel:
                average_gradients(curr_agent.policy)
            torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
            curr_agent.policy_optimizer.step()
            if logger is not None:
                logger.add_scalars('agent%i/losses' % agent_i,
                                   {'vf_loss': vf_loss,
                                    'pol_loss': pol_loss},
                                   self.niter)
                logger.add_scalars('id count',
                                   {'id_count': id_count},
                                   self.niter)
        else:
            if self.alg_types[agent_i] == 'MADDPG':
                if self.discrete_action: # one-hot encode action
                    all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
                                    zip(self.target_policies, next_obs)]
                else:
                    all_trgt_acs = [pi(nobs) for pi, nobs in zip(self.target_policies,
                                                                 next_obs)]
                trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1)
            else:  # DDPG
                if self.discrete_action:
                    trgt_vf_in = torch.cat((next_obs[agent_i],
                                            onehot_from_logits(
                                                curr_agent.target_policy(
                                                    next_obs[agent_i]))),
                                           dim=1)
                else:
                    trgt_vf_in = torch.cat((next_obs[agent_i],
                                            curr_agent.target_policy(next_obs[agent_i])),
                                           dim=1)
            target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                            curr_agent.target_critic(trgt_vf_in) *
                            (1 - dones[agent_i].view(-1, 1)))

            if self.alg_types[agent_i] == 'MADDPG':
                vf_in = torch.cat((*obs, *acs), dim=1)
            else:  # DDPG
                vf_in = torch.cat((obs[agent_i], acs[agent_i]), dim=1)
            actual_value = curr_agent.critic(vf_in)
            vf_loss = MSELoss(actual_value, target_value.detach())
            vf_loss.backward()
            if parallel:
                average_gradients(curr_agent.critic)
            torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)
            curr_agent.critic_optimizer.step()

            curr_agent.policy_optimizer.zero_grad()

            if self.discrete_action:
                # Forward pass as if onehot (hard=True) but backprop through a differentiable
                # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
                # through discrete categorical samples, but I'm not sure if that is
                # correct since it removes the assumption of a deterministic policy for
                # DDPG. Regardless, discrete policies don't seem to learn properly without it.
                curr_pol_out = curr_agent.policy(obs[agent_i])
                curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
            else:
                curr_pol_out = curr_agent.policy(obs[agent_i])
                curr_pol_vf_in = curr_pol_out
            if self.alg_types[agent_i] == 'MADDPG':
                all_pol_acs = []
                for i, pi, ob in zip(range(self.nagents), self.policies, obs):
                    if i == agent_i:
                        all_pol_acs.append(curr_pol_vf_in)
                    elif self.discrete_action:
                        all_pol_acs.append(onehot_from_logits(pi(ob)))
                    else:
                        all_pol_acs.append(pi(ob))
                vf_in = torch.cat((*obs, *all_pol_acs), dim=1)
            else:  # DDPG
                vf_in = torch.cat((obs[agent_i], curr_pol_vf_in),
                                  dim=1)
            pol_loss = -curr_agent.critic(vf_in).mean()
            pol_loss += (curr_pol_out**2).mean() * 1e-3
            pol_loss.backward()
            if parallel:
                average_gradients(curr_agent.policy)
            torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
            curr_agent.policy_optimizer.step()
            if logger is not None:
                logger.add_scalars('agent%i/losses' % agent_i,
                                   {'vf_loss': vf_loss,
                                    'pol_loss': pol_loss},
                                   self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1


    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.policy.train()
            a.critic.train()
            a.target_policy.train()
            a.target_critic.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device
        if self.consensus:
            if not self.consensus_builder_dev == device:
                self.consensus_builder = fn(self.consensus_builder)
                self.consensus_builder_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, agent_alg="MADDPG", adversary_alg="MADDPG",
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64, consensus=True, consensus_builder_dim=4):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for
                     atype in env.agent_types]
        for acsp, obsp, algtype in zip(env.action_space, env.observation_space,
                                       alg_types):
            num_in_pol = obsp.shape[0]
            if isinstance(acsp, Box):
                discrete_action = False
                get_shape = lambda x: x.shape[0]
            else:  # Discrete
                discrete_action = True
                get_shape = lambda x: x.n
            num_out_pol = get_shape(acsp)
            if algtype == "MADDPG":
                num_in_critic = 0
                for oobsp in env.observation_space:
                    num_in_critic += oobsp.shape[0]
                for oacsp in env.action_space:
                    num_in_critic += get_shape(oacsp)
                if consensus:
                    for a in env.agent_types:
                        num_in_critic += consensus_builder_dim
            else:
                num_in_critic = obsp.shape[0] + get_shape(acsp)
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'hidden_dim': hidden_dim,
                     'alg_types': alg_types,
                     'agent_init_params': agent_init_params,
                     'discrete_action': discrete_action,
                     'consensus': consensus,
                     'consensus_builder_dim': consensus_builder_dim}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return 