#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:19:50 2022

@author: vittoriogiammarino
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

from models.Models import BEAR_models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BEAR(object):
    def __init__(self, state_dim, action_dim, max_action,  num_qs=2, version=0, threshold=0.05, mode='auto', 
                 num_samples_match=100, mmd_sigma=20.0, lagrange_thresh=10.0, use_ensemble=True, kernel_type='laplacian', discount=0.99, tau=0.005):
        
        latent_dim = action_dim * 2
        self.actor = BEAR_models.RegularActor(state_dim, action_dim, max_action).to(device)
        self.actor_target = BEAR_models.RegularActor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = BEAR_models.EnsembleCritic(num_qs, state_dim, action_dim).to(device)
        self.critic_target = BEAR_models.EnsembleCritic(num_qs, state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.vae = BEAR_models.VAE(state_dim, action_dim, latent_dim, max_action).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=3e-4) 

        self.max_action = max_action
        self.action_dim = action_dim
        self.version = version
        self.threshold = threshold
        self.mode = mode
        self.num_qs = num_qs
        self.num_samples_match = num_samples_match
        self.mmd_sigma = mmd_sigma
        self.lagrange_thresh = lagrange_thresh
        self.use_ensemble = use_ensemble
        self.kernel_type = kernel_type
        self.discount = discount
        self.tau = tau
        
        if self.mode == 'auto':
            # Use lagrange multipliers on the constraint if set to auto mode 
            # for the purpose of maintaing support matching at all times
            self.log_lagrange2 = torch.randn((), requires_grad=True, device=device)
            self.lagrange2_opt = torch.optim.Adam([self.log_lagrange2], lr=1e-3)

        self.epoch = 0

    def mmd_loss_laplacian(self, samples1, samples2, sigma=10):
        """MMD constraint with Laplacian kernel for support matching"""
        # sigma is set to 10.0 for hopper, cheetah and 20 for walker/ant
        diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)  # B x N x N x d
        diff_x_x = torch.mean((-(diff_x_x.abs()).sum(-1)/(2.0 * sigma)).exp(), dim=(1,2))

        diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
        diff_x_y = torch.mean((-(diff_x_y.abs()).sum(-1)/(2.0 * sigma)).exp(), dim=(1, 2))

        diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)  # B x N x N x d
        diff_y_y = torch.mean((-(diff_y_y.abs()).sum(-1)/(2.0 * sigma)).exp(), dim=(1,2))

        overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
        return overall_loss
    
    def mmd_loss_gaussian(self, samples1, samples2, sigma=10):
        """MMD constraint with Gaussian Kernel support matching"""
        # sigma is set to 10.0 for hopper, cheetah and 20 for walker/ant
        diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)  # B x N x N x d
        diff_x_x = torch.mean((-(diff_x_x.pow(2)).sum(-1)/(2.0 * sigma)).exp(), dim=(1,2))

        diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
        diff_x_y = torch.mean((-(diff_x_y.pow(2)).sum(-1)/(2.0 * sigma)).exp(), dim=(1, 2))

        diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)  # B x N x N x d
        diff_y_y = torch.mean((-(diff_y_y.pow(2)).sum(-1)/(2.0 * sigma)).exp(), dim=(1,2))

        overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
        return overall_loss

    def kl_loss(self, samples1, state, sigma=0.2):
        """We just do likelihood, we make sure that the policy is close to the
           data in terms of the KL."""
        state_rep = state.unsqueeze(1).repeat(1, samples1.size(1), 1).view(-1, state.size(-1))
        samples1_reshape = samples1.view(-1, samples1.size(-1))
        samples1_log_pis = self.actor.log_pis(state=state_rep, raw_action=samples1_reshape)
        samples1_log_prob = samples1_log_pis.view(state.size(0), samples1.size(1))
        return (-samples1_log_prob).mean(1)
    
    def entropy_loss(self, samples1, state, sigma=0.2):
        state_rep = state.unsqueeze(1).repeat(1, samples1.size(1), 1).view(-1, state.size(-1))
        samples1_reshape = samples1.view(-1, samples1.size(-1))
        samples1_log_pis = self.actor.log_pis(state=state_rep, raw_action=samples1_reshape)
        samples1_log_prob = samples1_log_pis.view(state.size(0), samples1.size(1))
        # print (samples1_log_prob.min(), samples1_log_prob.max())
        samples1_prob = samples1_log_prob.clamp(min=-5, max=4).exp()
        return (samples1_prob).mean(1)
    
    def select_action(self, state):      
        """When running the actor, we just select action based on the max of the Q-function computed over
            samples from the policy -- which biases things to support."""
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).repeat(10, 1).to(device)
            action = self.actor(state)
            q1 = self.critic.q1(state, action)
            ind = q1.max(0)[1]
        return action[ind].cpu().data.numpy().flatten()
    
    def train(self, replay_buffer, batch_size=256):
        
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
            
        # Train the Behaviour cloning policy to be able to take more than 1 sample for MMD
        recon, mean, std = self.vae(state, action)
        recon_loss = F.mse_loss(recon, action)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * KL_loss

        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

        # Critic Training: In this step, we explicitly compute the actions 
        with torch.no_grad():
            # Duplicate next state 10 times
            next_state = torch.repeat_interleave(next_state, 10, 0)
            
            # Compute value of perturbed actions sampled from the VAE
            target_Qs = self.critic_target(next_state, self.actor_target(next_state))

            # Soft Clipped Double Q-learning 
            target_Q = 0.75 * target_Qs.min(0)[0] + 0.25 * target_Qs.max(0)[0]
            target_Q = target_Q.view(batch_size, -1).max(1)[0].view(-1, 1)
            target_Q = reward + not_done * self.discount * target_Q

        current_Qs = self.critic(state, action, with_var=False)

        critic_loss = F.mse_loss(current_Qs[0], target_Q) + F.mse_loss(current_Qs[1], target_Q) 

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Action Training
        # If you take less samples (but not too less, else it becomes statistically inefficient), it is closer to a uniform support set matching
        num_samples = self.num_samples_match
        sampled_actions, raw_sampled_actions = self.vae.decode_multiple(state, num_decode=num_samples)  # B x N x d
        actor_actions, raw_actor_actions = self.actor.sample_multiple(state, num_samples)#  num)

        # MMD done on raw actions (before tanh), to prevent gradient dying out due to saturation
        if self.kernel_type == 'gaussian':
            mmd_loss = self.mmd_loss_gaussian(raw_sampled_actions, raw_actor_actions, sigma=self.mmd_sigma)
        else:
            mmd_loss = self.mmd_loss_laplacian(raw_sampled_actions, raw_actor_actions, sigma=self.mmd_sigma)


        # Update through TD3 style
        critic_qs, std_q = self.critic.q_all(state, actor_actions[:, 0, :], with_var=True)
            
        if self.version == '0':
            critic_qs = critic_qs.min(0)[0]
        elif self.version == '1':
            critic_qs = critic_qs.mean(0)

        # We do support matching with a warmstart which happens to be reasonable around epoch 20 during training
        if self.epoch >= 8: 
            if self.mode == 'auto':
                actor_loss = (-critic_qs + self.log_lagrange2.exp()*(mmd_loss - self.threshold)).mean()
            else:
                actor_loss = (-critic_qs + 100.0*mmd_loss).mean()      # This coefficient is hardcoded, and is different for different tasks. I would suggest using auto, as that is the one used in the paper and works better.
        else:
            if self.mode == 'auto':
                actor_loss = (self.log_lagrange2.exp()*(mmd_loss - self.threshold)).mean()
            else:
                actor_loss = 100.0*mmd_loss.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.mode == 'auto':
            lagrange_loss = (self.log_lagrange2.exp() * (mmd_loss.detach() - self.threshold)).mean()

            self.lagrange2_opt.zero_grad()
            (-lagrange_loss).backward()
            self.lagrange2_opt.step() 
            self.log_lagrange2.data.clamp_(min=-5.0, max=self.lagrange_thresh)   
        
        # Update Target Networks 
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.epoch = self.epoch + 1