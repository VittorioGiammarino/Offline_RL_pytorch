#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:05:53 2022

@author: vittoriogiammarino
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

from models.Models import BCQ_models 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BCQ(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, lmbda=0.75, phi=0.05):
        
        latent_dim = action_dim * 2

        self.actor = BCQ_models.Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = BCQ_models.Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = BCQ_models.Critic(state_dim, action_dim).to(device)
        self.critic_target = BCQ_models.Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.vae = BCQ_models.VAE(state_dim, action_dim, latent_dim, max_action).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters()) 

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.lmbda = lmbda
        self.device = device

    def select_action(self, state):     
        with torch.no_grad():
                state = torch.FloatTensor(state.reshape(1, -1)).repeat(10, 1).to(device)
                action = self.actor(state, self.vae.decode(state))
                q1 = self.critic.q1(state, action)
                ind = q1.max(0)[1]
        return action[ind].cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
  
        # Variational Auto-Encoder Training
        recon, mean, std = self.vae(state, action)
        recon_loss = F.mse_loss(recon, action)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * KL_loss
  
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()
  
        # Critic Training
        with torch.no_grad():
            # Duplicate next state 10 times
            next_state = torch.repeat_interleave(next_state, 10, 0)
            
            # Compute value of perturbed actions sampled from the VAE
            target_Q1, target_Q2 = self.critic_target(next_state, self.actor_target(next_state, self.vae.decode(next_state)))
  
            # Soft Clipped Double Q-learning 
            target_Q = self.lmbda*torch.min(target_Q1, target_Q2) + (1 - self.lmbda)*torch.max(target_Q1, target_Q2)
            # Take max over each action sampled from the VAE
            target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)
  
            target_Q = reward + not_done * self.discount * target_Q
  
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
  
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
  
        # Pertubation Model / Action Training
        sampled_actions = self.vae.decode(state)
        perturbed_actions = self.actor(state, sampled_actions)
  
        # Update through DPG
        actor_loss = -self.critic.q1(state, perturbed_actions).mean()
            
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
  
        # Update Target Networks 
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
  
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
    def save_actor(self, filename):
        option = 0
        torch.save(self.actor.state_dict(), filename + f"_pi_lo_option_{option}")
        torch.save(self.actor_optimizer.state_dict(), filename + f"_pi_lo_optimizer_option_{option}")
        
    def load_actor(self, filename):
        option = 0
        self.actor.load_state_dict(torch.load(filename + f"_pi_lo_option_{option}"))
        self.actor_optimizer.load_state_dict(torch.load(filename + f"_pi_lo_optimizer_option_{option}"))
        
