#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 16:37:45 2022

@author: vittoriogiammarino
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Models import SAC_models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class AWAC(object):
    def __init__(self, state_dim, action_dim, action_space_cardinality, max_action, min_action, Entropy = True, alpha=0.2,
                 l_rate_actor=3e-4, l_rate_critic=3e-4, l_rate_alpha=3e-4, discount=0.99, tau=0.005, beta=3, critic_freq=2):
        
        if np.isinf(action_space_cardinality):
            self.actor = SAC_models.Actor(state_dim, action_dim, max_action).to(device)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=l_rate_actor)
            self.action_space = "Continuous"
            
            self.critic = SAC_models.Critic_flat(state_dim, action_dim).to(device)
            self.critic_target = copy.deepcopy(self.critic).to(device)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=l_rate_critic)
        else:
            self.actor = SAC_models.Actor_discrete(state_dim, action_space_cardinality).to(device)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=l_rate_actor)
            self.action_space = "Discrete"
                    
            self.critic = SAC_models.Critic_flat_discrete(state_dim, action_space_cardinality).to(device)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=l_rate_critic)   

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space_cardinality = action_space_cardinality
        self.max_action = max_action
        
        self.discount = discount
        self.tau = tau
        self.beta = beta
        self.critic_freq = critic_freq
        
        self.Entropy = Entropy
        self.target_entropy = -torch.FloatTensor([action_dim]).to(device)
        try:
            self.log_alpha = torch.zeros(1, requires_grad=True, device="cuda")
        except:
            self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr = l_rate_alpha) 
        self.alpha = alpha

        self.total_it = 0
        
    def select_action(self, state):
        if self.action_space == "Discrete":
            state = torch.FloatTensor(state.reshape(1,-1)).to(device)
            action, _ = self.actor.sample(state)
            return int((action).cpu().data.numpy().flatten())
        
        if self.action_space == "Continuous":
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            action, _, _ = self.actor.sample_SAC_continuous(state)
            return (action).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

		# Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
			
            if self.action_space == "Discrete":
                next_action, _ = self.actor.sample(next_state)
                log_pi_next_state, _ = self.actor.sample_log(next_state, torch.LongTensor(next_action.detach().numpy().flatten()))
                next_action_prob = self.actor(next_state)
                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_state)
                target_Q = next_action_prob*(torch.min(target_Q1, target_Q2))
                target_Q = reward + not_done * self.discount * target_Q.sum(dim=1).unsqueeze(-1)
                
            elif self.action_space == "Continuous":
                next_action, log_pi_next_state, _ = self.actor.sample_SAC_continuous(next_state)
                    
                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                
                if self.Entropy:
                    target_Q = torch.min(target_Q1, target_Q2) - self.alpha*log_pi_next_state
                    target_Q = reward + not_done * self.discount * target_Q
                else:
                    target_Q = torch.min(target_Q1, target_Q2)
                    target_Q = reward + not_done * self.discount * target_Q

        if self.action_space == "Discrete":
            Q1, Q2 = self.critic(state)
            current_Q1 = Q1.gather(1, action.detach().long()) 
            current_Q2 = Q2.gather(1, action.detach().long()) 
        
        elif self.action_space == "Continuous":
            #current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.action_space == "Discrete":
            Q1, Q2 = self.critic(state)
            pi_action, _ = self.actor.sample(state)
            log_pi_state, _ = self.actor.sample_log(state, torch.LongTensor(pi_action.detach().numpy().flatten()))
            action_prob = self.actor(state)
            value_function = torch.min((action_prob*Q1).sum(dim=1),(action_prob*Q2).sum(dim=1))
            
            old_Q1 = Q1.gather(1, action.detach().long()) 
            old_Q2 = Q2.gather(1, action.detach().long()) 
            old_Q = torch.min(old_Q1, old_Q2)
            adv_pi = old_Q - value_function
            weights = F.softmax(adv_pi/self.beta, dim=0).detach()
            log_pi_state_action = self.actor.sample_log(state, action)
            
            actor_loss = (-1)*(log_pi_state_action*weights).mean()
            
        elif self.action_space == "Continuous":
            pi_action, log_pi_state, _ = self.actor.sample_SAC_continuous(state)
            Q1, Q2 = self.critic(state, pi_action)
            value_function = torch.min(Q1,Q2)
            
            Q1_old, Q2_old = self.critic(state, action)
            Q_old_actions = torch.min(Q1_old, Q2_old)
            
            adv_pi = Q_old_actions - value_function
            weights = F.softmax(adv_pi/self.beta, dim=0).detach()
            log_pi_state_action = self.actor.sample_log(state, action)

            if self.Entropy:
                actor_loss = (-1)*(-self.alpha*log_pi_state + log_pi_state_action*weights).mean()
            else:
                actor_loss = (-1)*(log_pi_state_action*weights).mean()
			
        # Optimize the actor 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        if self.Entropy:    
            if self.action_space == "Discrete":
                alpha_loss = -(self.log_alpha * (torch.sum(log_pi_state * action_prob, dim=1) + self.target_entropy).detach()).mean()
            
            elif self.action_space == "Continuous":
                alpha_loss = -(self.log_alpha * (log_pi_state + self.target_entropy).detach()).mean()
    
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
    
            self.alpha = self.log_alpha.exp()

        # Update the frozen target models
        if self.total_it % self.critic_freq == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_actor(self, filename):
        option = 0
        torch.save(self.actor.state_dict(), filename + f"_pi_lo_option_{option}")
        torch.save(self.actor_optimizer.state_dict(), filename + f"_pi_lo_optimizer_option_{option}")
        
    def load_actor(self, filename):
        option = 0
        self.actor.load_state_dict(torch.load(filename + f"_pi_lo_option_{option}"))
        self.actor_optimizer.load_state_dict(torch.load(filename + f"_pi_lo_optimizer_option_{option}"))
        
    def save_critic(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
    
    def load_critic(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
		