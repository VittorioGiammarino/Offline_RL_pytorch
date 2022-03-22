#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 09:54:01 2021

@author: vittorio
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Models import PPO_off_models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPO:
    def __init__(self, state_dim, action_dim, action_space_cardinality, max_action, min_action,  
                 num_steps_per_rollout=5000, l_rate_actor=3e-4, gae_gamma = 0.99, gae_lambda = 0.99, 
                 epsilon = 0.2, c1 = 1, c2 = 1e-2, minibatch_size=64, num_epochs=10):
        
        if np.isinf(action_space_cardinality):
            self.actor = PPO_off_models.Actor(state_dim, action_dim, max_action).to(device)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=l_rate_actor)
            self.action_space = "Continuous"
            
        else:
            self.actor = PPO_off_models.Actor_discrete(state_dim, action_space_cardinality).to(device)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=l_rate_actor)
            self.action_space = "Discrete"
            
        self.value_function = PPO_off_models.Value_net(state_dim).to(device)
        self.value_function_optimizer = torch.optim.Adam(self.value_function.parameters(), lr=l_rate_actor)
      
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space_cardinality = action_space_cardinality
        self.max_action = max_action
        
        self.num_steps_per_rollout = num_steps_per_rollout
        self.gae_gamma = gae_gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.minibatch_size = minibatch_size
        self.num_epochs = num_epochs
        
        self.Total_t = 0
        self.Total_iter = 0
        self.states = []
        self.actions = []
        self.returns = []
        self.advantage = []
        self.gammas = []
        
    def reset_counters(self):
        self.Total_t = 0
        self.Total_iter = 0
        
    def select_action(self, state):
        with torch.no_grad():
            if self.action_space == "Discrete":
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                action, _ = self.actor.sample(state)
                return int((action).cpu().data.numpy().flatten())
            
            if self.action_space == "Continuous":
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                action, _, _ = self.actor.sample(state)
                return (action).cpu().data.numpy().flatten()
        
    def GAE(self, replay_buffer, ntrajs, traj_size):
        states = []
        actions = []
        returns = []
        advantage = []
        gammas_list = []
        lambdas_list = []
        
        sampled_states, sampled_actions, sampled_rewards, sampled_lengths = replay_buffer.sample_trajectories(ntrajs, traj_size)
        gammas_list = []
        lambdas_list = []
        
        for t in range(traj_size):
            gammas_list.append(self.gae_gamma**t)
            lambdas_list.append(self.gae_lambda**t)
        
        gammas = torch.FloatTensor(np.array(gammas_list)).to(device)
        lambdas = torch.FloatTensor(np.array(lambdas_list)).to(device)
        
        for l in range(ntrajs):
            
            episode_states = sampled_states[l]
            episode_actions = sampled_actions[l]
            episode_rewards = sampled_rewards[l].squeeze()    
                
            episode_discounted_rewards = gammas*episode_rewards
            episode_discounted_returns = torch.FloatTensor([sum(episode_discounted_rewards[i:]) for i in range(traj_size)]).to(device)
            episode_returns = episode_discounted_returns
            
            self.returns.append(episode_returns)
            self.value_function.eval()
            current_values = self.value_function(episode_states).detach()
            next_values = torch.cat((self.value_function(episode_states)[1:].detach(), torch.FloatTensor([[0.]]).to(device)))
            episode_deltas = episode_rewards.unsqueeze(-1) + self.gae_gamma*next_values - current_values
            episode_advantage = torch.FloatTensor([((gammas*lambdas)[:traj_size-j].unsqueeze(-1)*episode_deltas[j:]).sum() for j in range(traj_size)]).to(device)
            
            states.append(episode_states)
            actions.append(episode_actions)
            returns.append(episode_returns)
            advantage.append(episode_advantage)
            
        return states, actions, returns, advantage
            
    def train(self, states, actions, returns, advantage, Entropy = False):
        
        rollout_states = torch.cat(states)
        rollout_actions = torch.cat(actions)
        rollout_returns = torch.cat(returns)
        rollout_advantage = torch.cat(advantage)
        
        rollout_advantage = (rollout_advantage-rollout_advantage.mean())/(rollout_advantage.std()+1e-6)
        
        self.actor.eval()
        
        if self.action_space == "Discrete":
            _, old_log_prob_rollout = self.actor.sample_log(rollout_states, rollout_actions)
            old_log_prob_rollout = old_log_prob_rollout.detach()
        elif self.action_space == "Continuous": 
            old_log_prob_rollout = self.actor.sample_log(rollout_states, rollout_actions)
            old_log_prob_rollout = old_log_prob_rollout.detach()
        
        self.value_function.train()
        self.actor.train()
        
        max_steps = self.num_epochs * (self.num_steps_per_rollout // self.minibatch_size)
        
        for _ in range(max_steps):
            
            minibatch_indices = np.random.choice(range(self.num_steps_per_rollout), self.minibatch_size, False)
            batch_states=rollout_states[minibatch_indices]
            batch_actions = rollout_actions[minibatch_indices]
            batch_returns = rollout_returns[minibatch_indices]
            batch_advantage = rollout_advantage[minibatch_indices]      
            
            if self.action_space == "Discrete":
                log_prob, log_prob_rollout = self.actor.sample_log(batch_states, batch_actions)

            elif self.action_space == "Continuous": 
                log_prob_rollout = self.actor.sample_log(batch_states, batch_actions)
                
            batch_old_log_pi = old_log_prob_rollout[minibatch_indices]
                
            r = (torch.exp(log_prob_rollout - batch_old_log_pi)).squeeze()
            L_clip = torch.minimum(r*batch_advantage, torch.clip(r, 1-self.epsilon, 1+self.epsilon)*batch_advantage)
            L_vf = (self.value_function(batch_states).squeeze() - batch_returns)**2
            
            # r_old = (torch.exp(batch_old_log_pi)).squeeze()
            # r = (torch.exp(log_prob_rollout)).squeeze()
            # L_clip = torch.minimum(r*batch_advantage, torch.clip(r, r_old-self.epsilon, r_old+self.epsilon)*batch_advantage)
            # L_vf = (self.value_function(batch_states).squeeze() - batch_returns)**2
            
            if self.action_space == "Discrete":
                if Entropy:
                    S = (-1)*torch.sum(torch.exp(log_prob)*log_prob, 1)
                else:
                    S = torch.zeros_like(torch.sum(torch.exp(log_prob)*log_prob, 1))
                    
            elif self.action_space == "Continuous": 
                if Entropy:
                    S = self.actor.Distb(batch_states).entropy()
                else:
                    S = torch.zeros_like(self.actor.Distb(batch_states).entropy())
                
            self.value_function_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            loss = (-1) * (L_clip - self.c1 * L_vf + self.c2 * S).mean()
            loss.backward()
            self.value_function_optimizer.step()
            self.actor_optimizer.step()        
        
    def save_actor(self, filename):
        option = 0
        torch.save(self.actor.state_dict(), filename + f"_pi_lo_option_{option}")
        torch.save(self.actor_optimizer.state_dict(), filename + f"_pi_lo_optimizer_option_{option}")
    
    def load_actor(self, filename):
        option = 0
        self.actor.load_state_dict(torch.load(filename + f"_pi_lo_option_{option}"))
        self.actor_optimizer.load_state_dict(torch.load(filename + f"_pi_lo_optimizer_option_{option}"))

    def save_critic(self, filename):
        torch.save(self.value_function.state_dict(), filename + "_value_function")
        torch.save(self.value_function_optimizer.state_dict(), filename + "_value_function_optimizer")
    
    def load_critic(self, filename):
        self.value_function.load_state_dict(torch.load(filename + "_value_function"))      
        self.value_function_optimizer.load_state_dict(torch.load(filename + "_value_function_optimizer")) 
        

        
        
        
        
        
        

            
            
        
            
            
            

        