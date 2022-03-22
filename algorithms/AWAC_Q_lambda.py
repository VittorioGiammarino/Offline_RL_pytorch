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
    
class AWAC_Q_lambda(object):
    def __init__(self, state_dim, action_dim, action_space_cardinality, max_action, min_action, Entropy = False, alpha=0.2,
                 l_rate_actor=3e-4, l_rate_critic=3e-4, l_rate_alpha=3e-4, discount=0.99, tau=0.005, beta=3, critic_freq=2,
                 gae_gamma = 0.99, gae_lambda = 0.99, minibatch_size=64, num_epochs=10):
        
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
        self.gae_gamma = gae_gamma
        self.gae_lambda = gae_lambda
        self.minibatch_size = minibatch_size
        self.num_epochs = num_epochs
        
        self.Entropy = Entropy
        self.target_entropy = -torch.FloatTensor([action_dim]).to(device)
        self.log_alpha = torch.zeros(1, requires_grad=True, device="cuda")
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
        
    def Q_Lambda_Peng(self, replay_buffer, ntrajs, traj_size):
        states = []
        actions = []
        target_Q = []
        advantage = []
        gammas_list = []
        lambdas_list = []
        
        sampled_states, sampled_actions, sampled_rewards, sampled_lengths = replay_buffer.sample_trajectories(ntrajs, traj_size)
        
        for t in range(traj_size):
            gammas_list.append(self.gae_gamma**t)
            lambdas_list.append(self.gae_lambda**t)
        
        gammas = torch.FloatTensor(np.array(gammas_list)).to(device)
        lambdas = torch.FloatTensor(np.array(lambdas_list)).to(device)
        
        for l in range(ntrajs):
        
            with torch.no_grad():            
                episode_states = sampled_states[l]
                episode_actions = sampled_actions[l]
                episode_rewards = sampled_rewards[l].squeeze()    
                
                self.critic_target.eval()
                self.actor.eval()
                
                pi_action, log_pi, _ = self.actor.sample_SAC_continuous(episode_states)
                Q1_on, Q2_on = self.critic_target(episode_states, pi_action)
                
                if self.Entropy:
                    values = torch.min(Q1_on, Q2_on) - self.alpha*log_pi
                else:
                    values = torch.min(Q1_on, Q2_on)
                
                final_bootstrap = values[-1].unsqueeze(-1)
                next_values = values[1:]
                next_action_values = torch.cat((episode_rewards[:-1].unsqueeze(-1) + self.gae_gamma*next_values, final_bootstrap))
                
                episode_adv = []
                episode_Q = []
                
                for j in range(traj_size):
                    off_policy_adjust = torch.cat((torch.FloatTensor([[0.]]).to(device), values[j+1:]))     
                    episode_deltas = next_action_values[j:] - off_policy_adjust
                    episode_Q.append(((gammas*lambdas)[:traj_size-j].unsqueeze(-1)*episode_deltas).sum())
                    episode_adv.append(((gammas*lambdas)[:traj_size-j].unsqueeze(-1)*episode_deltas).sum() - values[j])

                episode_advantage = torch.FloatTensor(episode_adv).to(device)
                episode_target_Q = torch.FloatTensor(episode_Q).to(device)
                
                states.append(episode_states)
                actions.append(episode_actions)
                target_Q.append(episode_target_Q)
                advantage.append(episode_advantage)
            
        return states, actions, target_Q, advantage
    
    def Q_Lambda_Haru(self, replay_buffer, ntrajs, traj_size):
        states = []
        actions = []
        target_Q = []
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
        
            with torch.no_grad():            
                episode_states = sampled_states[l]
                episode_actions = sampled_actions[l]
                episode_rewards = sampled_rewards[l].squeeze()    
                
                self.critic_target.eval()
                self.actor.eval()
                
                Q1_off, Q2_off = self.critic_target(episode_states, episode_actions)
                values_off = torch.min(Q1_off, Q2_off) 
                
                pi_action, log_pi_on, _ = self.actor.sample_SAC_continuous(episode_states)
                Q1_on, Q2_on = self.critic_target(episode_states, pi_action)
                
                if self.Entropy:
                    values = torch.min(Q1_on, Q2_on) - self.alpha*log_pi_on
                else:
                    values = torch.min(Q1_on, Q2_on)
                
                final_bootstrap = values[-1].unsqueeze(-1)
                next_values = values[1:]
                next_action_values = torch.cat((episode_rewards[:-1].unsqueeze(-1) + self.gae_gamma*next_values, final_bootstrap))
                
                episode_adv = []
                episode_Q = []
                
                for j in range(traj_size):
                    off_policy_adjust = values_off[j:] 
                    episode_deltas = next_action_values[j:] - off_policy_adjust
                    episode_Q.append(values_off[j] + ((gammas*lambdas)[:traj_size-j].unsqueeze(-1)*episode_deltas).sum())
                    episode_adv.append(values_off[j] + ((gammas*lambdas)[:traj_size-j].unsqueeze(-1)*episode_deltas).sum() - values[j])

                episode_advantage = torch.FloatTensor(episode_adv).to(device)
                episode_target_Q = torch.FloatTensor(episode_Q).to(device)
                
                states.append(episode_states)
                actions.append(episode_actions)
                target_Q.append(episode_target_Q)
                advantage.append(episode_advantage)
            
        return states, actions, target_Q, advantage
    
    def TB_Lambda(self, replay_buffer, ntrajs, traj_size):
        states = []
        actions = []
        target_Q = []
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
        
            with torch.no_grad():            
                episode_states = sampled_states[l]
                episode_actions = sampled_actions[l]
                episode_rewards = sampled_rewards[l].squeeze()    
                
                self.critic_target.eval()
                self.actor.eval()
                
                Q1_off, Q2_off = self.critic_target(episode_states, episode_actions)
                values_off = torch.min(Q1_off, Q2_off)
                
                pi_action, log_pi_on, _ = self.actor.sample_SAC_continuous(episode_states)
                Q1_on, Q2_on = self.critic_target(episode_states, pi_action)
                
                if self.Entropy:
                    values = torch.min(Q1_on, Q2_on) - self.alpha*log_pi_on
                else:
                    values = torch.min(Q1_on, Q2_on)
                    
                final_bootstrap = values[-1].unsqueeze(-1)
                next_values = values[1:]
                next_action_values = torch.cat((episode_rewards[:-1].unsqueeze(-1) + self.gae_gamma*next_values, final_bootstrap))
                
                if self.action_space == "Discrete":
                    _, log_prob_episode_full = self.actor.sample_log(episode_states, episode_actions)
    
                elif self.action_space == "Continuous": 
                    log_prob_episode_full = self.actor.sample_log(episode_states, episode_actions)
                
                episode_adv = []
                episode_Q = []

                for j in range(traj_size):
                    
                    try:
                        log_prob_episode = log_prob_episode_full[j:]    
                        r = torch.clamp((torch.exp(log_prob_episode)).squeeze(),0,1)
                        pi_adjust = torch.FloatTensor([(r[:k]).prod() for k in range(1, len(r))]).to(device)
                        pi_adjust_full = torch.cat((torch.FloatTensor([1.]).to(device), pi_adjust))
                        
                    except:
                        pi_adjust_full = torch.FloatTensor([1.]).to(device)
                    
                    off_policy_adjust = values_off[j:] 
                    episode_deltas = next_action_values[j:] - off_policy_adjust
                    episode_Q.append(values_off[j] + ((gammas*lambdas)[:traj_size-j].unsqueeze(-1)*pi_adjust_full*episode_deltas).sum())
                    episode_adv.append(values_off[j] + ((gammas*lambdas)[:traj_size-j].unsqueeze(-1)*pi_adjust_full*episode_deltas).sum() - values[j])

                episode_advantage = torch.FloatTensor(episode_adv).to(device)
                episode_target_Q = torch.FloatTensor(episode_Q).to(device)
                
                states.append(episode_states)
                actions.append(episode_actions)
                target_Q.append(episode_target_Q)
                advantage.append(episode_advantage)
            
        return states, actions, target_Q, advantage
    
    def train(self, states, actions, target_Q, advantage):
        self.total_it += 1

        rollout_states = torch.cat(states)
        rollout_actions = torch.cat(actions)
        rollout_target_Q = torch.cat(target_Q)
        rollout_advantage = torch.cat(advantage)
        
        rollout_advantage = (rollout_advantage-rollout_advantage.mean())/(rollout_advantage.std()+1e-6)
        
        self.critic.train()
        self.actor.train()
        
        self.num_steps_per_rollout = len(rollout_advantage)
        
        max_steps = self.num_epochs * (self.num_steps_per_rollout // self.minibatch_size)
        
        for _ in range(max_steps):
            
            minibatch_indices = np.random.choice(range(self.num_steps_per_rollout), self.minibatch_size, False)
            batch_states = rollout_states[minibatch_indices]
            batch_actions = rollout_actions[minibatch_indices]
            batch_target_Q = rollout_target_Q[minibatch_indices]
            batch_advantage = rollout_advantage[minibatch_indices]      
            
            if self.action_space == "Discrete":
                log_prob, log_prob_rollout = self.actor.sample_log(batch_states, batch_actions)

            elif self.action_space == "Continuous": 
                log_prob_rollout = self.actor.sample_log(batch_states, batch_actions)
                
            r = (log_prob_rollout).squeeze()
            weights = F.softmax(batch_advantage/self.beta, dim=0).detach()
            L_clip = r*weights
            
            if self.action_space == "Discrete":
                Q1, Q2 = self.critic(batch_states)
                current_Q1 = Q1.gather(1, batch_actions.detach().long()) 
                current_Q2 = Q2.gather(1, batch_actions.detach().long()) 
            
            elif self.action_space == "Continuous":
                #current Q estimates
                current_Q1, current_Q2 = self.critic(batch_states, batch_actions)
            
            critic_loss = F.mse_loss(current_Q1.squeeze(), batch_target_Q) + F.mse_loss(current_Q1.squeeze(), batch_target_Q)

            self.critic_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            
            if self.Entropy:
                _, log_pi_state, _ = self.actor.sample_SAC_continuous(batch_states)
                loss = (-1) * (L_clip - self.alpha*log_pi_state).mean() + critic_loss
            else:
                loss = (-1)*(L_clip).mean() + critic_loss

            loss.backward()
            self.critic_optimizer.step()
            self.actor_optimizer.step() 
            
            if self.Entropy:    
                _, log_pi_state, _ = self.actor.sample_SAC_continuous(batch_states)
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
		