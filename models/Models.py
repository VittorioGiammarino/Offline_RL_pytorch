#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 17:27:18 2022

@author: vittoriogiammarino
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from torch.distributions.categorical import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

def atanh(x):
    one_plus_x = (1 + x).clamp(min=1e-7)
    one_minus_x = (1 - x).clamp(min=1e-7)
    return 0.5*torch.log(one_plus_x/ one_minus_x)

class BC_models:
    class RegularActor(nn.Module):
        """A probabilistic actor which does regular stochastic mapping of actions from states"""
        def __init__(self, state_dim, action_dim, max_action,):
            super(BC_models.RegularActor, self).__init__()
            self.l1 = nn.Linear(state_dim, 400)
            self.l2 = nn.Linear(400, 300)
            self.mean = nn.Linear(300, action_dim)
            self.log_std = nn.Linear(300, action_dim)
            self.max_action = max_action
        
        def forward(self, state):
            a = F.relu(self.l1(state))
            a = F.relu(self.l2(a))
            mean_a = self.mean(a)
            log_std_a = self.log_std(a)
            
            std_a = torch.exp(log_std_a)
            z = mean_a + std_a * torch.FloatTensor(np.random.normal(0, 1, size=(std_a.size()))).to(device) 
            return torch.FloatTensor(self.max_action).to(device) * torch.tanh(z)
    
        def sample_multiple(self, state, num_sample=10):
            a = F.relu(self.l1(state))
            a = F.relu(self.l2(a))
            mean_a = self.mean(a)
            log_std_a = self.log_std(a)
            
            std_a = torch.exp(log_std_a)
            # This trick stabilizes learning (clipping gaussian to a smaller range)
            z = mean_a.unsqueeze(1) + std_a.unsqueeze(1) * torch.FloatTensor(np.random.normal(0, 1, size=(std_a.size(0), num_sample, std_a.size(1)))).to(device).clamp(-0.5, 0.5)
            return torch.FloatTensor(self.max_action).to(device) * torch.tanh(z), z 
    
        def log_pis(self, state, action=None, raw_action=None):
            """Get log pis for the model."""
            a = F.relu(self.l1(state))
            a = F.relu(self.l2(a))
            mean_a = self.mean(a)
            log_std_a = self.log_std(a)
            std_a = torch.exp(log_std_a)
            normal_dist = td.Normal(loc=mean_a, scale=std_a, validate_args=True)
            if raw_action is None:
                raw_action = atanh(action)
            else:
                action = torch.tanh(raw_action)
            log_normal = normal_dist.log_prob(raw_action)
            log_pis = log_normal.sum(-1)
            log_pis = log_pis - (1.0 - action**2).clamp(min=1e-6).log().sum(-1)
            return log_pis

class BCQ_models:
    class Actor(nn.Module):
    	def __init__(self, state_dim, action_dim, max_action, phi=0.05):
    		super(BCQ_models.Actor, self).__init__()
    		self.l1 = nn.Linear(state_dim + action_dim, 400)
    		self.l2 = nn.Linear(400, 300)
    		self.l3 = nn.Linear(300, action_dim)
    		
    		self.max_action = max_action[0]
    		self.phi = phi
    
    	def forward(self, state, action):
    		a = F.relu(self.l1(torch.cat([state, action], 1)))
    		a = F.relu(self.l2(a))
    		a = self.phi * self.max_action * torch.tanh(self.l3(a))
    		return (a + action).clamp(-self.max_action, self.max_action)
        
    class Critic(nn.Module):
    	def __init__(self, state_dim, action_dim):
    		super(BCQ_models.Critic, self).__init__()
    		self.l1 = nn.Linear(state_dim + action_dim, 400)
    		self.l2 = nn.Linear(400, 300)
    		self.l3 = nn.Linear(300, 1)
    
    		self.l4 = nn.Linear(state_dim + action_dim, 400)
    		self.l5 = nn.Linear(400, 300)
    		self.l6 = nn.Linear(300, 1)
    
    	def forward(self, state, action):
    		q1 = F.relu(self.l1(torch.cat([state, action], 1)))
    		q1 = F.relu(self.l2(q1))
    		q1 = self.l3(q1)
    
    		q2 = F.relu(self.l4(torch.cat([state, action], 1)))
    		q2 = F.relu(self.l5(q2))
    		q2 = self.l6(q2)
    		return q1, q2
    
    	def q1(self, state, action):
    		q1 = F.relu(self.l1(torch.cat([state, action], 1)))
    		q1 = F.relu(self.l2(q1))
    		q1 = self.l3(q1)
    		return q1

    # Vanilla Variational Auto-Encoder 
    class VAE(nn.Module):
    	def __init__(self, state_dim, action_dim, latent_dim, max_action):
    		super(BCQ_models.VAE, self).__init__()
    		self.e1 = nn.Linear(state_dim + action_dim, 750)
    		self.e2 = nn.Linear(750, 750)
    
    		self.mean = nn.Linear(750, latent_dim)
    		self.log_std = nn.Linear(750, latent_dim)
    
    		self.d1 = nn.Linear(state_dim + latent_dim, 750)
    		self.d2 = nn.Linear(750, 750)
    		self.d3 = nn.Linear(750, action_dim)
    
    		self.max_action = max_action
    		self.latent_dim = latent_dim
    
    	def forward(self, state, action):
    		z = F.relu(self.e1(torch.cat([state, action], 1)))
    		z = F.relu(self.e2(z))
    
    		mean = self.mean(z)
    		# Clamped for numerical stability 
    		log_std = self.log_std(z).clamp(-4, 15)
    		std = torch.exp(log_std)
    		z = mean + std * torch.randn_like(std)
    		
    		u = self.decode(state, z)
    
    		return u, mean, std
    
    	def decode(self, state, z=None):
    		# When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
    		if z is None:
    			z = torch.randn((state.shape[0], self.latent_dim)).to(device).clamp(-0.5,0.5)
    
    		a = F.relu(self.d1(torch.cat([state, z], 1)))
    		a = F.relu(self.d2(a))
    		return torch.FloatTensor(self.max_action).to(device) * torch.tanh(self.d3(a))


class TD3_BC_models:
    class Actor(nn.Module):
    	def __init__(self, state_dim, action_dim, max_action):
    		super(TD3_BC_models.Actor, self).__init__()
    
    		self.l1 = nn.Linear(state_dim, 256)
    		self.l2 = nn.Linear(256, 256)
    		self.l3 = nn.Linear(256, action_dim)
    		
    		self.max_action = max_action[0]
    		
    	def forward(self, state):
    		a = F.relu(self.l1(state))
    		a = F.relu(self.l2(a))
    		return self.max_action * torch.tanh(self.l3(a))
    
    
    class Critic(nn.Module):
    	def __init__(self, state_dim, action_dim):
    		super(TD3_BC_models.Critic, self).__init__()
    
    		# Q1 architecture
    		self.l1 = nn.Linear(state_dim + action_dim, 256)
    		self.l2 = nn.Linear(256, 256)
    		self.l3 = nn.Linear(256, 1)
    
    		# Q2 architecture
    		self.l4 = nn.Linear(state_dim + action_dim, 256)
    		self.l5 = nn.Linear(256, 256)
    		self.l6 = nn.Linear(256, 1)
    
    	def forward(self, state, action):
    		sa = torch.cat([state, action], 1)
    
    		q1 = F.relu(self.l1(sa))
    		q1 = F.relu(self.l2(q1))
    		q1 = self.l3(q1)
    
    		q2 = F.relu(self.l4(sa))
    		q2 = F.relu(self.l5(q2))
    		q2 = self.l6(q2)
    		return q1, q2
    
    	def Q1(self, state, action):
    		sa = torch.cat([state, action], 1)
    
    		q1 = F.relu(self.l1(sa))
    		q1 = F.relu(self.l2(q1))
    		q1 = self.l3(q1)
    		return q1   
        
        
class BEAR_models:
    class RegularActor(nn.Module):
        """A probabilistic actor which does regular stochastic mapping of actions from states"""
        def __init__(self, state_dim, action_dim, max_action,):
            super(BEAR_models.RegularActor, self).__init__()
            self.l1 = nn.Linear(state_dim, 400)
            self.l2 = nn.Linear(400, 300)
            self.mean = nn.Linear(300, action_dim)
            self.log_std = nn.Linear(300, action_dim)
            self.max_action = max_action[0]
        
        def forward(self, state):
            a = F.relu(self.l1(state))
            a = F.relu(self.l2(a))
            mean_a = self.mean(a)
            log_std_a = self.log_std(a)
            
            std_a = torch.exp(log_std_a)
            z = mean_a + std_a * torch.FloatTensor(np.random.normal(0, 1, size=(std_a.size()))).to(device) 
            return self.max_action*torch.tanh(z)

        def sample_multiple(self, state, num_sample=10):
            a = F.relu(self.l1(state))
            a = F.relu(self.l2(a))
            mean_a = self.mean(a)
            log_std_a = self.log_std(a)
            
            std_a = torch.exp(log_std_a)
            # This trick stabilizes learning (clipping gaussian to a smaller range)
            z = mean_a.unsqueeze(1) + std_a.unsqueeze(1) * torch.FloatTensor(np.random.normal(0, 1, size=(std_a.size(0), num_sample, std_a.size(1)))).to(device).clamp(-0.5, 0.5)
            return self.max_action * torch.tanh(z), z 

        def log_pis(self, state, action=None, raw_action=None):
            """Get log pis for the model."""
            a = F.relu(self.l1(state))
            a = F.relu(self.l2(a))
            mean_a = self.mean(a)
            log_std_a = self.log_std(a)
            std_a = torch.exp(log_std_a)
            normal_dist = td.Normal(loc=mean_a, scale=std_a, validate_args=True)
            if raw_action is None:
                raw_action = atanh(action)
            else:
                action = torch.tanh(raw_action)
            log_normal = normal_dist.log_prob(raw_action)
            log_pis = log_normal.sum(-1)
            log_pis = log_pis - (1.0 - action**2).clamp(min=1e-6).log().sum(-1)
            return log_pis

    class EnsembleCritic(nn.Module):
        """ Critic which does have a network of 2 Q-functions"""
        def __init__(self, num_qs, state_dim, action_dim):
            super(BEAR_models.EnsembleCritic, self).__init__()
            
            self.num_qs = num_qs

            self.l1 = nn.Linear(state_dim + action_dim, 400)
            self.l2 = nn.Linear(400, 300)
            self.l3 = nn.Linear(300, 1)

            self.l4 = nn.Linear(state_dim + action_dim, 400)
            self.l5 = nn.Linear(400, 300)
            self.l6 = nn.Linear(300, 1)

        def forward(self, state, action, with_var=False):
            all_qs = []
            
            q1 = F.relu(self.l1(torch.cat([state, action], 1)))
            q1 = F.relu(self.l2(q1))
            q1 = self.l3(q1)

            q2 = F.relu(self.l4(torch.cat([state, action], 1)))
            q2 = F.relu(self.l5(q2))
            q2 = self.l6(q2)

            all_qs = torch.cat(
                [q1.unsqueeze(0), q2.unsqueeze(0),], 0) # q3.unsqueeze(0), q4.unsqueeze(0)], 0)   # Num_q x B x 1
            if with_var:
                std_q = torch.std(all_qs, dim=0, keepdim=False, unbiased=False)
                return all_qs, std_q
            return all_qs

        def q1(self, state, action):
            q1 = F.relu(self.l1(torch.cat([state, action], 1)))
            q1 = F.relu(self.l2(q1))
            q1 = self.l3(q1)
            return q1
        
        def q_all(self, state, action, with_var=False):
            all_qs = []
            
            q1 = F.relu(self.l1(torch.cat([state, action], 1)))
            q1 = F.relu(self.l2(q1))
            q1 = self.l3(q1)

            q2 = F.relu(self.l4(torch.cat([state, action], 1)))
            q2 = F.relu(self.l5(q2))
            q2 = self.l6(q2)

            all_qs = torch.cat(
                [q1.unsqueeze(0), q2.unsqueeze(0),], 0) # q3.unsqueeze(0), q4.unsqueeze(0)], 0)  # Num_q x B x 1
            if with_var:
                std_q = torch.std(all_qs, dim=0, keepdim=False, unbiased=False)
                return all_qs, std_q
            return all_qs

    # Vanilla Variational Auto-Encoder 
    class VAE(nn.Module):
        """VAE Based behavior cloning also used in Fujimoto et.al. (ICML 2019)"""
        def __init__(self, state_dim, action_dim, latent_dim, max_action):
            super(BEAR_models.VAE, self).__init__()
            self.e1 = nn.Linear(state_dim + action_dim, 750)
            self.e2 = nn.Linear(750, 750)

            self.mean = nn.Linear(750, latent_dim)
            self.log_std = nn.Linear(750, latent_dim)

            self.d1 = nn.Linear(state_dim + latent_dim, 750)
            self.d2 = nn.Linear(750, 750)
            self.d3 = nn.Linear(750, action_dim)

            self.max_action = max_action[0]
            self.latent_dim = latent_dim


        def forward(self, state, action):
            z = F.relu(self.e1(torch.cat([state, action], 1)))
            z = F.relu(self.e2(z))

            mean = self.mean(z)
            # Clamped for numerical stability 
            log_std = self.log_std(z).clamp(-4, 15)
            std = torch.exp(log_std)
            z = mean + std * torch.FloatTensor(np.random.normal(0, 1, size=(std.size()))).to(device) 
            
            u = self.decode(state, z)

            return u, mean, std
        
        def decode_softplus(self, state, z=None):
            if z is None:
                z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device).clamp(-0.5, 0.5)
            
            a = F.relu(self.d1(torch.cat([state, z], 1)))
            a = F.relu(self.d2(a))
            
        def decode(self, state, z=None):
            if z is None:
                    z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device).clamp(-0.5, 0.5)

            a = F.relu(self.d1(torch.cat([state, z], 1)))
            a = F.relu(self.d2(a))
            return self.max_action * torch.tanh(self.d3(a))
        
        def decode_bc(self, state, z=None):
            if z is None:
                    z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device)

            a = F.relu(self.d1(torch.cat([state, z], 1)))
            a = F.relu(self.d2(a))
            return self.max_action * torch.tanh(self.d3(a))

        def decode_bc_test(self, state, z=None):
            if z is None:
                    z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device).clamp(-0.25, 0.25)

            a = F.relu(self.d1(torch.cat([state, z], 1)))
            a = F.relu(self.d2(a))
            return self.max_action * torch.tanh(self.d3(a))
        
        def decode_multiple(self, state, z=None, num_decode=10):
            """Decode 10 samples atleast"""
            if z is None:
                z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), num_decode, self.latent_dim))).to(device).clamp(-0.5, 0.5)

            a = F.relu(self.d1(torch.cat([state.unsqueeze(0).repeat(num_decode, 1, 1).permute(1, 0, 2), z], 2)))
            a = F.relu(self.d2(a))
            return self.max_action * torch.tanh(self.d3(a)), self.d3(a)


class SAC_models:
    class Actor(nn.Module):
        def __init__(self, state_dim, action_dim, max_action):
            super(SAC_models.Actor, self).__init__()
            
            self.net = torch.nn.Sequential(
                torch.nn.Linear(state_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, action_dim),
            )
            
            self.action_dim = action_dim
            self.state_dim = state_dim
            self.log_std = torch.nn.Parameter(torch.zeros(action_dim))
            self.high = max_action
            self.low = -max_action
        		
        def forward(self, state):        
            mean = self.net(state)
            log_std = self.log_std.clamp(-20,2)
            std = torch.exp(log_std) 
            return mean, std
        
        def unsquash(self, values):
            normed_values = (values - self.low[0])/(self.high[0] - self.low[0])*2.0 - 1.0
            stable_normed_values = torch.clamp(normed_values, -1+1e-4, 1-1e-4)
            unsquashed = torch.atanh(stable_normed_values)
            return unsquashed.float()
        
        def sample_log(self, state, action):
            mean, std = self.forward(state)
            normal = torch.distributions.Normal(mean, std)
            x = self.unsquash(action)
            y = torch.tanh(x) 
            log_prob = torch.clamp(normal.log_prob(x), -5, 5)
            log_prob -= torch.log((1 - y.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            
            return log_prob
        
        def sample_SAC_continuous(self, state):
            mean, std = self.forward(state)
            normal = torch.distributions.Normal(mean, std)
            x = normal.rsample()
            y = torch.tanh(x)        
            action = y*self.high[0]
            log_prob = normal.log_prob(x)
            log_prob -= torch.log(self.high[0]*(1 - y.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            mean = torch.tanh(mean)*self.high[0]
            return action, log_prob, mean 
        
    class Actor_discrete(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(SAC_models.Actor_discrete, self).__init__()
            
            self.l1 = nn.Linear(state_dim, 128)
            nn.init.uniform_(self.l1.weight, -0.5, 0.5)
            self.l2 = nn.Linear(128,128)
            nn.init.uniform_(self.l2.weight, -0.5, 0.5)
            self.l3 = nn.Linear(128,action_dim)
            nn.init.uniform_(self.l3.weight, -0.5, 0.5)
            self.lS = nn.Softmax(dim=1)
            
        def forward(self, state):
            a = self.l1(state)
            a = F.relu(self.l2(a))
            return self.lS(torch.clamp(self.l3(a),-10,10))
        
        def sample(self, state):
            self.log_Soft = nn.LogSoftmax(dim=1)
            a = self.l1(state)
            a = F.relu(self.l2(a))
            log_prob = self.log_Soft(torch.clamp(self.l3(a),-10,10))
            
            prob = self.forward(state)
            m = Categorical(prob)
            action = m.sample()
            
            log_prob_sampled = log_prob[torch.arange(len(action)),action]
            
            return action, log_prob_sampled.reshape(-1,1)
        
        def sample_log(self, state, action):
            self.log_Soft = nn.LogSoftmax(dim=1)
            a = self.l1(state)
            a = F.relu(self.l2(a))
            log_prob = self.log_Soft(torch.clamp(self.l3(a),-10,10))
                        
            log_prob_sampled = log_prob[torch.arange(len(action)), action]
            
            return log_prob, log_prob_sampled.reshape(-1,1)
        
    class Critic_flat(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(SAC_models.Critic_flat, self).__init__()
    
            # Q1 architecture
            self.l1 = nn.Linear(state_dim + action_dim, 256)
            self.l2 = nn.Linear(256, 256)
            self.l3 = nn.Linear(256, 1)
    
            # Q2 architecture
            self.l4 = nn.Linear(state_dim + action_dim, 256)
            self.l5 = nn.Linear(256, 256)
            self.l6 = nn.Linear(256, 1)
    
        def forward(self, state, action):
            sa = torch.cat([state, action], 1)
            
            q1 = F.relu(self.l1(sa))
            q1 = F.relu(self.l2(q1))
            q1 = self.l3(q1)
            
            q2 = F.relu(self.l4(sa))
            q2 = F.relu(self.l5(q2))
            q2 = self.l6(q2)
            return q1, q2
    
        def Q1(self, state, action):
            sa = torch.cat([state, action], 1)
            
            q1 = F.relu(self.l1(sa))
            q1 = F.relu(self.l2(q1))
            q1 = self.l3(q1)
            return q1
        
    class Critic_flat_discrete(nn.Module):
        def __init__(self, state_dim, action_cardinality):
            super(SAC_models.Critic_flat_discrete, self).__init__()
    
            # Q1 architecture
            self.l1 = nn.Linear(state_dim, 256)
            self.l2 = nn.Linear(256, 256)
            self.l3 = nn.Linear(256, action_cardinality)
    
            # Q2 architecture
            self.l4 = nn.Linear(state_dim, 256)
            self.l5 = nn.Linear(256, 256)
            self.l6 = nn.Linear(256, action_cardinality)
    
        def forward(self, state):
            
            q1 = F.relu(self.l1(state))
            q1 = F.relu(self.l2(q1))
            q1 = self.l3(q1)
            
            q2 = F.relu(self.l4(state))
            q2 = F.relu(self.l5(q2))
            q2 = self.l6(q2)
            return q1, q2
    
        def Q1(self, state):
            
            q1 = F.relu(self.l1(state))
            q1 = F.relu(self.l2(q1))
            q1 = self.l3(q1)
            return q1

class PPO_off_models:
    class Actor(nn.Module):
        def __init__(self, state_dim, action_dim, max_action):
            super(PPO_off_models.Actor, self).__init__()
            
            self.net = torch.nn.Sequential(
                torch.nn.Linear(state_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, action_dim),
            )
            
            self.action_dim = action_dim
            self.state_dim = state_dim
            self.log_std = torch.nn.Parameter(torch.zeros(action_dim))
            self.high = max_action[0]
            self.low = -max_action[0]
            
            # Initialize parameters correctly
            self.apply(init_params)
        		
        def forward(self, state):        
            mean = self.net(state)
            log_std = self.log_std.clamp(-20,2)
            std = torch.exp(log_std) 
            return mean, std
        
        def squash(self, raw_values):
            squashed = ((torch.tanh(raw_values)+1)/2.0)*(self.high-self.low)+self.low
            for a in range(self.action_dim):
                squashed[:,a] = torch.clamp(squashed[:,a], self.low, self.high)
            return squashed.float()
        
        def unsquash(self, values):
            normed_values = (values - self.low)/(self.high - self.low)*2.0 - 1.0
            stable_normed_values = torch.clamp(normed_values, -1+1e-4, 1-1e-4)
            unsquashed = torch.atanh(stable_normed_values)
            return unsquashed.float()
        
        def sample(self, state):
            mean, std = self.forward(state)
            normal = torch.distributions.Normal(mean, std)
            x = normal.rsample()
            y = torch.tanh(x)        
            action = self.high*y
            log_prob = torch.clamp(normal.log_prob(x), -5, 5)
            log_prob -= torch.log((1 - y.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            return action, log_prob, mean 
        
        def sample_log(self, state, action):
            mean, std = self.forward(state)
            normal = torch.distributions.Normal(mean, std)
            x = self.unsquash(action)
            y = torch.tanh(x) 
            log_prob = torch.clamp(normal.log_prob(x), -5, 5)
            log_prob -= torch.log((1 - y.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            
            return log_prob
        
        def Distb(self, state):
            mean = self.net(state)
            log_std = self.log_std.clamp(-20,2)
            std = torch.exp(log_std)
            cov_mtx = torch.eye(self.action_dim).to(device) * (std ** 2)
            distb = torch.distributions.MultivariateNormal(mean, cov_mtx)

            return distb
        
    class Actor_discrete(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(SAC_models.Actor_discrete, self).__init__()
            
            self.l1 = nn.Linear(state_dim, 128)
            nn.init.uniform_(self.l1.weight, -0.5, 0.5)
            self.l2 = nn.Linear(128,128)
            nn.init.uniform_(self.l2.weight, -0.5, 0.5)
            self.l3 = nn.Linear(128,action_dim)
            nn.init.uniform_(self.l3.weight, -0.5, 0.5)
            self.lS = nn.Softmax(dim=1)
            
        def forward(self, state):
            a = self.l1(state)
            a = F.relu(self.l2(a))
            return self.lS(torch.clamp(self.l3(a),-10,10))
        
        def sample(self, state):
            self.log_Soft = nn.LogSoftmax(dim=1)
            a = self.l1(state)
            a = F.relu(self.l2(a))
            log_prob = self.log_Soft(torch.clamp(self.l3(a),-10,10))
            
            prob = self.forward(state)
            m = Categorical(prob)
            action = m.sample()
            
            log_prob_sampled = log_prob[torch.arange(len(action)),action]
            
            return action, log_prob_sampled.reshape(-1,1)
        
        def sample_log(self, state, action):
            self.log_Soft = nn.LogSoftmax(dim=1)
            a = self.l1(state)
            a = F.relu(self.l2(a))
            log_prob = self.log_Soft(torch.clamp(self.l3(a),-10,10))
                        
            log_prob_sampled = log_prob[torch.arange(len(action)), action]
            
            return log_prob, log_prob_sampled.reshape(-1,1)
        
    class Value_net(nn.Module):
        def __init__(self, state_dim):
            super(PPO_off_models.Value_net, self).__init__()
            # Value_net architecture
            self.l1 = nn.Linear(state_dim, 256)
            self.l2 = nn.Linear(256, 256)
            self.l3 = nn.Linear(256, 1)

        def forward(self, state):
            q1 = F.relu(self.l1(state))
            q1 = F.relu(self.l2(q1))
            q1 = self.l3(q1)    
            return q1









