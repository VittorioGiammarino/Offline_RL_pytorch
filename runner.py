#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 20:59:24 2022

@author: vittoriogiammarino
"""

import numpy as np
import time

from evaluation import eval_policy

class run_AWAC:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer, mean, std, args, seed):
        # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(0, seed, self.agent, args, mean, std)
        evaluation_RL.append(avg_reward) 
    
        it_num = 0
        start_time = time.time()

        for t in range(int(args.ntrajs*args.traj_size*args.max_iter)):
		
            self.agent.train(replay_buffer)
            it_num+=1

            # Evaluate episode
            if it_num % (args.eval_freq*args.ntrajs*args.traj_size) == 0:
                avg_reward = eval_policy(it_num, seed, self.agent, args, mean, std)
                evaluation_RL.append(avg_reward)    
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent
    
class run_AWAC_GAE:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer, mean, std, args, seed):
        # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(0, seed, self.agent, args, mean, std)
        evaluation_RL.append(avg_reward) 
    
        it_num = 0
        start_time = time.time()

        for t in range(int(args.max_iter)):
		
            states, actions, returns, advantage = self.agent.GAE(replay_buffer, args.ntrajs, args.traj_size)
            self.agent.train(states, actions, returns, advantage)
            it_num+=1

            # Evaluate episode
            if it_num % 1 == 0:
                avg_reward = eval_policy(it_num, seed, self.agent, args, mean, std)
                evaluation_RL.append(avg_reward)    
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent
    
class run_AWAC_Q_lambda_Peng:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer, mean, std, args, seed):
        # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(0, seed, self.agent, args, mean, std)
        evaluation_RL.append(avg_reward) 
    
        it_num = 0
        start_time = time.time()

        for t in range(int(args.max_iter)):
		
            states, actions, target_Q, advantage = self.agent.Q_Lambda_Peng(replay_buffer, args.ntrajs, args.traj_size)
            self.agent.train(states, actions, target_Q, advantage)
            it_num+=1

            # Evaluate episode
            if it_num % 1 == 0:
                avg_reward = eval_policy(it_num, seed, self.agent, args, mean, std)
                evaluation_RL.append(avg_reward)    
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent
    
class run_AWAC_Q_lambda_Haru:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer, mean, std, args, seed):
        # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(0, seed, self.agent, args, mean, std)
        evaluation_RL.append(avg_reward) 
    
        it_num = 0
        start_time = time.time()

        for t in range(int(args.max_iter)):
		
            states, actions, target_Q, advantage = self.agent.Q_Lambda_Haru(replay_buffer, args.ntrajs, args.traj_size)
            self.agent.train(states, actions, target_Q, advantage)
            it_num+=1

            # Evaluate episode
            if it_num % 1 == 0:
                avg_reward = eval_policy(it_num, seed, self.agent, args, mean, std)
                evaluation_RL.append(avg_reward)    
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent
    
class run_AWAC_TB_lambda:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer, mean, std, args, seed):
        # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(0, seed, self.agent, args, mean, std)
        evaluation_RL.append(avg_reward) 
    
        it_num = 0
        start_time = time.time()

        for t in range(int(args.max_iter)):
		
            states, actions, target_Q, advantage = self.agent.TB_Lambda(replay_buffer, args.ntrajs, args.traj_size)
            self.agent.train(states, actions, target_Q, advantage)
            it_num+=1

            # Evaluate episode
            if it_num % 1 == 0:
                avg_reward = eval_policy(it_num, seed, self.agent, args, mean, std)
                evaluation_RL.append(avg_reward)    
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent

class run_SAC:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer, mean, std, args, seed):
        # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(0, seed, self.agent, args, mean, std)
        evaluation_RL.append(avg_reward) 
    
        it_num = 0
        start_time = time.time()

        for t in range(int(args.ntrajs*args.traj_size*args.max_iter)):
		
            self.agent.train(replay_buffer)
            it_num+=1

            # Evaluate episode
            if it_num % (args.eval_freq*args.ntrajs*args.traj_size) == 0 :
                avg_reward = eval_policy(it_num, seed, self.agent, args, mean, std)
                evaluation_RL.append(avg_reward)    
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent
    
class run_BC:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer, mean, std, args, seed):
        # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(0, seed, self.agent, args, mean, std)
        evaluation_RL.append(avg_reward) 
    
        it_num = 0
        start_time = time.time()

        for t in range(int(args.ntrajs*args.traj_size*args.max_iter)):
		
            self.agent.train(replay_buffer)
            it_num+=1

            # Evaluate episode
            if it_num % (args.eval_freq*args.ntrajs*args.traj_size) == 0:
                avg_reward = eval_policy(it_num, seed, self.agent, args, mean, std)
                evaluation_RL.append(avg_reward)    
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent
    
class run_BCQ:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer, mean, std, args, seed):
        # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(0, seed, self.agent, args, mean, std)
        evaluation_RL.append(avg_reward) 
    
        it_num = 0
        start_time = time.time()
        
        for t in range(int(args.ntrajs*args.traj_size*args.max_iter)):
		
            self.agent.train(replay_buffer)
            it_num+=1

            # Evaluate episode
            if it_num % (args.eval_freq*args.ntrajs*args.traj_size) == 0:
                avg_reward = eval_policy(it_num, seed, self.agent, args, mean, std)
                evaluation_RL.append(avg_reward)    
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent
    
class run_BEAR:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer, mean, std, args, seed):
        # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(0, seed, self.agent, args, mean, std)
        evaluation_RL.append(avg_reward) 
    
        it_num = 0
        start_time = time.time()

        for t in range(int(args.ntrajs*args.traj_size*args.max_iter)):
		
            self.agent.train(replay_buffer)
            it_num+=1

            # Evaluate episode
            if it_num % (args.eval_freq*args.ntrajs*args.traj_size) == 0:
                avg_reward = eval_policy(it_num, seed, self.agent, args, mean, std)
                evaluation_RL.append(avg_reward)    
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent
    
class run_TD3_BC:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer, mean, std, args, seed):
        # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(0, seed, self.agent, args, mean, std)
        evaluation_RL.append(avg_reward) 
    
        it_num = 0
        start_time = time.time()

        for t in range(int(args.ntrajs*args.traj_size*args.max_iter)):
		
            self.agent.train(replay_buffer)
            it_num+=1

            # Evaluate episode
            if it_num % (args.eval_freq*args.ntrajs*args.traj_size) == 0:
                avg_reward = eval_policy(it_num, seed, self.agent, args, mean, std)
                evaluation_RL.append(avg_reward)    
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent
    
class run_PPO_off:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer, mean, std, args, seed):
        # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(0, seed, self.agent, args, mean, std)
        evaluation_RL.append(avg_reward) 
    
        it_num = 0
        start_time = time.time()

        for t in range(int(args.number_steps_per_iter*args.max_iter)):
            
            states, actions, returns, advantage = self.agent.Tree_backup(replay_buffer, args.ntrajs, args.traj_size)
            self.agent.train(states, actions, returns, advantage, Entropy = False)
            it_num+=1
            
            # Evaluate episode
            if it_num % 1 == 0:
                avg_reward = eval_policy(it_num, seed, self.agent, args, mean, std)
                evaluation_RL.append(avg_reward)   
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent
    
class run_PPO:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer, mean, std, args, seed):
        # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(0, seed, self.agent, args, mean, std)
        evaluation_RL.append(avg_reward) 
    
        it_num = 0
        start_time = time.time()

        for t in range(int(args.number_steps_per_iter*args.max_iter)):
		
            states, actions, returns, advantage = self.agent.GAE(replay_buffer, args.ntrajs, args.traj_size)
            self.agent.train(states, actions, returns, advantage, Entropy = True)
            it_num+=1

            # Evaluate episode
            if it_num % 1 == 0:
                avg_reward = eval_policy(it_num, seed, self.agent, args, mean, std)
                evaluation_RL.append(avg_reward)    
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent