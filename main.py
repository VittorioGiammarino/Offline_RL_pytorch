#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 17:00:31 2022

@author: vittoriogiammarino
"""

import gym
import numpy as np
import torch
import argparse
import os

import d4rl

import runner

from Buffers.Buffers import ReplayBuffer

from algorithms.SAC import SAC
from algorithms.AWAC import AWAC
from algorithms.AWAC_GAE import AWAC_GAE
from algorithms.AWAC_Q_lambda import AWAC_Q_lambda
from algorithms.BC import BC
from algorithms.BCQ import BCQ
from algorithms.BEAR import BEAR
from algorithms.TD3_BC import TD3_BC
from algorithms.PPO_off import PPO_off
from algorithms.PPO import PPO

def RL(env, args, seed):
    
    data_set = env.unwrapped.get_dataset()
    
    if args.env_reward_type=="Sparse":
        data_set["observations_with_goal"] = np.concatenate((data_set["observations"], data_set["infos/goal"]), axis=1)
    else:
        NotImplemented
            
    
    if args.action_space == 'Continuous':
        action_dim = env.action_space.shape[0] 
        action_space_cardinality = np.inf
        max_action = np.zeros((action_dim,))
        min_action = np.zeros((action_dim,))
        for a in range(action_dim):
            max_action[a] = env.action_space.high[a]   
            min_action[a] = env.action_space.low[a]  
            
    elif args.action_space == 'Discrete':
        
        try:
            action_dim = env.action_space.shape[0] 
        except:
            action_dim = 1

        action_space_cardinality = env.action_space.n
        max_action = np.nan
        min_action = np.nan
                
    if args.env_reward_type=="Sparse" and args.goal_in_observations:
        state_dim = data_set["observations_with_goal"].shape[1]
    else:
        state_dim = data_set["observations"].shape[1]
        
    replay_buffer = ReplayBuffer(state_dim, action_dim, len(data_set["observations"]))
    
    if args.env_reward_type=="Dense":
        replay_buffer.convert_D4RL_dense(data_set)
    elif args.env_reward_type=="Sparse":
        replay_buffer.convert_D4RL(data_set, args.goal_in_observations) 
    else:
        NotImplemented

    mean, std = replay_buffer.normalize_states()
    
    if args.policy == "SAC":
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "action_space_cardinality": action_space_cardinality,
         "max_action": max_action,
         "min_action": min_action
        }

        Agent_RL = SAC(**kwargs)
        
        run_sim = runner.run_SAC(Agent_RL)
        wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer, mean, std, args, seed)
        return wallclock_time, evaluation_RL, Agent_RL
    
    if args.policy == "AWAC":
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "action_space_cardinality": action_space_cardinality,
         "max_action": max_action,
         "min_action": min_action,
         "Entropy": args.Entropy
        }

        Agent_RL = AWAC(**kwargs)
        
        run_sim = runner.run_AWAC(Agent_RL)
        wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer, mean, std, args, seed)
        return wallclock_time, evaluation_RL, Agent_RL
    
    if args.policy == "AWAC_GAE":
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "action_space_cardinality": action_space_cardinality,
         "max_action": max_action,
         "min_action": min_action,
         "Entropy": args.Entropy
        }

        Agent_RL = AWAC_GAE(**kwargs)
        
        run_sim = runner.run_AWAC_GAE(Agent_RL)
        wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer, mean, std, args, seed)
        return wallclock_time, evaluation_RL, Agent_RL
    
    if args.policy == "AWAC_Q_lambda_Peng":
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "action_space_cardinality": action_space_cardinality,
         "max_action": max_action,
         "min_action": min_action,
         "Entropy": args.Entropy
        }

        Agent_RL = AWAC_Q_lambda(**kwargs)
        
        run_sim = runner.run_AWAC_Q_lambda_Peng(Agent_RL)
        wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer, mean, std, args, seed)
        return wallclock_time, evaluation_RL, Agent_RL
    
    if args.policy == "AWAC_Q_lambda_Haru":
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "action_space_cardinality": action_space_cardinality,
         "max_action": max_action,
         "min_action": min_action,
         "Entropy": args.Entropy
        }

        Agent_RL = AWAC_Q_lambda(**kwargs)
        
        run_sim = runner.run_AWAC_Q_lambda_Haru(Agent_RL)
        wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer, mean, std, args, seed)
        return wallclock_time, evaluation_RL, Agent_RL
    
    if args.policy == "AWAC_TB_lambda":
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "action_space_cardinality": action_space_cardinality,
         "max_action": max_action,
         "min_action": min_action,
         "Entropy": args.Entropy
        }

        Agent_RL = AWAC_Q_lambda(**kwargs)
        
        run_sim = runner.run_AWAC_TB_lambda(Agent_RL)
        wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer, mean, std, args, seed)
        return wallclock_time, evaluation_RL, Agent_RL
    
    if args.policy == "BC":
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "max_action": max_action,
         "min_action": min_action
        }

        Agent_RL = BC(**kwargs)
        
        run_sim = runner.run_BC(Agent_RL)
        wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer, mean, std, args, seed)
        
        return wallclock_time, evaluation_RL, Agent_RL
    
    if args.policy == "BEAR":
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "max_action": max_action
        }

        Agent_RL = BEAR(**kwargs)
        
        run_sim = runner.run_BEAR(Agent_RL)
        wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer, mean, std, args, seed)
        
        return wallclock_time, evaluation_RL, Agent_RL

    if args.policy == "TD3_BC":
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "max_action": max_action
        }

        Agent_RL = TD3_BC(**kwargs)
        
        run_sim = runner.run_TD3_BC(Agent_RL)
        wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer, mean, std, args, seed)
        
        return wallclock_time, evaluation_RL, Agent_RL
    
    if args.policy == "BCQ":
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "max_action": max_action
        }

        Agent_RL = BCQ(**kwargs)
        
        run_sim = runner.run_BCQ(Agent_RL)
        wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer, mean, std, args, seed)
        
        return wallclock_time, evaluation_RL, Agent_RL
    
    if args.policy == "PPO_off":
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "action_space_cardinality": action_space_cardinality,
         "max_action": max_action,
         "min_action": min_action
        }

        Agent_RL = PPO_off(**kwargs)
        
        run_sim = runner.run_PPO_off(Agent_RL)
        wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer, mean, std, args, seed)
        
        return wallclock_time, evaluation_RL, Agent_RL
    
    if args.policy == "PPO":
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "action_space_cardinality": action_space_cardinality,
         "max_action": max_action,
         "min_action": min_action
        }

        Agent_RL = PPO(**kwargs)
        
        run_sim = runner.run_PPO(Agent_RL)
        wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, replay_buffer, mean, std, args, seed)
        
        return wallclock_time, evaluation_RL, Agent_RL
        

def train(args, seed): 
    
    env = gym.make(args.env)
    
    try:
        if env.action_space.n>0:
            args.action_space = "Discrete"
            print("Environment supports Discrete action space.")
    except:
        args.action_space = "Continuous"
        print("Environment supports Continuous action space.")
            
    # Set seeds
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
        
    wallclock_time, evaluations, policy = RL(env, args, seed)
    
    return wallclock_time, evaluations, policy

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    #General
    parser.add_argument("--mode", default="offline_RL", help='supported modes are HVI, HRL and RL (default = "HVI")')     
    parser.add_argument("--env", default="halfcheetah-expert-v2")               # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--env_reward_type", default="Dense", help = "Dense or Sparse")               # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--action_space", default="Continuous")               # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--goal_in_observations", action = "store_false")     # Sets Gym, PyTorch and Numpy seeds
    
    parser.add_argument("--policy", default="AWAC_Q_lambda_Peng") # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--Entropy", action="store_false") # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--seed", default=0, type=int)    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--number_steps_per_iter", default=110, type=int) # Time steps initial random policy is used 25e3
    parser.add_argument("--eval_freq", default=1, type=int)          # How often (time steps) we evaluate
    parser.add_argument("--max_iter", default=20, type=int)    # Max time steps to run environment
    # HRL
    parser.add_argument("--start_timesteps", default=25e3, type=int) # Time steps before training default=25e3
    parser.add_argument("--expl_noise", default=0.1)                 # Std of Gaussian exploration noise    
    parser.add_argument("--save_model", action="store_false")         # Save model and optimizer parameters
    parser.add_argument("--load_model", default=True, type=bool)               # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--load_model_path", default="") 
    # PPO_off
    parser.add_argument("--ntrajs", default=30, type=int) #maze 300
    parser.add_argument("--traj_size", default=1000, type=int) # traj_size 100
    # Evaluation
    parser.add_argument("--evaluation_episodes", default=10, type=int)
    parser.add_argument("--evaluation_max_n_steps", default = 2000, type=int)
    # Experiments
    parser.add_argument("--detect_gradient_anomaly", action="store_true")
    args = parser.parse_args()
    
    torch.autograd.set_detect_anomaly(args.detect_gradient_anomaly)
    
    if args.mode == "offline_RL":
        
        if args.Entropy:
            file_name = f"{args.mode}_{args.policy}_entropy_{args.env}_{args.seed}"
            print("---------------------------------------")
            print(f"Mode: {args.mode}, Policy: {args.policy}, Entropy: {args.Entropy}, Env: {args.env}, Seed: {args.seed}")
            print("---------------------------------------")
            
        else:
            file_name = f"{args.mode}_{args.policy}_{args.env}_{args.seed}"
            print("---------------------------------------")
            print(f"Mode: {args.mode}, Policy: {args.policy}, Entropy: {args.Entropy}, Env: {args.env}, Seed: {args.seed}")
            print("---------------------------------------")
        
        if not os.path.exists(f"./results/{args.mode}"):
            os.makedirs(f"./results/{args.mode}")
            
        if not os.path.exists(f"./models/{args.mode}/{file_name}"):
            os.makedirs(f"./models/{args.mode}/{file_name}")
        
        
        wallclock_time, evaluations, policy = train(args, args.seed)
        
        if args.save_model: 
            np.save(f"./results/{args.mode}/evaluation_{file_name}", evaluations)
            np.save(f"./results/{args.mode}/wallclock_time_{file_name}", wallclock_time)
            # policy.save_actor(f"./models/{args.mode}/{file_name}/{file_name}")
            # policy.save_critic(f"./models/{args.mode}/{file_name}/{file_name}")