#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 17:19:32 2022

@author: vittoriogiammarino
"""
import numpy as np
import gym

# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(it_num, seed, policy, args, mean=0, std=1, seed_offset=100):
    eval_env = gym.make(args.env)
    eval_env.seed(seed + seed_offset)

    avg_reward = 0.
    for _ in range(args.evaluation_episodes):
        state, done = eval_env.reset(), False
        
        if args.env_reward_type == "Sparse":
            try:
                goal = eval_env.unwrapped.get_target()
            except:
                goal = eval_env.env.wrapped_env._goal
            state = np.concatenate((state, [goal[0], goal[1]]))
        state = (state.reshape(1,-1) - mean)/std
        
        for _ in range(args.evaluation_max_n_steps):

            action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action)
            if args.env_reward_type == "Sparse":
                try:
                    goal = eval_env.unwrapped.get_target()
                except:
                    goal = eval_env.env.wrapped_env._goal
                state = np.concatenate((state, [goal[0], goal[1]]))
            state = (state.reshape(1,-1) - mean)/std
            avg_reward += reward
            
            if done:
                break

    avg_reward /= args.evaluation_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

    print("---------------------------------------")
    print(f"Iteration: {it_num} ,Evaluation over {args.evaluation_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
    print("---------------------------------------")
    return avg_reward