#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 16:38:23 2022

@author: vittoriogiammarino
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# %%

RL_algorithms = ['AWAC_GAE', 'AWAC_GAE_entropy', 'AWAC', 'AWAC_Q_lambda_Haru', 'AWAC_Q_lambda_Haru_entropy',
                  'AWAC_Q_lambda_Peng', 'AWAC_Q_lambda_Peng_entropy', 'AWAC_TB_lambda', 'AWAC_TB_lambda_entropy', 
                  'BC', 'SAC', 'TD3_BC']

# RL_algorithms = ['AWAC_Q_lambda_Haru', 'AWAC_Q_lambda_Haru_entropy', 'TD3_BC']

# RL_algorithms = ['PPO', 'PPO_from_videos']

colors = {}

colors['AWAC_GAE'] = 'tab:blue'
colors['AWAC_GAE_entropy'] = 'tab:orange'
colors['AWAC'] = 'tab:pink'
colors['AWAC_Q_lambda_Haru'] = 'lime'
colors['AWAC_Q_lambda_Haru_entropy'] = 'tab:red'
colors['AWAC_Q_lambda_Peng'] = 'tab:purple'
colors['AWAC_Q_lambda_Peng_entropy'] = 'tab:brown'
colors['AWAC_TB_lambda'] = 'tab:green'
colors['AWAC_TB_lambda_entropy'] = 'tab:gray'
colors['BC'] = 'chocolate'
colors['SAC'] = 'tab:olive'
colors['TD3_BC'] = 'tab:cyan'
# colors['HPPO_3'] = 'lightcoral'
# colors['HTRPO_3'] = 'fuchsia'
# colors['HUATRPO_3'] = 'gold'
# colors['HGePPO_3'] = 'magenta'
# colors['HTD3_3'] = 'lightseagreen'
# colors['HSAC_3'] = 'peru'

environments = ['maze2d-large-v1', 'antmaze-large-diverse-v0', 'antmaze-large-play-v0', 'antmaze-medium-diverse-v0', 'antmaze-medium-play-v0',
                'halfcheetah-random-v2', 'halfcheetah-medium-v2', 'halfcheetah-expert-v2', 'walker2d-random-v2', 'walker2d-medium-v2', 
                'walker2d-expert-v2']

for env in environments:
    
    columns = 1
    rows = 1
    
    fig, ax = plt.subplots(rows, columns)
    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.2)
    fig.suptitle(env, fontsize="xx-large")
    
    for i in range(len(RL_algorithms)):
    
    # for k, ax_row in enumerate(ax):
    #     for j, axes in enumerate(ax_row):
            
        policy = RL_algorithms[i]
        
        RL = []
        
        for seed in range(10): 
            try:
                with open(f'results_partial/offline_RL/evaluation_offline_RL_{policy}_{env}_{seed}.npy', 'rb') as f:
                    RL.append(np.load(f, allow_pickle=True))  
            except:
                continue
                          
        try:
            mean = np.mean(np.array(RL), 0)
            steps = np.linspace(0, (len(mean)-1)*30000, len(mean))
            std = np.std(np.array(RL),0)
            ax.plot(steps, mean, label=policy, c=colors[policy])
            ax.fill_between(steps, mean-std, mean+std, alpha=0.2, facecolor=colors[policy])
        except:
            continue
                
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Reward')

# %%

# RL_algorithms = ['AWAC_GAE', 'AWAC_GAE_entropy', 'AWAC', 'AWAC_Q_lambda_Haru', 'AWAC_Q_lambda_Haru_entropy',
#                   'AWAC_Q_lambda_Peng', 'AWAC_Q_lambda_Peng_entropy', 'AWAC_TB_lambda', 'AWAC_TB_lambda_entropy', 
#                   'BC', 'SAC', 'TD3_BC']

RL_algorithms = ['AWAC_Q_lambda_Haru_entropy']

# RL_algorithms = ['AWAC_Q_lambda_Haru', 'AWAC_Q_lambda_Haru_entropy', 'TD3_BC']

# RL_algorithms = ['PPO', 'PPO_from_videos']

colors = {}

colors['AWAC_GAE'] = 'tab:blue'
colors['AWAC_GAE_entropy'] = 'tab:orange'
colors['AWAC'] = 'tab:pink'
colors['AWAC_Q_lambda_Haru'] = 'lime'
colors['AWAC_Q_lambda_Haru_entropy'] = 'tab:red'
colors['AWAC_Q_lambda_Peng'] = 'tab:purple'
colors['AWAC_Q_lambda_Peng_entropy'] = 'tab:brown'
colors['AWAC_TB_lambda'] = 'tab:green'
colors['AWAC_TB_lambda_entropy'] = 'tab:gray'
colors['BC'] = 'chocolate'
colors['SAC'] = 'tab:olive'
colors['TD3_BC'] = 'tab:cyan'
# colors['HPPO_3'] = 'lightcoral'
# colors['HTRPO_3'] = 'fuchsia'
# colors['HUATRPO_3'] = 'gold'
# colors['HGePPO_3'] = 'magenta'
# colors['HTD3_3'] = 'lightseagreen'
# colors['HSAC_3'] = 'peru'

environments = ['maze2d-large-v1', 'antmaze-large-diverse-v0', 'antmaze-large-play-v0', 'antmaze-medium-diverse-v0', 'antmaze-medium-play-v0']
                # 'halfcheetah-random-v2', 'halfcheetah-medium-v2', 'halfcheetah-expert-v2', 'walker2d-random-v2', 'walker2d-medium-v2', 
                # 'walker2d-expert-v2']

RL_tot = []
for env in environments:
    
    columns = 1
    rows = 1
    
    fig, ax = plt.subplots(rows, columns)
    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.2)
    fig.suptitle(env, fontsize="xx-large")
    
    
    for i in range(len(RL_algorithms)):
    
    # for k, ax_row in enumerate(ax):
    #     for j, axes in enumerate(ax_row):
            
        policy = RL_algorithms[i]
        
        RL = []
        
        for seed in range(10): 
            try:
                with open(f'results_partial/offline_RL/evaluation_offline_RL_{policy}_{env}_{seed}.npy', 'rb') as f:
                    RL.append(np.load(f, allow_pickle=True))  
            except:
                continue
                          
        try:
            mean = np.mean(np.array(RL), 0)
            steps = np.linspace(0, (len(mean)-1)*30000, len(mean))
            std = np.std(np.array(RL),0)
            
            RL_tot.append(mean)
            RL_tot.append(std)
            
            ax.plot(steps, mean, label=policy, c=colors[policy])
            ax.fill_between(steps, mean-std, mean+std, alpha=0.2, facecolor=colors[policy])
        except:
            continue
                
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Reward')

# %%

# RL_algorithms = ['AWAC_GAE', 'AWAC_GAE_entropy', 'AWAC', 'AWAC_Q_lambda_Haru', 'AWAC_Q_lambda_Haru_entropy',
#                   'AWAC_Q_lambda_Peng', 'AWAC_Q_lambda_Peng_entropy', 'AWAC_TB_lambda', 'AWAC_TB_lambda_entropy', 
#                   'BC', 'SAC', 'TD3_BC']

RL_algorithms = ['TD3_BC']

# RL_algorithms = ['AWAC_Q_lambda_Haru', 'AWAC_Q_lambda_Haru_entropy', 'TD3_BC']

# RL_algorithms = ['PPO', 'PPO_from_videos']

colors = {}

colors['AWAC_GAE'] = 'tab:blue'
colors['AWAC_GAE_entropy'] = 'tab:orange'
colors['AWAC'] = 'tab:pink'
colors['AWAC_Q_lambda_Haru'] = 'lime'
colors['AWAC_Q_lambda_Haru_entropy'] = 'tab:red'
colors['AWAC_Q_lambda_Peng'] = 'tab:purple'
colors['AWAC_Q_lambda_Peng_entropy'] = 'tab:brown'
colors['AWAC_TB_lambda'] = 'tab:green'
colors['AWAC_TB_lambda_entropy'] = 'tab:gray'
colors['BC'] = 'chocolate'
colors['SAC'] = 'tab:olive'
colors['TD3_BC'] = 'tab:cyan'
# colors['HPPO_3'] = 'lightcoral'
# colors['HTRPO_3'] = 'fuchsia'
# colors['HUATRPO_3'] = 'gold'
# colors['HGePPO_3'] = 'magenta'
# colors['HTD3_3'] = 'lightseagreen'
# colors['HSAC_3'] = 'peru'

environments = ['maze2d-large-v1', 'antmaze-large-diverse-v0', 'antmaze-large-play-v0', 'antmaze-medium-diverse-v0', 'antmaze-medium-play-v0']
                # 'halfcheetah-random-v2', 'halfcheetah-medium-v2', 'halfcheetah-expert-v2', 'walker2d-random-v2', 'walker2d-medium-v2', 
                # 'walker2d-expert-v2']

RL_tot_wallclock = []
for env in environments:
    
    columns = 1
    rows = 1
    
    fig, ax = plt.subplots(rows, columns)
    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.2)
    fig.suptitle(env, fontsize="xx-large")
    
    
    for i in range(len(RL_algorithms)):
    
    # for k, ax_row in enumerate(ax):
    #     for j, axes in enumerate(ax_row):
            
        policy = RL_algorithms[i]
        
        time = []
        
        for seed in range(10): 
            try:
                with open(f'results_partial/offline_RL/wallclock_time_offline_RL_{policy}_{env}_{seed}.npy', 'rb') as f:
                    time.append(np.load(f, allow_pickle=True))  
            except:
                continue
                          
        try:
            mean = np.mean(np.array(time)/3600, 0)
            std = np.std(np.array(time)/3600,0)
            
            RL_tot_wallclock.append(mean)
            RL_tot_wallclock.append(std)
            
        except:
            continue
                

# %%

for env in environments:
    
    columns = 1
    rows = 1
    
    fig, ax = plt.subplots(rows, columns)
    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.2)
    fig.suptitle(env, fontsize="xx-large")
    
    for i in range(len(RL_algorithms)):
    
    # for k, ax_row in enumerate(ax):
    #     for j, axes in enumerate(ax_row):
            
        policy = RL_algorithms[i]
        
        RL = []
        wallclock = []
        
        for seed in range(10):
            
            if policy == "TD3" or policy == "SAC":
                with open(f'results/RL/evaluation_RL_{policy}_{env}_{seed}.npy', 'rb') as f:
                    RL.append(np.load(f, allow_pickle=True))  
                    
                with open(f'results/RL/wallclock_time_RL_{policy}_{env}_{seed}.npy', 'rb') as f:
                    wallclock.append(np.load(f, allow_pickle=True))  
                    
            else:
                with open(f'results/RL/evaluation_RL_{policy}_Entropy_True_{env}_{seed}.npy', 'rb') as f:
                    RL.append(np.load(f, allow_pickle=True))     
                    
                with open(f'results/RL/wallclock_time_RL_{policy}_Entropy_True_{env}_{seed}.npy', 'rb') as f:
                    wallclock.append(np.load(f, allow_pickle=True))  
                
        mean = np.mean(np.array(RL), 0)
        steps = np.append(0, np.mean(np.array(wallclock), 0))
        std = np.std(np.array(RL),0)
        ax.plot(steps, mean, label=policy, c=colors[policy])
        ax.fill_between(steps, mean-std, mean+std, alpha=0.2, facecolor=colors[policy])
                
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3)
ax.set_xlabel('Time')
ax.set_ylabel('Reward')