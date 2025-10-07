import atexit
import copy
import time
import os
from collections import deque
import numpy as np
# from stable_baselines3.common.vec_env import VecEnvWrapper
import torch
import torch.nn as nn
import jpype
from jpype.imports import registerDomain
from jpype.types import JArray



class Selfplay:
    def __init__(self, num_selfplay_envs, main_agent,
                 mapPath="maps/16x16/basesWorkers16x16.xml", device='cuda'):

        self.num_selfplay_envs = num_selfplay_envs
        self.mapPath = mapPath
        self.device = device
        self.agent0 = main_agent
        self.agent1 = copy.deepcopy(main_agent)
        self.agent1.eval()


    def get_actions(self, x, Sc, z, action=None, invalid_action_masks=None, envs=None):
        
        # TODO: benutze immer denselben Agent aber wechsle Gewichte?
        action0, logprob0, entropy0, invalid_action_mask0 = self.agent0.get_action(x, Sc, z, action, invalid_action_masks, envs)
        action1, logprob1, entropy1, invalid_action_mask1 = self.agent1.get_action(x, Sc, z, action, invalid_action_masks, envs)
        
        # kombiniere die actions, indem immer abwechselnd eine action von agent0, dann von agent1 im Array liegt. (das für alle environments)
        # vorher: hat ppo_BC_PPO selfplay als 2 environments dargestellt
        # beide actions werden gleichzeitig ausgeführt
        # action = [agent0_action_env0, agent1_action_env0, agent0_action_env1, agent1_action_env1, ...]
        new_action = torch.zeros((self.num_selfplay_envs, action0.shape[1], action0.shape[2]), dtype=torch.int64).to(self.device)
        new_logprob = torch.zeros((self.num_selfplay_envs), dtype=torch.float32).to(self.device)
        new_entropy = torch.zeros((self.num_selfplay_envs), dtype=torch.float32).to(self.device)
        new_invalid_action_mask = torch.zeros((self.num_selfplay_envs, invalid_action_mask0.shape[1], invalid_action_mask0.shape[2]), dtype=torch.float32).to(self.device)

        for i in range(self.num_selfplay_envs):
            if i % 2 == 0:
                new_action[i] = action0[i//2]
                new_logprob[i] = logprob0[i//2]
                new_entropy[i] = entropy0[i//2]
                new_invalid_action_mask[i] = invalid_action_mask0[i//2]
            else:
                new_action[i] = action1[i//2]
                new_logprob[i] = logprob1[i//2]
                new_entropy[i] = entropy1[i//2]
                new_invalid_action_mask[i] = invalid_action_mask1[i//2]

        ### debugging
        # print("Actions shape: ", new_action.shape)
        # print("Logprobs shape: ", new_logprob.shape)
        # print("Entropies shape: ", new_entropy.shape)
        # print("Invalid action masks shape: ", new_invalid_action_mask.shape)
        
        return new_action, new_logprob, new_entropy, new_invalid_action_mask

# TODO: Exploiters trainieren, ohne dass ppo_BC_PPO etwas davon mitbekommt (außer, dass die Ausgaben etwas anders sind.)
# ppo_BC_PPO soll an allen Ausgaben, die übergeben werden trainieren dürfen!!!!!


