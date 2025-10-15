from jpype.types import JArray, JInt
from torch.utils.data import DataLoader
from time import process_time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from sympy.codegen import Print
from torch.distributions import Categorical, kl_divergence
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import argparse
from torch.amp import autocast, GradScaler
from distutils.util import strtobool
import numpy as np
import gc
import zstandard as zstd
import io
import gym
import gym_microrts
# from gym_microrts.envs.vec_env2 import MicroRTSGridModeVecEnv
from gym_microrts.envs.microrts_vec_env import MicroRTSGridModeVecEnv
from gym_microrts.envs.microrts_bot_vec_env import MicroRTSBotGridVecEnv

from gym_microrts import microrts_ai
from gym.wrappers import TimeLimit, Monitor
from typing import Any, Dict, List, Optional, TypeVar
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import json
import random
import psutil
import os
from stable_baselines3.common.vec_env import VecEnvWrapper, VecVideoRecorder
from microrts_space_transform import MicroRTSSpaceTransform
from microrts_space_transformbots import MicroRTSSpaceTransformbot



import jpype



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="MicrortsBC",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-5,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=10000000000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument('--n-minibatch', type=int, default=4,
                        help='the number of mini batch')
    parser.add_argument('--num-bot-envs', type=int, default=1,
                        help='the number of bot game environment; 16 bot envs means 16 games')
    parser.add_argument('--num-selfplay-envs', type=int, default=4,
                        help='the number of self play envs; 16 self play envs means 8 games')
    parser.add_argument('--num-steps', type=int, default=512,
                        help='the number of steps per game environment')
    parser.add_argument('--gamma', type=float, default=1,
                        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='the lambda for the general advantage estimation')
    parser.add_argument('--ent-coef', type=float, default=0.005,
                        help="coefficient of the entropy")
    parser.add_argument('--vf-coef', type=float, default=0.01,
                        help="coefficient of the value function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--clip-coef', type=float, default=0.1,
                        help="the surrogate clipping coefficient")
    parser.add_argument('--update-epochs', type=int, default=4,
                         help="the K epochs to update the policy")
    parser.add_argument('--kle-stop', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                         help='If toggled, the policy updates will be early stopped w.r.t target-kl')
    parser.add_argument('--kle-rollback', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                         help='If toggled, the policy updates will roll back to previous policy if KL exceeds target-kl')
    parser.add_argument('--target-kl', type=float, default=0.03,
                         help='the target-kl variable that is referred by --kl')
    parser.add_argument('--kl_coeff', type=float, default=0.4,
                         help='')
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help="Toggles advantages normalization")
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')
    # new
    parser.add_argument('--BCtraining', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='If toggled, run behavior cloning training')
    parser.add_argument('--newdata', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='If toggled, collect new expert data for BC training')
    parser.add_argument('--nurwins', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='If toggled, only use winning episodes for BC training')
    parser.add_argument('--render', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='If toggled, render the environment during training')
    parser.add_argument('--render_all', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='If toggled, render ALL underlying environments (inefficient, for debugging)')
    parser.add_argument('--dbg', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='If toggled, run the script in debug mode (JPype in debug mode, wait for debugger to attach to port 5005)')
    parser.add_argument('--resume', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='If toggled, resume training from the latest checkpoint')

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

# TODO (selfplay): setze 
# export MICRORTS_JDWP_PORT=5005
# export MICRORTS_JDWP_SUSPEND=y   # oder 'n' wenn nicht suspendiert werden soll
# für Debugging, schließe die Fehlermeldung erst nach ein paar Sekunden mit "Debug anyway"



args.num_envs = args.num_selfplay_envs + args.num_bot_envs
args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.n_minibatch)




class VecstatsMonitor(VecEnvWrapper):
    def __init__(self, venv, gamma=None):
        super().__init__(venv)
        self.eprets = None
        self.eplens = None
        self.epcount = 0
        self.tstart = time.time()
        self.gamma = gamma
        self.raw_rewards = None

    def reset(self):
        obs = self.venv.reset()
        n = self.num_envs
        self.eprets = np.zeros(n, dtype=float)
        self.eplens = np.zeros(n, dtype=int)
        self.raw_rewards = [[] for _ in range(n)]
        self.tstart = time.time()
        return obs

    def step_wait(self):
        ### Debugging (scorerews always == 0)
        # if '_last_scorerews' not in globals():
        #     global _last_scorerews
        #     global scorerews
        #     _last_scorerews = None
        #     scorerews = None




        obs, denserews, attackrews, winlossrews, scorerews, dones, infos, res = self.venv.step_wait()
        # obs			//Observation per Enviorment [[Observation][Observation]...] Observation == []
        # nach envsT._from_microrts_obs(obs) # obs zu Tensor mit shape (24, 16, 16, 73) (von (24, X))
        # for map of size h x w, e environment: 
        # np.shape (e, h, w, n_f) n_f = 73
        # Environment Specification (Observation Space)
        
        

        # denserews         //weighted 6 Rewards per Environment ({'ResourceGatherRewardFunction': 0.0, 'ProduceWorkerRewardFunction': 0.0, 'ProduceBuildingRewardFunction': 1.0, 
        # 'ProduceLightUnitRewardFunction': 0.0, 'ProduceRangedUnitRewardFunction': 0.0, 'ScoreRewardFunction': 0.0})
        # print("denserews:", denserews)

        # attackrews        //weighted Attack Reward per Environment (AttackRewardFunction) (1 per attack) (Later in PPO: * 0.05)
        # # print("attackrews: ", attackrews)

        # winlossrews        //weighted Win/Loss Reward per Environment (RAIWinLossRewardFunction) (Win -> 1, Loss -> -1) (Later in PPO: * 10)
        # # print("winlossrews: ", winlossrews)

        # scorerews        //weighted Score Reward per Environment (ScoreRewardFunction)
        # ScoreRewardFunction (Score = own unit cost - enemy unit cost) (Later in PPO: * 0.2)
        # # print("scorerews: ", scorerews)
        # print("scorerews.shape: ", np.shape(scorerews))  # (24,)

        # dones			//Done Flags per Environment (Done == True --> Game Terminated in Step)
        # print("dones: " + dones)

        # res             //Resources per Enviorment [[Player0res, Player1res],[Player0res, Player1res]...]
        # print("res.shape: ", np.shape(res))  # (24, 2)

        # infos			//(Results of 9 Reward-functions per Enviorment) bevore applying @ reward_weight (== np.array([0.0, 1.0, 0.0, 0.0, 0.0, 5.0]))
        # Infos: {'raw_rewards': array([0., 0., 0., 1., 0., 0., 0., 0., 0.])} pro Environment
        # ==
        # {'RAIWinLossRewardFunction': 0.0, 'ResourceGatherRewardFunction': 0.0, 'ProduceWorkerRewardFunction': 0.0, 'ProduceBuildingRewardFunction': 1.0, 
        # 'AttackRewardFunction': 0.0, 'ProduceLightUnitRewardFunction': 0.0, 'ProduceRangedUnitRewardFunction': 0.0, 'ProduceHeavyUnitRewardFunction': 0.0, 'ScoreRewardFunction': 0.0}
        # print("Infos.shape: " + np.shape(infos))  # (24,)
        # reward_weight = np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0, 5.25, 6.0, 0])
        
        ### Debugging (scorerews always == 0)
        # Breakpoint if values change
        # if not np.array_equal(scorerews, _last_scorerews):
        #     breakpoint()
        # _last_scorerews = np.copy(scorerews)
    
        


        self.eprets += denserews + winlossrews + scorerews + attackrews
        self.eplens += 1

        for i, info in enumerate(infos):
            if 'raw_rewards' in info:
                self.raw_rewards[i].append(info['raw_rewards'])

        newinfos = list(infos)

        for i, done in enumerate(dones):
            if done:
                info = infos[i].copy()
                ep_ret = float(self.eprets[i])
                ep_len = int(self.eplens[i])
                ep_time = round(time.time() - self.tstart, 6)
                info['episode'] = {'r': ep_ret, 'l': ep_len, 't': ep_time}

                self.epcount += 1

                if self.raw_rewards[i]:
                    agg = np.sum(np.array(self.raw_rewards[i]), axis=0)
                    raw_names = [str(rf) for rf in self.rfs]
                    info['microrts_stats'] = dict(zip(raw_names, agg.tolist()))
                else:
                    info['microrts_stats'] = {}

                if winlossrews[i] == 0:
                    info['microrts_stats']['draw'] = True
                else:
                    info['microrts_stats']['draw'] = False

                self.eprets[i] = 0.0
                self.eplens[i] = 0
                self.raw_rewards[i] = []
                newinfos[i] = info



        return obs, denserews, attackrews, winlossrews, scorerews, dones, newinfos, res

    def step(self, actions):
        self.venv.step_async(actions) 
        # MicroRTSGridModeVecEnv.step_async (durch Delegation mit __getattribute__ in MicroRTSSpaceTransform aufgerufen)
        # Asynchron-Vorbereiten (Java-Array int[][][] gebaut)
        return self.step_wait()



# =========================
# Logging Setup (TensorBoard & WandB)
# =========================
experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
    '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))

RUN_ID_PATH = f"models/{experiment_name}/wandb_run_id.txt"
if args.prod_mode:
    import wandb
    if os.path.exists(RUN_ID_PATH):
        # Resume: read the previous run ID
        with open(RUN_ID_PATH, "r") as f:
            run_id = f.read().strip()
        resume_mode = "must"
    else:
        # First time: no file exists
        run_id = None
        resume_mode = "allow"
    run = wandb.init(
        project=args.wandb_project_name, entity=args.wandb_entity,
        # sync_tensorboard=True,
        config=vars(args), name=experiment_name, monitor_gym=True, resume=resume_mode, id=run_id, save_code=False)

    if resume_mode == "allow":
        os.makedirs(os.path.dirname(RUN_ID_PATH), exist_ok=True)
        with open(RUN_ID_PATH, "w") as f:
            f.write(run.id)
    wandb.tensorboard.patch(save=False)
    writer = SummaryWriter(f"/tmp/{experiment_name}")
    CHECKPOINT_FREQUENCY = 10
# =========================

# device setup
device = torch.device('cuda' if torch.cuda.is_available()
                      and args.cuda else 'cpu')
print(device)


# =========================
# creating Environments
# =========================
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

# TODO (selfplay), TODO (League training): remove opponents for testing (in running_BC.py steht das was in Basis Thesis verwendet wurde)
opponents = [microrts_ai.coacAI for _ in range(args.num_bot_envs)]

# opponents = [
#         microrts_ai.coacAI for _ in range(3)] + [
#             microrts_ai.mayari for _ in range(4)] + [
#                 microrts_ai.mixedBot for _ in range(4)] + [
#                     microrts_ai.izanagi for _ in range(3)] + [
#                         microrts_ai.droplet for _ in range(4)] + [
#                             microrts_ai.tiamat for _ in range(3)] + [
#                                 microrts_ai.workerRushAI for _ in range(3)]


envs = MicroRTSGridModeVecEnv(
    num_selfplay_envs=args.num_selfplay_envs,
    num_bot_envs=args.num_bot_envs,
    max_steps=2000,
    always_player_1=False,
    render_theme=1,
    ai2s=opponents,
    map_paths=["maps/16x16/basesWorkers16x16.xml"],
    reward_weight=np.array(
        [
            1.0,
            1.0,
            1.0,
            0.2,
            1.0,
            4.0,
            5.25,
            6.0,
            0]))
envsT = MicroRTSSpaceTransform(envs)
# print(envsT.__class__.mro())
# print(hasattr(envsT, "step_async"))
# print(envsT.step_async.__qualname__)
# print(envsT.step_wait.__qualname__)

envsT = VecstatsMonitor(envsT, args.gamma)

# Helper to render all underlying environments (inefficient, for debugging)
def render_all_envs(env_transform):
    """Try to render every underlying env. This is intentionally
    inefficient and for debugging only. It attempts multiple strategies
    depending on the wrapped object (vec client, interface, or sub-clients).
    """
    # Try direct vector render first
    try:
        if hasattr(env_transform, 'interface') and hasattr(env_transform.interface, 'vec_client'):
            vec_client = env_transform.interface.vec_client
            # try selfPlayClients first
            if hasattr(vec_client, 'selfPlayClients') and len(vec_client.selfPlayClients) > 0:
                for c in vec_client.selfPlayClients:
                    try:
                        c.render(False)
                    except Exception:
                        pass
            # try regular clients
            if hasattr(vec_client, 'clients') and len(vec_client.clients) > 0:
                for c in vec_client.clients:
                    try:
                        c.render(False)
                    except Exception:
                        pass
            return
    except Exception:
        pass

    # Fallback: call env_transform.render() which may render a single view
    try:
        env_transform.render()
    except Exception:
        pass


# Video?
if args.capture_video:
    envs = VecVideoRecorder(
        envs,
        f'videos/{experiment_name}',
        record_video_trigger=lambda x: x %
        1000000 == 0,
        video_length=2000)
    




from enum import IntEnum

# Selfplay Enum for different Agents
class SelfplayAgentType(IntEnum):
    CUR_MAIN = 0    # used, when the current main agent plays against a bot (not for non-selfplaying envs)
    OLD_MAIN = 1
    MAIN_EXPLOITER = 2
    LEAGUE_EXPLOITER = 3

    @staticmethod
    def not_implemented(name):
        raise NotImplementedError(f"{name} get_action wasn't found in get_action_calls")

    @staticmethod
    def get_agents_action(split_x, split_Sc, split_z, split_action, split_invalid_action_masks, split_env_Indices, base_env):
        # TODO (League training): fill in the get_action calls
        # TODO (League training): rufe die get_action Methode des jeweiligen Agenten auf (dafür braucht man den richtigen Agenten)
        get_action_calls = {
            SelfplayAgentType.CUR_MAIN: lambda *args: SelfplayAgentType.not_implemented("CUR_MAIN"),  # already handled in selfplay_get_action
            SelfplayAgentType.OLD_MAIN: lambda x, Sc, z, action, invalid_action_masks, env: SelfplayAgentType.not_implemented("OLD_MAIN"),
            SelfplayAgentType.MAIN_EXPLOITER: lambda x, Sc, z, action, invalid_action_masks, env: SelfplayAgentType.not_implemented("MAIN_EXPLOITER"),
            SelfplayAgentType.LEAGUE_EXPLOITER: lambda x, Sc, z, action, invalid_action_masks, env: SelfplayAgentType.not_implemented("LEAGUE_EXPLOITER")
        }

        num_non_cur_main_selfplay_envs = sum(1 for t in split_env_Indices if t != SelfplayAgentType.CUR_MAIN and split_env_Indices[t].numel() > 0)
        num_predicted_parameters = len(base_env.action_plane_space.nvec)
        shape = (num_non_cur_main_selfplay_envs, 256, num_predicted_parameters)

        split_action = torch.empty(shape).to(device)
        split_logproba = torch.empty(shape).to(device)
        split_entropy = torch.empty(shape).to(device)
        split_invalid_action_masks = torch.empty(num_non_cur_main_selfplay_envs, 256, sum(base_env.action_plane_space.nvec.tolist()) + 1).to(device)
        for t in SelfplayAgentType:
            if (t == SelfplayAgentType.CUR_MAIN):
                continue
            
            sub_env = SubEnvView(base_env, split_env_Indices[t])
            if t in split_env_Indices and len(split_env_Indices[t]) > 0:
                if t in get_action_calls:
                    get_action_call = get_action_calls[t](split_x[t], split_Sc[t], split_z[t], split_action[t], split_invalid_action_masks[t], sub_env)
                    if len(split_env_Indices[t]) == 1:
                        split_action[t], split_logproba[t], split_entropy[t], split_invalid_action_masks[t] = \
                            get_action_call
                    else:
                        for i in range(split_env_Indices[t]):
                            split_action[t][i], split_logproba[t][i], split_entropy[t][i], split_invalid_action_masks[t][i] = \
                                get_action_call[i]
                else:
                    SelfplayAgentType.not_implemented(t)

        # TODO (League training): Sum für split_logproba, split_entropy testen
        return split_action, split_logproba.sum(1).sum(1), split_entropy.sum(1).sum(1), split_invalid_action_masks


# TODO (League training): get the rigth Agent representation for each agent_type per Environment (don't fill non selfplaying envs) Fokus auf die agenten gibt, die gegen main gewinnen (falls --PFSP), schalte League training an mit (--leaguetraining), gegen nur cur main und alte main: (--FSP), gegen nur cur Main: wenn alles False
# agent_type = torch.tensor(
#     [SelfplayAgentType.main for _ in range()],
#     dtype=torch.long).to(device)
# agent_type = torch.tensor([
#     SelfplayAgentType.OLD_MAIN,
#     SelfplayAgentType.LEAGUE_EXPLOITER,
#     SelfplayAgentType.MAIN_EXPLOITER,
#     SelfplayAgentType.OLD_MAIN,
# ], dtype=torch.long)
# agent_type = None
agent_type = torch.tensor([
    SelfplayAgentType.CUR_MAIN,
    SelfplayAgentType.CUR_MAIN,
    SelfplayAgentType.CUR_MAIN,
    SelfplayAgentType.CUR_MAIN,
], dtype=torch.long)
assert args.num_selfplay_envs == len(agent_type), "Number of selfplay envs must be equal to the number of agent types (each agent plays against itself)"

# =========================


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(
                CategoricalMasked,
                self).__init__(
                probs,
                logits,
                validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits,
                                 torch.tensor(-1e+8).to(device))
            super(
                CategoricalMasked,
                self).__init__(
                probs,
                logits,
                validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(device))
        return -p_log_p.sum(-1)


# get scalar feature representation from observation (globale
# Zustandsinformationen in das Modell hinzufügen) for ScalarEncoder
def getScalarFeatures(obs, res, numenvs):
    ScFeatures = torch.zeros(numenvs, 11)

    for i in range(numenvs):

        res_plane = (obs[i, :, :, 1] * obs[i, :, :, 7])
        lightunit_plane = (obs[i, :, :, 11])
        heavyunit_plane = (obs[0, :, :, 12])
        rangedunit_plane = (obs[0, :, :, 13])
        total_res = res_plane.sum().item()

        worker_plane = obs[i, :, :, 10]
        building_plane = obs[i, :, :, 9]
        player0_plane = obs[i, :, :, 4]
        player1_plane = obs[i, :, :, 5]

        ScFeatures[i, 0] = res[i][0]  # Player0 res
        ScFeatures[i, 1] = res[i][1]  # Player1 res
        ScFeatures[i, 2] = total_res  # vorhandene res
        ScFeatures[i, 3] = (worker_plane *
                            player0_plane).sum().item()  # Player0 worker
        ScFeatures[i, 4] = (lightunit_plane *
                            player0_plane).sum().item()  # Player0 light
        ScFeatures[i, 5] = (heavyunit_plane *
                            player0_plane).sum().item()  # Player0 heavy
        ScFeatures[i, 6] = (rangedunit_plane *
                            player0_plane).sum().item()  # Player0 ranged
        ScFeatures[i, 7] = (worker_plane *
                            player1_plane).sum().item()  # Player1 worker
        ScFeatures[i, 8] = (lightunit_plane *
                            player1_plane).sum().item()  # Player1 light
        ScFeatures[i, 9] = (heavyunit_plane *
                            player1_plane).sum().item()  # Player1 heavy
        ScFeatures[i, 10] = (rangedunit_plane *
                             player1_plane).sum().item()  # Player1 ranged

    # Time step in the game
    return ScFeatures

# zEncoder


class ZSampler(nn.Module):
    def __init__(self, obs_dim, z_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, z_dim)
        )

    def forward(self, obs):
        return self.encoder(obs)

# ScalarEncoder


class ScalarFeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=32, num_layers=2):
        super(ScalarFeatureEncoder, self).__init__()

        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ResBlock for CNN-Backbone


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = layer_init(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1))
        self.conv2 = layer_init(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1))
        reduced_channels = max(1, channels // 16)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            layer_init(nn.Conv2d(channels, reduced_channels, kernel_size=1)),
            nn.GELU(),
            layer_init(nn.Conv2d(reduced_channels, channels, kernel_size=1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = F.gelu(self.conv1(x))
        out = self.conv2(out)
        w = self.se(out)
        out = out * w
        return F.gelu(out + x)

# Agent with CNN-Backbone, ScalarEncoder and zEncoder



# TODO (selfplay): funktioniert die SubEnvView so wie gewünscht? Testen
class SubEnvView:
    """
    Leichter Wrapper, um einen VecEnv-Transform, der nur eine Teilmenge
    der Environment-Indizes exponiert, ohne Kopien der Masken-/Space-Objekte.
    Gebraucht in selfplay_get_action, um die Enviornments auf die Agententypen aufzuteilen.
    """
    def __init__(self, base_env, indices):
        self.base = base_env
        self.indices = list(indices)
        self.num_envs = len(self.indices)
        self.action_plane_space = base_env.action_plane_space
    
    def debug_matrix_mask(self, i):
        return self.base.debug_matrix_mask(self.indices[i])
        # create a lightweight view that maps indices into the original env wrapper






class Agent(nn.Module):
    def __init__(self, mapsize=16 * 16, lstm_hidden=384, lstm_layers=3):
        super(Agent, self).__init__()
        self.mapsize = mapsize

        # CNN-Backbone
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(73, 64, kernel_size=3, stride=2, padding=1)),
            nn.GELU(),
            ResBlock(64),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)),
            nn.GELU(),
            ResBlock(64),
            layer_init(nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)),
            nn.GELU(),
            ResBlock(64),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 8 * 8, 256)),
            nn.ReLU()
        )
        self.z_embedding = nn.Embedding(num_embeddings=2, embedding_dim=8)
        self.z_encoder = ZSampler(obs_dim=self.mapsize * 73, z_dim=8)
        self.scalar_encoder = ScalarFeatureEncoder(11)
        # print(envsT.action_plane_space.nvec.sum())

        # Actor, Critic
        self.actor = layer_init(
            nn.Linear(
                256 +
                32 +
                8,
                self.mapsize *
                envsT.action_plane_space.nvec.sum()),
            std=0.01)
        self.critic = layer_init(nn.Linear(256 + 32 + 8, 1), std=1)

    def forward(self, x, sc, z):
        # in sc wurde Spieler 1 zu Spieler 0, für jedes 2te selfplay env (da der Agent immer aus Sicht von Spieler 0 handelt)
        # ScalarEncoder
        sc_feat = self.scalar_encoder(sc)

        # CNN-Backbone
        obs_feat = self.network(x.permute((0, 3, 1, 2)))
        # zEncoder
        z = z.view(z.size(0), -1)
        # concatination of CNN-Backbone, ScalarEncoder and zEncoder
        feat = torch.cat([obs_feat, sc_feat, z], dim=-1)

        return feat

    # Behavior Cloning Loss Function
    # Wahrscheinlichkeit dieselbe Action, wie der Expert, in jeder Position des Grids zu haben
    # mit Anzahl an Positionen im Batch normalisiert
    # + 0.01 * kl_loss
    def bc_loss_fn(self, obs, sc, expert_actions, z):

        B, H, W, C = obs.shape
        features = self.forward(obs, sc, z)
        flat = self.actor(features)
        grid_logits = flat.view(-1, envsT.action_plane_space.nvec.sum())
        split_logits = torch.split(
            grid_logits, envsT.action_plane_space.nvec.tolist(), dim=1)
        invalid_action_masks = torch.ones(
            (B * H * W,
             envsT.action_plane_space.nvec.sum() + 1),
            dtype=torch.bool).to(device)
        invalid_action_masks = invalid_action_masks.view(
            -1, invalid_action_masks.shape[-1])
        split_invalid_action_masks = torch.split(
            invalid_action_masks[:, 1:], envsT.action_plane_space.nvec.tolist(), dim=1
        )

        multi_categoricals = [
            CategoricalMasked(logits=l, masks=m)
            for (l, m) in zip(split_logits, split_invalid_action_masks)
        ]

        expert_actions = expert_actions.view(-1, expert_actions.shape[-1]).T

        logprob = torch.stack([categorical.log_prob(
            a) for a, categorical in zip(expert_actions, multi_categoricals)])

        bc_loss = - logprob.sum() / (B * H * W)

        kl = 0.0
        for cat in multi_categoricals:
            probs = cat.probs
            masked_cat = cat

            policy_dist = Categorical(logits=masked_cat.logits)
            expert_dist = Categorical(probs=probs)

            kl += kl_divergence(expert_dist, policy_dist).sum()

        kl_loss = kl / (B * H * W)

        return bc_loss + 0.01 * kl_loss
    
    def selfplay_get_action(self, x, sc, z, num_selfplay_envs, num_envs, action_old=None, agent_type=None, invalid_action_masks=None, envs=None):
        '''
        returns action, logprob, entropy, invalid_action_masks for selfplay and bot envs combined.
        Also returns action, logprob, entropy, invalid_action_masks for not main Agents
        '''



        if agent_type != None:
            indices_by_type = {t: (agent_type == t).nonzero(as_tuple=True)[0] for t in SelfplayAgentType}
            
            # Teile die Inputs nach Agententypen auf und speichere auch den Index der Envs, rufe get_action von den jeweiligen Agenten auf
            # TODO (selfplay): alle Daten werden hier kopiert aber ich verändere diese nicht -> versuche eine View jeweils daraus zu machen, benutze
            # z.B.: split_x = {t: [x[i] for i in idx] for t, idx in indices_by_type.items()} (array of Pointers)
            split_x = {t: x[idx] for t, idx in indices_by_type.items()}
            split_Sc = {t: sc[idx] for t, idx in indices_by_type.items()}
            split_z = {t: z[idx] for t, idx in indices_by_type.items()}
            split_action_old = {t: action_old[idx] for t, idx in indices_by_type.items()} \
                if action_old is not None else [None] * len(agent_type)
            split_invalid_action_masks = {t: invalid_action_masks[idx] for t, idx in indices_by_type.items()} \
                if invalid_action_masks is not None else [None] * len(agent_type)
            split_env_Indices = {t: idx for t, idx in indices_by_type.items()}

            # Handle current Main Agents in Selfplay Envs
            sub_env = SubEnvView(envs, split_env_Indices[0])
            cur_action, cur_logproba, cur_entropy, cur_invalid_action_masks = self.get_action(split_x[0], split_Sc[0], split_z[0], split_action_old[0], split_invalid_action_masks[0], sub_env)

            # for every agenttype: 1 SubEnvView (in ENUM method) with the respective indices
            split_action, split_logproba, split_entropy, split_invalid_action_masks = \
                SelfplayAgentType.get_agents_action(split_x, split_Sc, split_z, split_action_old, split_invalid_action_masks, split_env_Indices, envs)

            # concat Results from current main agent with the Results from other selfplaying Agents
            split_action = torch.cat([cur_action, split_action], dim=0)
            split_logproba = torch.cat([cur_logproba, split_logproba], dim=0)
            split_entropy = torch.cat([cur_entropy, split_entropy], dim=0)
            split_invalid_action_masks = torch.cat([cur_invalid_action_masks, split_invalid_action_masks], dim=0)


            # Handle non Selfplay Agents (current main Agent)
            if num_selfplay_envs < num_envs:
                agent_type_len = len(agent_type)
                env_view = SubEnvView(envs, list(range(agent_type_len, num_envs)))
                main_actions, main_logproba, main_entropy, main_invalid_action_masks = \
                        self.get_action(x[agent_type_len:], sc[agent_type_len:], z[agent_type_len:], action_old[agent_type_len:] if action_old is not None else None, invalid_action_masks[agent_type_len:] if invalid_action_masks is not None else None, env_view)
            
            # resort original order
            split_env_Indices = torch.cat([split_env_Indices[i] for i in range(len(split_env_Indices))], dim=0)
            
            action = torch.empty_like(split_action)
            for i, idx in enumerate(split_env_Indices):
                action[i] = split_action[idx]
            
            logprob = torch.empty_like(split_logproba)
            for i, idx in enumerate(split_env_Indices):
                logprob[i] = split_logproba[idx]
            
            entropy = torch.empty_like(split_entropy)
            for i, idx in enumerate(split_env_Indices):
                entropy[i] = split_entropy[idx]
            
            invalid_action_masks = torch.empty_like(split_invalid_action_masks)
            for i, idx in enumerate(split_env_Indices):
                invalid_action_masks[i] = split_invalid_action_masks[idx]
            
            if num_selfplay_envs < num_envs:
                # concatinate
                action = torch.cat([action, main_actions], dim=0)
                logprob = torch.cat([logprob, main_logproba], dim=0)
                entropy = torch.cat([entropy, main_entropy], dim=0)
                invalid_action_masks = torch.cat([invalid_action_masks, main_invalid_action_masks], dim=0)

            return action, logprob, entropy, invalid_action_masks
            
        else:
            return self.get_action(x, sc, z, action_old, invalid_action_masks, envs)



    # new envs statt envsT benutzen (man darf daran nichts verändern, da nicht alle envsT selfplay envs sind)
    def get_action(self, x, sc, z, action=None, invalid_action_masks=None, envs=None):
        logits = self.actor(self.forward(x, sc, z))
        # print("logits size:", logits.size())

        grid_logits = logits.view(-1, envs.action_plane_space.nvec.sum())
        split_logits = torch.split(
            grid_logits, envs.action_plane_space.nvec.tolist(), dim=1)
        # print("split_logits size:", [l.size() for l in split_logits])

        if action is None:
            all_arrays = []
            for i in range(envs.num_envs):
                # TODO (selfplay): Die Methode gibt 2 mal die Maske für Player 0 zurück nicht die Maske für Player 1 in der Reprästentation von Player 0
                arr = np.array(envs.debug_matrix_mask(i))
                all_arrays.append(arr)
            mask = np.stack(all_arrays)
            
            invalid_action_masks = torch.tensor(mask).to(device)

            # TODO (selfplay): drehe die Maske (Player 1 -> Player 0) für jedes 2te selfplay env (muss man nicht machen, da es schon in der debug_matrix_mask gemacht wird?) entfernen?
            # if args.num_selfplay_envs > 1:
            #     if 2 < args.num_selfplay_envs:
            #         tmp = invalid_action_masks[1:args.num_selfplay_envs:2].flip(1, 2).contiguous().clone()
            #         invalid_action_masks[1:args.num_selfplay_envs:2] = tmp
            #     else:
            #         tmp = invalid_action_masks[1].flip(0, 1).contiguous().clone()
            #         invalid_action_masks[1] = tmp

            # es wird immer der erste Wert in der Maske entfernt, weil es immer eine Positionsangabe geben darf
            invalid_action_masks = invalid_action_masks.view(
                -1, invalid_action_masks.shape[-1])
            split_invalid_action_masks = torch.split(
                invalid_action_masks[:, 1:], envs.action_plane_space.nvec.tolist(), dim=1
            )
            # split_invalid_action_masks: shape (16*16* num_envs, 7, ...)
            

            # TODO (selfplay): muss man nicht machen, da es schon in der debug_matrix_mask gemacht wird? entfernen? und ineffizient
            # if args.num_selfplay_envs > 1:
                # da die Dimensionen hintereinander gepackt sind -> jede 256 Positionen (16*16) sind ein Grid
                # Richtungen anpassen (move direction, harvest direction, return direction, produce direction)
                # for j in range(1, args.num_selfplay_envs, 2):
                #     for i in range(1, 5):
                #         split_invalid_action_masks[i][256*j:512*j] = torch.roll(split_invalid_action_masks[i][256*j:512*j], shifts=2, dims=1)
                #         
                #     # relative attack position anpassen (nur für a_r = 7)
                #     split_invalid_action_masks[6][256*j:512*j] = split_invalid_action_masks[6][256*j:512*j].flip(1)
            
            multi_categoricals = [
                CategoricalMasked(logits=l, masks=m)
                for (l, m) in zip(split_logits, split_invalid_action_masks)
            ]
            # multi_categoricals wo Mask == 0 -> -1e8 logit, Mask == 1 -> original logit

            action = torch.stack([c.sample() for c in multi_categoricals])
            # sample action from the categorical distribution
            # softmax(-1e8) ≈ 0

        else:
            # … expert‐action logprob branch (same as before) …
            invalid_action_masks = invalid_action_masks.reshape(
                -1, invalid_action_masks.shape[-1])
            action = action.view(-1, action.shape[-1]).T
            split_invalid_action_masks = torch.split(
                invalid_action_masks[:, 1:], envs.action_plane_space.nvec.tolist(), dim=1
            )
            multi_categoricals = [
                CategoricalMasked(logits=l, masks=m)
                for (l, m) in zip(split_logits, split_invalid_action_masks)
            ]

        logprob = torch.stack([categorical.log_prob(a)
                              for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy()
                              for categorical in multi_categoricals])
        num_predicted_parameters = len(envs.action_plane_space.nvec)

        logprob = logprob.T.reshape(-1, 256, num_predicted_parameters)
        entropy = entropy.T.reshape(-1, 256, num_predicted_parameters)

        action = action.T.reshape(-1, 256, num_predicted_parameters)

        invalid_action_masks = invalid_action_masks.view(
            -1, 256, envs.action_plane_space.nvec.sum() + 1)

        return action, logprob.sum(1).sum(1), entropy.sum(
            1).sum(1), invalid_action_masks

    def get_value(self, x, sc, z):
        return self.critic(self.forward(x, sc, z))


class ReplayDataset(IterableDataset):
    def __init__(self, data_files):
        self.data_files = data_files

    def __iter__(self):
        for path in self.data_files:
            try:
                with open(path, 'rb') as f:
                    dctx = zstd.ZstdDecompressor()
                    with dctx.stream_reader(f) as reader:
                        buffer = io.BytesIO(reader.read())
                        data = torch.load(
                            buffer, map_location="cpu", weights_only=True)

                for sample in zip(
                        data["obs"],
                        data["act"],
                        data["sc"],
                        data["z"]):
                    yield sample
                del data
                del buffer
                del sample

                gc.collect()

            except Exception as e:
                print(
                    f"[ReplayDataset] Error loading {os.path.basename(path)}: {e}")


# =========================
# Agent Setup
# =========================
agent = Agent().to(device)


# =========================


# (if resuming) Get the epoch, load the agent, start training from the loaded epoch
start_epoch = 1
if args.prod_mode and wandb.run.resumed:
    if run.summary.get('charts/BCepoch'):
        start_epoch = run.summary.get('charts/BCepoch') + 1
    else:
        start_epoch = 1

if args.resume:
    ckpt_path = f"models/agent.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
    agent.load_state_dict(
        torch.load(
            ckpt_path,
            map_location=device,
            weights_only=True))
    agent.train()  # agent.forward() nur im train mode
    print(f"resumed at epoch {start_epoch}")

    # new (moved out of if-Statement)
    path_BCagent = f"models/agent.pt"
    supervised_agent = Agent().to(device)
    supervised_agent.load_state_dict(
        torch.load(
            path_BCagent,
            map_location=device,
            weights_only=True))
    for param in supervised_agent.parameters():
        param.requires_grad = False
    supervised_agent.eval()


# =========================
# BC training Setup
# =========================

# new
BCtraining = args.BCtraining
newdata = args.newdata
nurwins = args.nurwins


if BCtraining:
    print("BC training Setup")
    if newdata:
        # temp
        # opponents =[
        #             [microrts_ai.coacAI,microrts_ai.workerRushAI], 
        #             [microrts_ai.coacAI,microrts_ai.passiveAI], 
        #             [microrts_ai.coacAI,microrts_ai.lightRushAI],
        #             [microrts_ai.mayari,microrts_ai.workerRushAI], 
        #             [microrts_ai.mayari,microrts_ai.passiveAI], 
        #             [microrts_ai.mayari,microrts_ai.lightRushAI],
        #             ]
        # new
        opponents =[
                    [microrts_ai.coacAI,microrts_ai.workerRushAI], 
                    [microrts_ai.coacAI,microrts_ai.passiveAI], 
                    [microrts_ai.coacAI,microrts_ai.lightRushAI], 
                    [microrts_ai.coacAI,microrts_ai.coacAI], 
                    [microrts_ai.coacAI,microrts_ai.mayari], 
                    [microrts_ai.coacAI,microrts_ai.randomAI], 
                    [microrts_ai.coacAI,microrts_ai.randomBiasedAI], 
                    [microrts_ai.coacAI, microrts_ai.rojo], 
                    [microrts_ai.coacAI, microrts_ai.mixedBot], # mixedBot erzeugt oft Fehler
                    [microrts_ai.coacAI, microrts_ai.izanagi], 
                    [microrts_ai.coacAI, microrts_ai.droplet], 
                    [microrts_ai.coacAI, microrts_ai.tiamat],
                    #-------------
                    [microrts_ai.mayari,microrts_ai.workerRushAI],
                    [microrts_ai.mayari,microrts_ai.passiveAI], 
                    [microrts_ai.mayari,microrts_ai.lightRushAI], 
                    [microrts_ai.mayari,microrts_ai.coacAI], 
                    [microrts_ai.mayari,microrts_ai.mayari], 
                    [microrts_ai.mayari,microrts_ai.randomAI], 
                    [microrts_ai.mayari,microrts_ai.randomBiasedAI], 
                    [microrts_ai.mayari, microrts_ai.rojo], 
                    [microrts_ai.mayari, microrts_ai.mixedBot], # mixedBot erzeugt oft Fehler
                    [microrts_ai.mayari, microrts_ai.izanagi], 
                    [microrts_ai.mayari, microrts_ai.droplet], 
                    [microrts_ai.mayari, microrts_ai.tiamat],
                    ]
        # end new
        # opponents =[[microrts_ai.workerRushAI,microrts_ai.workerRushAI], [microrts_ai.workerRushAI,microrts_ai.passiveAI],[microrts_ai.workerRushAI,microrts_ai.lightRushAI], [microrts_ai.workerRushAI,microrts_ai.coacAI], [microrts_ai.workerRushAI,microrts_ai.mayari], [microrts_ai.workerRushAI,microrts_ai.randomAI],[microrts_ai.workerRushAI,microrts_ai.randomBiasedAI],[microrts_ai.workerRushAI, microrts_ai.rojo],
        #             [microrts_ai.coacAI,microrts_ai.workerRushAI], [microrts_ai.coacAI,microrts_ai.passiveAI],[microrts_ai.coacAI,microrts_ai.lightRushAI], [microrts_ai.coacAI,microrts_ai.coacAI], [microrts_ai.coacAI,microrts_ai.mayari], [microrts_ai.coacAI,microrts_ai.randomAI],[microrts_ai.coacAI,microrts_ai.randomBiasedAI],[microrts_ai.coacAI, microrts_ai.rojo],
        #             [microrts_ai.mayari,microrts_ai.workerRushAI], [microrts_ai.mayari,microrts_ai.passiveAI], [microrts_ai.mayari,microrts_ai.lightRushAI],[microrts_ai.mayari,microrts_ai.coacAI], [microrts_ai.mayari,microrts_ai.mayari], [microrts_ai.mayari,microrts_ai.randomAI], [microrts_ai.mayari,microrts_ai.randomBiasedAI],[microrts_ai.mayari, microrts_ai.rojo],
        #             [microrts_ai.lightRushAI, microrts_ai.workerRushAI], [microrts_ai.lightRushAI, microrts_ai.passiveAI], [microrts_ai.lightRushAI, microrts_ai.lightRushAI], [microrts_ai.lightRushAI, microrts_ai.coacAI], [microrts_ai.lightRushAI,microrts_ai.mayari], [microrts_ai.lightRushAI, microrts_ai.randomAI], [microrts_ai.lightRushAI, microrts_ai.randomBiasedAI],[microrts_ai.lightRushAI, microrts_ai.rojo]]
        #num_runs =[200,50,200,100,300,200,500,200,200,50,200,100,300,200,500,200,200,50,200,100,300,200,500,200,200,50,200,100,300,200,500,200,200,50,200,100,300,200,500,200]
        # new
        # num_runs = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100] # muss so lang sein, wie len(opponents)
        # end new
        # temp
        # num_runs = [2, 2, 2, 2, 2, 2]
        num_runs = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        expert_name_to_id = {
            "coacAI": 0,
            "mayari": 1,
            # new
            "workerRushAI": 2,
            "passiveAI": 3,
            "lightRushAI": 4,
            "randomAI": 5,
            "randomBiasedAI": 6,
            "rojo": 7,
            "mixedbot": 8,
            "izanagi": 9,
            "droplet": 10,
            "tiamat": 11,
            # end new
        }

        # für jede AI Kombination in opponents ein Environment erstellen und
        # Spiele durchführen
        for l, x in enumerate(opponents):
            # ai_list = [microrts_ai.coacAI if i % 2 == 0 else opponents[(i // 2) % len(opponents)] for i in range(24)]

            # new
            print(f"next opponent: {x[0].__name__} vs. {x[1].__name__}")
            # end new

            # =========================
            # Game Setup
            # =========================
            ai_list = x

            env = MicroRTSBotGridVecEnv(
                max_steps=2048,
                ais=ai_list,
                # [microrts_ai.coacAI for _ in range(2)],  # CoacAI plays as second player
                map_paths=["maps/16x16/basesWorkers16x16.xml"],
                reference_indexes=[0]  # ,1,2,3,4,5,6,7,8,9,10,11]
            )

            envT = MicroRTSSpaceTransformbot(env)

            obs_batch, _, res = envT.reset()

            obsten = torch.zeros((0, 16, 16, 73), dtype=torch.int32)
            actten = torch.zeros((0, 256, 7), dtype=torch.int8)
            scten = torch.zeros((0, 11), dtype=torch.int8)
            ztorch = torch.zeros((0, 1), dtype=torch.int8)
            # =========================
            for ep in range(num_runs[l]):
                dones = np.array([False])
                step = 0

                obs_arr = []
                act_arr = []

                while not dones.all():
                    acti = []

                    if args.render:
                        if getattr(args, 'render_all', False):
                            # render every underlying env (inefficient, debug only)
                            render_all_envs(envT)
                        else:
                            envT.render()

                    obs_arr.append(obs_batch)  # initial observation speichern

                    scten = torch.cat(
                        [scten, getScalarFeatures(obs_batch, res, 1)], dim=0)

                    obs_batch, mask, dones, action, res, reward = envT.step("")

                    arr = np.zeros((256, 7), dtype=np.int64)

                    for j in range(len(action[0])):
                        arr[action[0][j][0]] = action[0][j][1:]

                    act_arr.append(arr)

                    step += 1

                if nurwins:
                    if reward.item() == 1:

                        obsten = torch.cat(
                            (obsten, torch.tensor(
                                np.array(obs_arr)).squeeze(1)), dim=0)
                        actten = torch.cat(
                            (actten, torch.tensor(np.array(act_arr))), dim=0)
                        ztorch = torch.cat((ztorch, torch.tensor(
                            expert_name_to_id[x[0].__name__]).repeat(len(obs_arr), 1)), dim=0)
                else:

                    obsten = torch.cat(
                        (obsten, torch.tensor(
                            np.array(obs_arr)).squeeze(1)), dim=0)
                    actten = torch.cat(
                        (actten, torch.tensor(
                            np.array(act_arr))), dim=0)
                    ztorch = torch.cat((ztorch, torch.tensor(
                        expert_name_to_id[x[0].__name__]).repeat(len(obs_arr), 1)), dim=0)
                print(ep)
                # new
                if (ep + 1) % 50 == 0 or (ep + 1) == num_runs[l]:
                    print("Collecting Data ep: ", l, " ", ep)
                    with open(f"replays/replay_{l}_up_to_ep{ep + 1}.pt.zst", 'wb') as f:
                        cctx = zstd.ZstdCompressor(level=1)

                        with cctx.stream_writer(f) as compressor:
                            buffer = io.BytesIO()
                            torch.save({
                                'obs': obsten,
                                'act': actten,
                                'sc': scten,
                                'z': ztorch,
                            }, buffer)
                            compressor.write(buffer.getvalue())

                    obsten = torch.zeros((0, 16, 16, 73), dtype=torch.int32)
                    actten = torch.zeros((0, 256, 7), dtype=torch.int32)
                    scten = torch.zeros((0, 11), dtype=torch.int8)
                    ztorch = torch.zeros((0, 1), dtype=torch.int8)

                # z.append(torch.cat((buildorder,custat), dim=0))
        envT.close()
        env.close()
        # z_np = torch.stack(z).numpy()

        # np.savetxt("zFeatures.txt", z_np, fmt="%.6f")

# =========================


# =========================
# BC training
# =========================
    print("BC training started")
    all_files = sorted([
        os.path.join("replays/", f)
        for f in os.listdir("replays/")
        if f.endswith(".pt.zst")
    ])
    replay_dataset = ReplayDataset(all_files)

    agent.train()  # agent.forward() nur im train mode
    optimizer = optim.Adam(agent.parameters(),
                           lr=1e-4,
                           eps=1e-6,)
    # weight_decay=1e-5)

    # DataLoader (Replay datensatz bestimmen)
    train_loader = DataLoader(
        replay_dataset,
        batch_size=2048,
        num_workers=0,
        pin_memory=True,
        pin_memory_device="cuda",


    )

    warmup_epochs = 20

    for epoch in range(start_epoch, start_epoch + 1000):
        train_loss_sum = 0.0
        train_count = 0
        # nach warmup_epochs epochen wird alpha = 0
        alpha = max(0.0, 1.0 - (epoch - start_epoch) / warmup_epochs)

        for obs, expert_actions, sc, zt in train_loader:

            obs = obs.to(device)
            expert_actions = expert_actions.to(device)
            sc = sc.to(device)
            zt = zt.to(device)

            optimizer.zero_grad()

            z_embed = agent.z_embedding(zt)
            z_enc = agent.z_encoder(obs.view(obs.size(0), -1))
            z = alpha * z_embed.squeeze(1) + (1 - alpha) * z_enc

            loss = agent.bc_loss_fn(obs, sc, expert_actions, z)

            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

            train_loss_sum += loss.item() * obs.size(0)
            train_count += obs.size(0)

        avg_train_loss = train_loss_sum / train_count

        writer.add_scalar("charts/BCepoch", epoch)
        writer.add_scalar("charts/BCLossTrain", avg_train_loss, epoch)

        os.makedirs(f"models/{experiment_name}", exist_ok=True)
        torch.save(agent.state_dict(), f"models/{experiment_name}/agent.pt")
        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.6f}")

    print("BC training finished")
# =========================


# =========================
# PPO training Setup
# =========================




def adjust_obs_selfplay(args, next_obs, is_new_env=False):
    if is_new_env:
            # flippe jede zweite selfplay Umgebung (Spieler 1 -> Spieler 0)
            # da keine Unit eine Richtung bekommen hat müssen die Richtungen nicht angepasst werden
        if args.num_selfplay_envs > 1:
            if 2 < args.num_selfplay_envs:
                tmp = next_obs[1:args.num_selfplay_envs:2].flip(1, 2).contiguous().clone()
                next_obs[1:args.num_selfplay_envs:2] = tmp
                next_obs[1:args.num_selfplay_envs:2] = tmp
            else:
                tmp = next_obs[1].flip(0, 1).contiguous().clone()
                next_obs[1] = tmp
            return

    if args.num_selfplay_envs > 1:
            # jede zweite selfplay Umgebung:
        if 2 < args.num_selfplay_envs:
            tmp = next_obs[1:args.num_selfplay_envs:2].flip(1, 2).contiguous().clone()

                # flip Observations (Spieler 1 -> Spieler 0)
            next_obs[1:args.num_selfplay_envs:2] = tmp

                # switch players in the observation (player 1 -> player 0) 
            # next_obs[1:args.num_selfplay_envs:2, :, :, 4:6:-1] = next_obs[1:args.num_selfplay_envs:2, :, :, 6:4] muss man nicht machen (sind schon gedreht), wenn doch --> auch wenn is_new_env=True, im else-Teil
            # next_obs[1:args.num_selfplay_envs:2, :, :, 59:66] = tmp[:, :, :, 66:73]
            # next_obs[1:args.num_selfplay_envs:2, :, :, 66:73] = tmp[:, :, :, 59:66]

                # rottate directions 180°
            next_obs[1:args.num_selfplay_envs:2, :, :, 21:41] = (next_obs[1:args.num_selfplay_envs:2, :, :, 21:41] + 2) % 4
            next_obs[1:args.num_selfplay_envs:2, :, :, 49:54] = (next_obs[1:args.num_selfplay_envs:2, :, :, 49:54] + 2) % 4

        else:
            tmp = next_obs[1].flip(0, 1).contiguous().clone()
            next_obs[1] = tmp

                # switch players in the observation (player 1 -> player 0)
            # next_obs[1, :, :, 4] = tmp[:, :, 5]
            # next_obs[1, :, :, 5] = tmp[:, :, 4]
            # next_obs[1, :, :, 59:66] = tmp[:, :, 66:73]
            # next_obs[1, :, :, 66:73] = tmp[:, :, 59:66]

                # rottate directions 180° TODO (selfplay): auch alle Richtungen, die nicht benutzt werden, werden geändert (benutze torch.roll(next_obs[...], shifts=2, dims=...))
            permutation = [21, 24, 25, 22, 23, 26, 29, 30, 27, 28, 31, 34, 35, 32, 33, 36, 39, 40, 37, 38]
            for i, p in enumerate(permutation):
                next_obs[1, :, :, i+21] = tmp[:, :, p]
    
            permutation = [49, 52, 53, 50, 51]
            for i, p in enumerate(permutation):
                next_obs[1, :, :, i+49] = tmp[:, :, p]

def adjust_action_selfplay(args, device, valid_actions, valid_actions_counts):
    if args.num_selfplay_envs > 1:
            # Position anpassen
        index = 0
        for j, i in enumerate(valid_actions_counts):
            
            if j % 2 == 1 and j < args.num_selfplay_envs:
                # Position anpassen
                valid_actions[index:index+i, 0] = np.abs(valid_actions[index:index+i, 0] - 255)
                # Richtungen anpassen (move direction, harvest direction, return direction, produce direction)
                valid_actions[index:index+i, 2:6] = (valid_actions[index:index+i, 2:6] + 2) % 4
                # relative attack position anpassen (nur für a_r = 7)
                valid_actions[index:index+i, 7] = np.abs(valid_actions[index:index+i, 7] - 48)
            index += i

        # real_action[i, :, 0] = torch.tensor(range(255, -1, -1)).to(device)
            # TO DO (selfplay): wird die Arrayposition der Spielpositionen vorausgesetzt? (muss es aufsteigend sortiert sein?) (wenn nicht --> unten entfernen)
        # real_action[1:args.num_selfplay_envs:2] = real_action[1:args.num_selfplay_envs:2].flip(1)
            
            # Richtungen anpassen (move direction, harvest direction, return direction, produce direction)
        # real_action[1:args.num_selfplay_envs:2, :, 2:6] = (real_action[1:args.num_selfplay_envs:2, :, 2:6] + 2) % 4
            # relative attack position anpassen (nur für a_r = 7)
        #real_action[1:args.num_selfplay_envs:2, :, 7] = torch.abs(real_action[1:args.num_selfplay_envs:2, :, 7] - 48)



print("PPO training Setup")


optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
if args.anneal_lr:
    def lr(f): return f * args.learning_rate


mapsize = 16 * 16

action_space_shape = (mapsize, envsT.action_plane_space.shape[0])
invalid_action_shape = (mapsize, envsT.action_plane_space.nvec.sum() + 1)


obs = torch.zeros((args.num_steps, args.num_envs) +
                  envsT.single_observation_space.shape).to(device)
actions = torch.zeros((args.num_steps, args.num_envs) +
                      action_space_shape).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)

# rewards_dense = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards_attack = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards_winloss = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards_score = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps, args.num_envs)).to(device)
invalid_action_masks = torch.zeros(
    (args.num_steps, args.num_envs) + invalid_action_shape).to(device)

global_step = 0
start_time = time.time()
ob, mas, res = envsT.reset()

next_obs = torch.Tensor(ob).to(device)

ScFeatures = torch.zeros((args.num_steps, args.num_envs, 11)).to(device)
ScFeatures[0] = getScalarFeatures(next_obs, res, args.num_envs)


adjust_obs_selfplay(args, next_obs, is_new_env=True)

next_done = torch.zeros(args.num_envs).to(device)
num_updates = args.total_timesteps // args.batch_size

zFeatures = torch.zeros(
    (args.num_steps,
     args.num_envs,
     8),
    dtype=torch.long).to(device)


starting_update = 1
# =========================







# =========================
# PPO training
# =========================

print("PPO training started")

for update in range(starting_update, num_updates + 1):

    # =========================
    # PPO update step setup (kreiere PPO-Rollout-Batch)
    # Speichere für alle Enviorments pro Schritt
    # observations
    # actions
    # logprobs
    # values
    # rewards
    # dones
    # invalid_action_masks
    # =========================

    if args.anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = lr(frac)
        optimizer.param_groups[0]['lr'] = lrnow

    for step in range(0, args.num_steps):

        for i in range(args.num_envs):
            with torch.no_grad():
                # obs sind zuerst alles 0en, dannach jeweils Spieler 1 zu Spieler 0 geändert
                zFeatures[step][i] = agent.z_encoder(obs[step][i].view(-1))

        
        if args.render:
            if args.render_all:
                render_all_envs(envsT)
            else:
                envsT.render("human")

        global_step += 1 * args.num_envs

        obs[step] = next_obs

        dones[step] = next_done

        with torch.no_grad():
            values[step] = agent.get_value(obs[step], ScFeatures[step], zFeatures[step]).flatten()  # critic(forward(...))
            

            # gesamplete action (aus Verteilung der Logits) (24, 256, 7),
            # actor(forward(...)), invalid_action_masks
            # obs sind zuerst alles 0en, dannach jeweils Spieler 1 zu Spieler 0 geändert
            action, logproba, _, invalid_action_masks[step] = agent.selfplay_get_action(
                obs[step], ScFeatures[step], zFeatures[step], args.num_selfplay_envs, args.num_envs, agent_type=agent_type, envs=envsT)

        # (Shape: (step, num_envs, 256 (16 * 16), action) (step, 24, 256, 7))
        actions[step] = action
        # print("actions shape per step:", actions[step].shape)

        logprobs[step] = logproba

        # Die Grid-Position zu jedem Action hinzugefügt (24, 256, 8)
        real_action = torch.cat([torch.stack([torch.arange(0, mapsize).to(
            device) for i in range(envsT.num_envs)]).unsqueeze(2), action], 2)
        # print("real_action shape:", real_action.shape)
        # print("Grid-Position:", [real_action[0][i][0].item() for i in
        # range(10)]) # -> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        real_action = real_action.cpu().numpy()
    
        # =============
        # invalid_action_masks angewandt
        # =============


        # Debug Beispiel
        # valid_actions = np.array([np.array([34.0, 0.0, 1.0, 3.0, 1.0, 2.0, 3.0, 21.0]),
        #                            np.array([238.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        #                            np.array([34.0, 0.0, 2.0, 0.0, 0.0, 2.0, 3.0, 24.0])])
        # valid_actions_counts = [1, 1, 1]
        valid_actions = real_action[invalid_action_masks[step]
                                    [:, :, 0].bool().cpu().numpy()]
        valid_actions_counts = invalid_action_masks[step][:, :, 0].sum(
            1).long().cpu().numpy()
        

        # Anpassungen für Spieler 1 nach (Spieler 1 -> Spieler 0)
        adjust_action_selfplay(args, device, valid_actions, valid_actions_counts)
        # TODO (selfplay): actions wurden bis hier für step 0 durchgegangen#########################################################################################################################################

        '''
        valid_actions:
        [[Pos, Type, move direction, harvest direction, return (recource) direction, produce direction, produce type, relative attack position],
         [Spiel0 (Spieler1)],
         [Spiel1 (Spieler0)]]

        Pos: 0-255 (16*16) links oben nach rechts unten (obenecke = 0)
        Type: 0: NOP, 1: Move, 2: Harvest, 3: Return, 4: Produce (Produce direction + Produce type), 5: Attack (wenn z.B.: move direction = 1, aber Type = 2 --> move direction wird ignoriert)
        direction: 0: North, 1: East, 2: South, 3: West
        produce type: 0: (light), 1: (Ranged), 2: (Baracks / Heavy), 3: (Worker) (je nach Unit unterschiedlich)
        relative attack position: 0-255 (16*16) links oben nach rechts unten (obenecke = 0) wo angegriffen wird
        '''



        java_valid_actions = []
        valid_action_idx = 0
        for env_idx, valid_action_count in enumerate(valid_actions_counts):
            java_valid_action = []
            for c in range(valid_action_count):
                java_valid_action += [JArray(JInt)
                                      (valid_actions[valid_action_idx])]
                valid_action_idx += 1
            java_valid_actions += [JArray(JArray(JInt))(java_valid_action)]
        java_valid_actions = JArray(JArray(JArray(JInt)))(java_valid_actions)
        # java_valid_actions.shape: (Envs, num_valid_actions_in_Env, valid_action (8)) (py_arr = np.array(java_valid_actions))
        # np_valid_actions = np.array(
        # [[np.array(list(inner), dtype=np.int32) for inner in outer]
        #  for outer in java_valid_actions],
        # dtype=object
        # )
        # =============

        # =============
        # Schritt in der Umgebung mit der in get_action gesampleten Action
        # =============
        try:
            next_obs, denserew, attackrew, winlossrew, scorerew, ds, infos, res = envsT.step(
                java_valid_actions)
            next_obs = envsT._from_microrts_obs(next_obs) # next_obs zu Tensor mit shape (24, 16, 16, 73) (von (24, X))
            next_obs = torch.Tensor(next_obs).to(device)
        except Exception as e:
            e.printStackTrace()
            raise

        # TODO (selfplay): wenn die Schleife für updates neu anfängt: muss eigentlich noch ein mal getScalarFeatures ausgeführt werden
        if step + 1 < args.num_steps:
            ScFeatures[step+1] = getScalarFeatures(next_obs, res, args.num_envs)

        adjust_obs_selfplay(args, next_obs)

        '''winloss = min(0.01, 6.72222222e-9 * global_step)
        densereward = max(0, 0.8 + (-4.44444444e-9 * global_step))

        if global_step < 100000000:
            scorew = 0.19 + 1.754e-8 * global_step
        else:
            scorew = 0.5 - 1.33e-8 * global_step'''
        densereward = 0
        winloss = 10
        scorew = 0.2
        attack = 0.05
        # =============

        # update rewards
        # rewards_dense[step] = torch.Tensor(denserew* densereward).to(device)




        ### Debugging (scorerews always == 0)
        # Breakpoint if values change
        # if not np.array_equal(scorerews, _last_scorerews):
        #     breakpoint()
        # _last_scorerews = np.copy(scorerews)

        rewards_attack[step] = torch.Tensor(attackrew * attack).to(device)
        rewards_winloss[step] = torch.Tensor(winlossrew * winloss).to(device)
        rewards_score[step] = torch.Tensor(scorerew * scorew).to(device)
        next_done = torch.Tensor(ds).to(device)

        # =============
        # Logging PPO training
        # =============
        for info in infos:

            if 'episode' in info.keys():

                game_length = info['episode']['l']
                winloss = winloss * (-0.00013 * game_length + 1.16) # ca. 0.9 bei 2000 und 1.1 bei 500
                print(f"global_step={global_step}, episode_reward={info['microrts_stats']['RAIWinLossRewardFunction'] * winloss + info['microrts_stats']['AttackRewardFunction'] * attack}")
                writer.add_scalar("charts/old_episode_reward", info['episode']['r'], global_step)
                writer.add_scalar("charts/Game_length", game_length,global_step)
                writer.add_scalar("charts/Episode_reward", info['microrts_stats']['RAIWinLossRewardFunction'] * winloss + info['microrts_stats']['AttackRewardFunction'] * attack, global_step)
                writer.add_scalar("charts/AttackReward", info['microrts_stats']['AttackRewardFunction'] * attack, global_step)
                writer.add_scalar("charts/WinLossRewardFunction", info['microrts_stats']['RAIWinLossRewardFunction'] * winloss, global_step)

                break
        # =============
    # =========================



# =========================
# PPO update
# =========================

    with torch.no_grad():

        last_value = agent.get_value(
            next_obs.to(device), ScFeatures[step], zFeatures[step]).reshape(
            1, -1)

        # =============
        # GAE
        # =============

        advantages = torch.zeros_like(rewards_winloss).to(device)
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                # zuerst nonterminal, nextvalues aus dem Schritt im Setup
                # benutzen (anders repräsentiert)
                nextnonterminal = 1.0 - next_done
                nextvalues = last_value
            else:
                # für jede Umgebung: 1 -> nicht done in step t+1, 0 -> done in
                # step t+1
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]

            # TD-Error = R_(t+1) + γ * V(S_(t+1)) - V(S_t) per environment
            # V(S_t): Value of the state reached in the rollout after Action in step t-1
            # nextvalues: Critic-approximated values per environment in step t
            # for the next step in the rollout (if not terminated)
            # rewards_dense[t] + + rewards_dense[t] +rewards_score[t]
            delta = rewards_winloss[t] + rewards_attack[t] + \
                args.gamma * nextvalues * nextnonterminal - values[t]
            # A_t="TD-Error" + γ * λ * A_(t-1)
            advantages[t] = lastgaelam = delta + args.gamma * \
                args.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values
        # =============

    # flatten the batch
    # args.num_steps, args.num_envs Dimensionen vereinigen  (shape (steps*envs, 8))
    # (zFeatures für jeden Step, Environment sortiert Step, dann nach Environments)
    b_z = zFeatures.reshape(-1, 8)
    # args.num_steps, args.num_envs Dimensionen vereinigen  (shape (steps*envs, 11))
    # (ScFeatures für jeden Step, Environment sortiert Step, dann nach Environments)
    b_Sc = ScFeatures.reshape(-1, 11)
    # dasselbe mit obs                                      (shape (steps*envs, 16, 16, 73))
    b_obs = obs.reshape((-1,) + envsT.single_observation_space.shape)
    # dasselbe mit actions                                  (shape (steps*envs, 256, 7))
    b_actions = actions.reshape((-1,) + action_space_shape)
    # dasselbe mit logprobs, advantages, returns, values    (shape (steps*envs,))
    b_logprobs = logprobs.reshape(-1)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)
    # dasselbe mit invalid_action_masks                     (shape (steps*envs, 256, 79))
    b_invalid_action_masks = invalid_action_masks.reshape(
        (-1,) + invalid_action_shape)

    # Optimizing policy and value network with minibatch updates
    # --num_minibatches, --update-epochs
    # minibatches_size = int(args.batch_size // args.num_minibatches)
    # inds: indices from the batch
    inds = np.arange(args.batch_size, )

    # Go (update_epochs times) through all mini-batches
    '''
    für jeden Minibatch im Batch berechne Â_t (Advantage Schätzer (R_t^((λ))-V_ϕ^(π_old)) (hier eher (V_ϕ (s_t )-R_t^((λ)) und dann - genommen in pg_loss))) (normalisiert),
    Wahrscheinlichkeit Action a in State s zu bekommen mit neuem θ / Wahrscheinlichkeit Action a in State s zu bekommen mit altem θ_old
    pgLoss (gegenteil von L_clip) ausrechnen, kombinieren mit Entropie Bonus, KL Divergenz Loss und Value Loss mit Updates minimieren
    '''
    for i_epoch_pi in range(args.update_epochs):
        np.random.shuffle(inds)

        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            minibatch_ind = inds[start:end]
            mb_advantages = b_advantages[minibatch_ind]

            # normalize the advantages
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()
                                 ) / (mb_advantages.std() + 1e-8)

            # forward pass: get network output for the minibatch
            # We also provide actions here
            new_values = agent.get_value(
                b_obs[minibatch_ind], b_Sc[minibatch_ind], b_z[minibatch_ind]).view(-1)
            

            # get_action nur für logprobs und entropy, um ratio zu berechnen (um zu vergleichen, wie wahrscheinlich die Action mit dem neuen θ im Vergleich zu dem alten θ_old ist)
            # TODO (League training): Die Enviornments für die Exploiters vorher entfernen??? (wenn, dann vor dem Schuffeln (np.random.shuffle(inds)))
            # (oder einfach auch berechnen und dann das Update 0 werden lassen 
            # (z.B.: ratio 0 werden lassen indem die Actions der Exploiter None zurück zu PPO geben (dann auf None testen)))
            # oder einfach wie in BC von den Exploitern auch trainieren (vielleicht schlechter, da die Exploiter nicht optimal spielen und ein Bias in die Richtung entsteht)
            _, newlogproba, entropy, _ = agent.get_action(
                b_obs[minibatch_ind],
                b_Sc[minibatch_ind],
                b_z[minibatch_ind],
                b_actions.long()[minibatch_ind],
                b_invalid_action_masks[minibatch_ind],
                envsT)
            ratio = (newlogproba - b_logprobs[minibatch_ind]).exp()

            # for logging
            approx_kl = (b_logprobs[minibatch_ind] - newlogproba).mean()

            # Policy loss L^CLIP(θ) = E ̂_t ["min" (r_t (θ)*Â_t,"clip" (r_t (θ),1-ϵ,1+ϵ)*Â_t )]
            # --clip-coef
            # pg_loss = -L^CLIP(θ) (opposite)
            # it is the same (but negative), because in loss1 and 2 there is a minus sign and advantages are calculated differently
            # geht gegen 0, wenn es keine Verbesserung mehr gibt
            # gibt ein Wert für die Verbesserung der Policy in der aktuellen Iteration an
            # < 0  ⇒ Surrogate im Mittel verbessert (guter Update) (Policy verbessert sich)
            # ≈ 0  ⇒ kaum/keine (geclippte) Verbesserung
            # > 0  ⇒ Surrogate im Mittel schlechter (schlechter Update)
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * \
                torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            entropy_loss = entropy.mean()

            # Value loss Clipping
            # --clip_vloss
            # MSE(approximierte Values, returns) with or without clip()
            if args.clip_vloss:
                v_loss_unclipped = (
                    (new_values - b_returns[minibatch_ind]) ** 2)
                v_clipped = b_values[minibatch_ind] + torch.clamp(
                    new_values - b_values[minibatch_ind], -args.clip_coef, args.clip_coef)
                v_loss_clipped = (v_clipped - b_returns[minibatch_ind]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((new_values - b_returns[minibatch_ind]) ** 2)

            # KL Divergence Loss
            with torch.no_grad():
                
                # get_action nur für logprobs, um KL Divergenz zu berechnen
                _, sl_logprobs, _, _ = supervised_agent.get_action(
                    b_obs[minibatch_ind],
                    b_Sc[minibatch_ind],
                    b_z[minibatch_ind],
                    b_actions.long()[minibatch_ind],
                    b_invalid_action_masks[minibatch_ind],
                    envsT
                )
            kl_div = F.kl_div(
                newlogproba,
                sl_logprobs,
                log_target=True,
                reduction="batchmean")
            kl_loss = args.kl_coeff * kl_div

            loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
# =========================

    # =========================
    # Logging PPO update
    # =========================

    if args.prod_mode and update % CHECKPOINT_FREQUENCY == 0:
        print("checkpoint")
        os.makedirs(f"models/{experiment_name}", exist_ok=True)
        torch.save(agent.state_dict(), f"models/{experiment_name}/agent.pt")


    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], global_step)
    writer.add_scalar("charts/update", update, global_step)
    writer.add_scalar("losses/value_loss", args.vf_coef * v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/kl_loss", kl_loss.item(), global_step)
    writer.add_scalar("losses/total_loss", loss.item(), global_step)
    writer.add_scalar("losses/entropy_loss", args.ent_coef * entropy_loss.item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    if args.kle_stop or args.kle_rollback:
        writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)
    writer.add_scalar("charts/sps", int(global_step / (time.time() - start_time)), global_step)
    print("SPS:", int(global_step / (time.time() - start_time)))

envsT.close()
writer.close()
# =========================
