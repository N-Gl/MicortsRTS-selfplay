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
import zstandard as zstd, io
import gym
import gym_microrts
#from gym_microrts.envs.vec_env2 import MicroRTSGridModeVecEnv
from gym_microrts.envs.microrts_vec_env import  MicroRTSGridModeVecEnv
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
    parser.add_argument('--num-bot-envs', type=int, default=24,
                        help='the number of bot game environment; 16 bot envs measn 16 games')
    parser.add_argument('--num-selfplay-envs', type=int, default=0,
                        help='the number of self play envs; 16 self play envs means 16 games')
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

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())


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
        obs, denserews, attackrews, winlossrews, scorerews , dones, infos, res = self.venv.step_wait()

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
    import  wandb
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
        config=vars(args), name=experiment_name, monitor_gym=True,resume=resume_mode,id=run_id, save_code=False)

    if resume_mode == "allow":
        os.makedirs(os.path.dirname(RUN_ID_PATH), exist_ok=True)
        with open(RUN_ID_PATH, "w") as f:
            f.write(run.id)
    wandb.tensorboard.patch(save=False)
    writer = SummaryWriter(f"/tmp/{experiment_name}")
    CHECKPOINT_FREQUENCY = 10
# =========================

# device setup
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
print(device)


# =========================
# creating Environments
# =========================
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
envs = MicroRTSGridModeVecEnv(
    num_selfplay_envs=args.num_selfplay_envs,
    num_bot_envs=args.num_bot_envs,
    max_steps=2000,
    render_theme=1,
    ai2s=[microrts_ai.coacAI for _ in range(3)] +[microrts_ai.mayari for _ in range(4)] +[microrts_ai.mixedBot for _ in range(4)] +[microrts_ai.izanagi for _ in range(3)] +[microrts_ai.droplet for _ in range(4)] +[microrts_ai.tiamat for _ in range(3)] +[microrts_ai.workerRushAI for _ in range(3)],
    map_paths=["maps/16x16/basesWorkers16x16.xml"],
    reward_weight=np.array([1.0, 1.0, 1.0, 0.2, 1.0, 4.0, 5.25, 6.0, 0])
)
envsT = MicroRTSSpaceTransform(envs)

envsT = VecstatsMonitor(envsT, args.gamma)
# Video?
if args.capture_video:
    envs = VecVideoRecorder(envs, f'videos/{experiment_name}',
                            record_video_trigger=lambda x: x % 1000000 == 0, video_length=2000)
# =========================



from torch.utils.data import DataLoader

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(device))
        return -p_log_p.sum(-1)


# get scalar feature representation from observation (globale Zustandsinformationen in das Modell hinzufügen) for ScalarEncoder
def getScalarFeatures(obs,res, numenvs):
    ScFeatures = torch.zeros(numenvs, 11)

    for i in range(numenvs):

        res_plane = (obs[i, :, :, 1]*obs[i, :, :, 7])
        lightunit_plane = (obs[i, :, :, 11])
        heavyunit_plane = (obs[0, :, :, 12])
        rangedunit_plane = (obs[0, :, :, 13])
        total_res = res_plane.sum().item()


        worker_plane = obs[i, :, :, 10]
        building_plane = obs[i, :, :, 9]
        player0_plane = obs[i, :, :, 4]
        player1_plane = obs[i, :, :, 5]

        ScFeatures[i,0]  = res[i][0] #Player0 res
        ScFeatures[i, 1] =  res[i][1] #Player1 res
        ScFeatures[i, 2] =  total_res #vorhandene res
        ScFeatures[i, 3] = (worker_plane * player0_plane).sum().item()  # Player0 worker
        ScFeatures[i, 4] = (lightunit_plane * player0_plane).sum().item()  # Player0 light
        ScFeatures[i, 5] = (heavyunit_plane * player0_plane).sum().item()  # Player0 heavy
        ScFeatures[i, 6] = (rangedunit_plane * player0_plane).sum().item()  # Player0 ranged
        ScFeatures[i, 7] = (worker_plane * player1_plane).sum().item()  # Player1 worker
        ScFeatures[i, 8] = (lightunit_plane * player1_plane).sum().item()  # Player1 light
        ScFeatures[i, 9] = (heavyunit_plane * player1_plane).sum().item()  # Player1 heavy
        ScFeatures[i, 10] = (rangedunit_plane * player1_plane).sum().item()  # Player1 ranged

    #Time step in the game
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
    def __init__(self, input_dim, hidden_dim = 32, output_dim = 32, num_layers=2):
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
        self.conv1 = layer_init(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        self.conv2 = layer_init(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
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
        self.scalar_encoder  = ScalarFeatureEncoder(11)
        #print(envsT.action_plane_space.nvec.sum())

        #Actor, Critic
        self.actor = layer_init(nn.Linear(256+32+ 8, self.mapsize * envsT.action_plane_space.nvec.sum()), std=0.01)
        self.critic = layer_init(nn.Linear(256+32+ 8, 1), std=1)

    def forward(self, x,sc,z):


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
    def bc_loss_fn(self, obs,sc, expert_actions,z):

        B, H, W, C = obs.shape
        features = self.forward(obs,sc,z)
        flat = self.actor(features)
        grid_logits = flat.view(-1, envsT.action_plane_space.nvec.sum())
        split_logits = torch.split(grid_logits, envsT.action_plane_space.nvec.tolist(), dim=1)
        invalid_action_masks = torch.ones((B * H*W, envsT.action_plane_space.nvec.sum()+1), dtype=torch.bool).to(device)
        invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
        split_invalid_action_masks = torch.split(
            invalid_action_masks[:, 1:], envsT.action_plane_space.nvec.tolist(), dim=1
        )

        multi_categoricals = [
            CategoricalMasked(logits=l, masks=m)
            for (l, m) in zip(split_logits, split_invalid_action_masks)
        ]

        expert_actions = expert_actions.view(-1, expert_actions.shape[-1]).T



        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(expert_actions, multi_categoricals)])

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




    def get_action(self, x,Sc,z ,action=None, invalid_action_masks=None, envs=None):
        logits = self.actor(self.forward(x,Sc,z))
        # print("logits size:", logits.size())

        grid_logits = logits.view(-1, envsT.action_plane_space.nvec.sum())
        split_logits = torch.split(grid_logits, envsT.action_plane_space.nvec.tolist(), dim=1)
        # print("split_logits size:", [l.size() for l in split_logits])

        if action is None:
            all_arrays = []
            for i in range(envsT.num_envs):
                arr = np.array(envsT.debug_matrix_mask(i))
                all_arrays.append(arr)
            mask = np.stack(all_arrays)
            invalid_action_masks = torch.tensor(mask).to(device)
            invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
            split_invalid_action_masks = torch.split(
                invalid_action_masks[:, 1:], envsT.action_plane_space.nvec.tolist(), dim=1
            )
            multi_categoricals = [
                CategoricalMasked(logits=l, masks=m)
                for (l, m) in zip(split_logits, split_invalid_action_masks)
            ]
            action = torch.stack([c.sample() for c in multi_categoricals])
        else:
            # … expert‐action logprob branch (same as before) …
            invalid_action_masks = invalid_action_masks.reshape(-1, invalid_action_masks.shape[-1])
            action = action.view(-1, action.shape[-1]).T
            split_invalid_action_masks = torch.split(
                invalid_action_masks[:, 1:], envsT.action_plane_space.nvec.tolist(), dim=1
            )
            multi_categoricals = [
                CategoricalMasked(logits=l, masks=m)
                for (l, m) in zip(split_logits, split_invalid_action_masks)
            ]

        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        num_predicted_parameters = len(envsT.action_plane_space.nvec)


        logprob = logprob.T.reshape(-1, 256, num_predicted_parameters)
        entropy = entropy.T.reshape(-1, 256, num_predicted_parameters)

        action = action.T.reshape(-1, 256, num_predicted_parameters)

        invalid_action_masks = invalid_action_masks.view(-1, 256, envsT.action_plane_space.nvec.sum() + 1)

        return action, logprob.sum(1).sum(1), entropy.sum(1).sum(1), invalid_action_masks

    def get_value(self, x, Sc, z):
        return self.critic(self.forward(x, Sc, z))


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
                        data = torch.load(buffer, map_location="cpu", weights_only=True)

                for sample in zip(data["obs"], data["act"], data["sc"], data["z"]):
                    yield sample
                del data
                del buffer
                del sample

                gc.collect()

            except Exception as e:
                print(f"[ReplayDataset] Error loading {os.path.basename(path)}: {e}")



# =========================
# Agent Setup
# =========================

agent = Agent().to(device)


# new
path_BCagent = f"models/BCagent.pt"
supervised_agent = Agent().to(device)
# =========================


# (if resuming) Get the epoch, load the agent, start training from the loaded epoch
start_epoch = 1
if args.prod_mode and wandb.run.resumed:
    if run.summary.get('charts/BCepoch'):
        start_epoch = run.summary.get('charts/BCepoch') + 1
    else:
        start_epoch =  1

    ckpt_path = f"models/{experiment_name}/agent.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
    agent.load_state_dict(torch.load(ckpt_path, map_location=device,weights_only=True))
    agent.train() # agent.forward() nur im train mode
    print(f"resumed at epoch {start_epoch}")

    # new (moved out of if-Statement)
    # path_BCagent = f"models/BCagent.pt"
    # supervised_agent = Agent().to(device)
    supervised_agent.load_state_dict(torch.load(path_BCagent, map_location=device, weights_only= True))
    for param in supervised_agent.parameters():
        param.requires_grad = False
    supervised_agent.eval()


# =========================
# PPO training Setup
# =========================


print("PPO training Setup")


optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
if args.anneal_lr:
    lr = lambda f: f * args.learning_rate


mapsize = 16 * 16

action_space_shape = (mapsize, envsT.action_plane_space.shape[0])
invalid_action_shape = (mapsize, envsT.action_plane_space.nvec.sum() + 1)


obs = torch.zeros((args.num_steps, args.num_envs) + envsT.single_observation_space.shape).to(device)
actions = torch.zeros((args.num_steps, args.num_envs) + action_space_shape).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)

#rewards_dense = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards_attack = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards_winloss = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards_score = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps, args.num_envs)).to(device)
invalid_action_masks = torch.zeros((args.num_steps, args.num_envs) + invalid_action_shape).to(device)

global_step = 0
start_time = time.time()
ob, mas, res = envsT.reset()

next_obs = torch.Tensor(ob).to(device)
next_done = torch.zeros(args.num_envs).to(device)
num_updates = args.total_timesteps // args.batch_size
ScFeatures = torch.zeros((args.num_steps, args.num_envs, 11)).to(device)
zFeatures = torch.zeros((args.num_steps, args.num_envs,8), dtype=torch.long).to(device)


starting_update = 1
from jpype.types import JArray, JInt
import time


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
                zFeatures[step][i] = agent.z_encoder(obs[step][i].view(-1))

        # neu
        # envsT.render("human")


        global_step += 1 * args.num_envs

        obs[step] = next_obs

        ScFeatures[step] = getScalarFeatures(next_obs, res, args.num_envs)
        dones[step] = next_done

        with torch.no_grad():

            values[step] = agent.get_value(obs[step], ScFeatures[step], zFeatures[step]).flatten() # critic(forward(...))
            action, logproba, _, invalid_action_masks[step] = agent.get_action(obs[step], ScFeatures[step], zFeatures[step], envs=envsT) # gesamplete action (aus Verteilung der Logits) (24, 256, 7), actor(forward(...)), invalid_action_masks




        actions[step] = action # (Shape: (step, num_envs, 256 (16 * 16), action) (step, 24, 256, 7))
        # print("actions shape per step:", actions[step].shape)

        logprobs[step] = logproba

        # Die Grid-Position zu jedem Action hinzugefügt (24, 256, 8)
        real_action = torch.cat([
            torch.stack(
                [torch.arange(0, mapsize).to(device) for i in range(envsT.num_envs)
                 ]).unsqueeze(2), action], 2)
        # print("real_action shape:", real_action.shape)
        # print("Grid-Position:", [real_action[0][i][0].item() for i in range(10)]) # -> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


        real_action = real_action.cpu().numpy()

        # =============
        # invalid_action_masks angewandt
        # =============
        valid_actions = real_action[invalid_action_masks[step][:, :, 0].bool().cpu().numpy()]
        valid_actions_counts = invalid_action_masks[step][:, :, 0].sum(1).long().cpu().numpy()


        java_valid_actions = []
        valid_action_idx = 0
        for env_idx, valid_action_count in enumerate(valid_actions_counts):
            java_valid_action = []
            for c in range(valid_action_count):
                java_valid_action += [JArray(JInt)(valid_actions[valid_action_idx])]
                valid_action_idx += 1
            java_valid_actions += [JArray(JArray(JInt))(java_valid_action)]
        java_valid_actions = JArray(JArray(JArray(JInt)))(java_valid_actions)
        # =============



        # =============
        # Schritt in der Umgebung mit der in get_action gesampleten Action
        # =============
        try:
            next_obs, denserew, attackrew, winlossrew, scorerew , ds, infos,res = envsT.step(java_valid_actions)
            next_obs = envsT._from_microrts_obs(next_obs)
            next_obs = torch.Tensor(next_obs).to(device)
        
        except Exception as e:
            e.printStackTrace()
            raise

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
        #rewards_dense[step] = torch.Tensor(denserew* densereward).to(device)

        rewards_attack[step] = torch.Tensor(attackrew*attack).to(device)
        rewards_winloss[step] = torch.Tensor(winlossrew * winloss).to(device)
        rewards_score[step] = torch.Tensor(scorerew*scorew).to(device)
        next_done = torch.Tensor(ds).to(device)
     

        # =============
        # Logging PPO training
        # ============= 
        for info in infos:

            if 'episode' in info.keys():


                game_length = info['episode']['l']
                winloss = winloss* (-0.00013*game_length+1.16) # ca. 0.9 bei 2000 und 1.1 bei 500
                print(f"global_step={global_step}, episode_reward={info['microrts_stats']['RAIWinLossRewardFunction'] * winloss + info['microrts_stats']['AttackRewardFunction'] * attack}")
                writer.add_scalar("charts/old_episode_reward", info['episode']['r'], global_step)
                writer.add_scalar("charts/Game_length", game_length,global_step)
                writer.add_scalar("charts/Episode_reward",info['microrts_stats']['RAIWinLossRewardFunction'] * winloss + info['microrts_stats']['AttackRewardFunction'] * attack, global_step)
                writer.add_scalar("charts/AttackReward", info['microrts_stats']['AttackRewardFunction'] * attack, global_step)
                writer.add_scalar("charts/WinLossRewardFunction", info['microrts_stats']['RAIWinLossRewardFunction']* winloss, global_step)


                break
        # =============
    # =========================

# =========================
# PPO update
# =========================

    with torch.no_grad():

        last_value = agent.get_value(next_obs.to(device),ScFeatures[step],zFeatures[step]).reshape(1, -1)

        # =============
        # GAE
        # ============= 


        advantages = torch.zeros_like(rewards_winloss).to(device)
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                # zuerst nonterminal, nextvalues aus dem Schritt im Setup benutzen (anders repräsentiert)
                nextnonterminal = 1.0 - next_done
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - dones[t + 1] # für jede Umgebung: 1 -> nicht done in step t+1, 0 -> done in step t+1
                nextvalues = values[t + 1]

            # TD-Error = R_(t+1) + γ * V(S_(t+1)) - V(S_t) per environment
            # V(S_t): Value of the state reached in the rollout after Action in step t-1
            # nextvalues: Critic-approximated values per environment in step t
            # for the next step in the rollout (if not terminated)
            delta =  rewards_winloss[t] + rewards_attack[t] + args.gamma * nextvalues * nextnonterminal - values[t] #rewards_dense[t] + + rewards_dense[t] +rewards_score[t]
            # A_t="TD-Error" + γ * λ * A_(t-1)
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values
        # =============

    # flatten the batch
    b_z = zFeatures.reshape(-1,8)                                       # args.num_steps, args.num_envs Dimensionen vereinigen (shape (steps*envs, 8)) (zFeatures für jeden Step, Environment sortiert Step, dann nach Environments)
    b_Sc = ScFeatures.reshape(-1, 11)                                   # args.num_steps, args.num_envs Dimensionen vereinigen (shape (steps*envs, 11)) (ScFeatures für jeden Step, Environment sortiert Step, dann nach Environments)
    b_obs = obs.reshape((-1,) + envsT.single_observation_space.shape)   # dasselbe mit obs          (shape (steps*envs, 16, 16, 73))
    b_actions = actions.reshape((-1,) + action_space_shape)             # dasselbe mit actions      (shape (steps*envs, 256, 7))
    b_logprobs = logprobs.reshape(-1)                                   # dasselbe mit logprobs     (shape (steps*envs,))
    b_advantages = advantages.reshape(-1)                               # dasselbe mit advantages   (shape (steps*envs,))
    b_returns = returns.reshape(-1)                                     # dasselbe mit returns      (shape (steps*envs,))
    b_values = values.reshape(-1)                                       # dasselbe mit values       (shape (steps*envs,))
    b_invalid_action_masks = invalid_action_masks.reshape((-1,) + invalid_action_shape) # dasselbe mit invalid_action_masks (shape (steps*envs, 256, 79))


    # Optimizing policy and value network with minibatch updates
    # --num_minibatches, --update-epochs
    # minibatches_size = int(args.batch_size // args.num_minibatches)
    # mb_inds: indices from the batch
    inds = np.arange(args.batch_size, )

    # Go (update_epochs times) through all mini-batches
    for i_epoch_pi in range(args.update_epochs):
        np.random.shuffle(inds)

        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            minibatch_ind = inds[start:end]
            mb_advantages = b_advantages[minibatch_ind]

            # normalize the advantages
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # forward pass: get network output for the minibatch
            # We also provide actions here
            new_values = agent.get_value(b_obs[minibatch_ind],b_Sc[minibatch_ind],b_z[minibatch_ind]).view(-1)
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

            # Policy loss L^CLIP (θ)=E ̂_t ["min" (r_t (θ)*Â_t,"clip" (r_t (θ),1-ϵ,1+ϵ)*Â_t )]
            # --clip_coef
            # it is the same, because in loss1 and 2 there is a minus sign and max is used at the end
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            entropy_loss = entropy.mean()

            # Value loss Clipping
            # --clip_vloss
            # MSE(approximierte Values, returns) with or without clip()
            if args.clip_vloss:
                v_loss_unclipped = ((new_values - b_returns[minibatch_ind]) ** 2)
                v_clipped = b_values[minibatch_ind] + torch.clamp(new_values - b_values[minibatch_ind], -args.clip_coef,
                                                                  args.clip_coef)
                v_loss_clipped = (v_clipped - b_returns[minibatch_ind]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((new_values - b_returns[minibatch_ind]) ** 2)


            # KL Divergence Loss
            with torch.no_grad():
                _, sl_logprobs, _, _ = supervised_agent.get_action(
                    b_obs[minibatch_ind],
                    b_Sc[minibatch_ind],
                    b_z[minibatch_ind],
                    b_actions.long()[minibatch_ind],
                    b_invalid_action_masks[minibatch_ind],
                    envsT
                )
            kl_div = F.kl_div(newlogproba, sl_logprobs, log_target=True, reduction="batchmean")
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