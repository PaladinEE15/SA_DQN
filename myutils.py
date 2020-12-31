import sys
sys.path.append("./common")
sys.path.append("./auto_LiRPA")
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from argparser import argparser
import numpy as np
from read_config import load_config_direct
from attacks import attack
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch, make_atari_cart
from models import QNetwork, model_setup
import torch.optim as optim
import torch
import torch.autograd as autograd
import time
import os
import argparse
import random
from datetime import datetime
from utils import Logger, get_acrobot_eps, test_plot 
from async_env import AsyncEnv
from train import get_logits_lower_bound

UINTS=[np.uint8, np.uint16, np.uint32, np.uint64]
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


def test_performance(target_model_config, config_path, is_attack=False, episode_num=5, max_frames=10000, compare_action=False, log_path=None, extra_model_config=None):
    config = load_config_direct(config_path)
    training_config = config['training_config']
    test_config = config['test_config']
    attack_config = test_config["attack_config"]
    env_params = training_config['env_params']
    env_params['clip_rewards'] = False
    env_params['episode_life'] = False    
    env_name = config['env_id']
    if "NoFrameskip" not in env_name:
        env = make_atari_cart(env_name)
    else:
        env = make_atari(env_name)
        env = wrap_deepmind(env, **env_params)
        env = wrap_pytorch(env)
    
    if log_path is None:
        log_path=os.path.join('logs', env_name+'prefix')

    logger = Logger(open(log_path, "w"))
    state = env.reset()
    dtype = state.dtype
    logger.log("env_shape: {}, num of actions: {}".format(env.observation_space.shape, env.action_space.n))

    dueling = target_model_config.get('dueling', True)
    model = model_setup(env_name, env, False, logger, True, dueling, 1)
    model_path = target_model_config['model_path']
    logger.log('model loaded from ' + model_path)
    model.features.load_state_dict(torch.load(model_path))
    if compare_action:
        dueling = extra_model_config.get('dueling', True)
        extra_model = model_setup(env_name, env, False, logger, True, dueling, 1)
        model_path = extra_model_config['model_path']
        logger.log('additional model loaded from ' + model_path)
        extra_model.features.load_state_dict(torch.load(model_path))      

    seed = random.randint(0, sys.maxsize)
    logger.log('reseting env with seed', seed)
    env.seed(seed)
    state = env.reset()      
    if dtype in UINTS:
        state_max = 1.0
        state_min = 0.0
    else:
        state_max = float('inf')
        state_min = float('-inf')
    reward_matrix = np.zeros(episode_num)
    rob_pst = np.zeros(episode_num)
    raw_pst = np.zeros(episode_num)
    for episode_idx in range(episode_num):
        episode_reward = 0
        same_rob_cnt = 0
        same_raw_cnt = 0
        for frame_idx in range(max_frames):
            state_tensor = torch.from_numpy(np.ascontiguousarray(state)).unsqueeze(0).cuda().to(torch.float32)
            if dtype in UINTS:
                state_tensor /= 255
            
            if compare_action: #compare the actions of two agents
                corrupted_state = attack(model, state_tensor, attack_config)
                action = model.act(state_tensor)[0]
                action_rob = model.act(corrupted_state)[0]
                if action_rob == action:
                    same_rob_cnt+=1
                action_raw = extra_model.act(state_tensor)[0]
                if action_raw == action:
                    same_raw_cnt+=1
            elif is_attack: #single model but also attack
                corrupted_state = attack(model, state_tensor, attack_config)
                action = model.act(corrupted_state)[0]
            else: #raw version
                action = model.act(state_tensor)[0]

            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward 

            if done:
                logger.log('done with reward {} and steps {}'.format(episode_reward, frame_idx))
                state = env.reset()
                reward_matrix[episode_idx] = episode_reward
                rob_pst[episode_idx] = same_rob_cnt/(frame_idx+1)
                raw_pst[episode_idx] = same_raw_cnt/(frame_idx+1)
                break
    
    logger.log('reward mean {} and std {}'.format(reward_matrix.mean(), reward_matrix.std()))
    if compare_action:
        logger.log('rob_pst mean {} and std {}'.format(rob_pst.mean(), rob_pst.std()))
        logger.log('raw_pst mean {} and std {}'.format(raw_pst.mean(), raw_pst.std()))

    return

                