import sys
sys.path.append("./common")
sys.path.append("./auto_LiRPA")
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from statebuffer import StateBuffer, save_to_pkl, load_from_pkl
from argparser import argparser
import numpy as np
from read_config import load_config_direct
from attacks import attack
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch, make_atari_cart
from models import QNetwork, model_setup
from torch.nn import functional as F
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
from tensorboardX import SummaryWriter
import copy

UINTS=[np.uint8, np.uint16, np.uint32, np.uint64]
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

env_params_set=[{"frame_stack": False, "color_image": False, "central_crop": True,"restrict_actions": True},{"frame_stack": False, "color_image": False, "central_crop": True,"crop_shift": 0},{"frame_stack": False, "color_image": False, "central_crop": True,"crop_shift": 10, "restrict_actions": 4},{"frame_stack": False, "color_image": False, "central_crop": True, "crop_shift": 20, "restrict_actions": True}]
env_name_set = ["BankHeistNoFrameskip-v4","FreewayNoFrameskip-v4","PongNoFrameskip-v4","RoadRunnerNoFrameskip-v4"]

def generate_stb(model, env, save_path, bf_size=100000):
    state = env.reset()
    state_buffer = StateBuffer(d_type=state.dtype,obs_shape=state.shape,buffer_size=bf_size)
    for steps in range(bf_size):
        state_tensor = torch.from_numpy(np.ascontiguousarray(state)).unsqueeze(0).cuda().to(torch.float32)
        state_tensor /= 255
        action = model.act(state_tensor)
        state, reward, done, _ = env.step(action)
        state_buffer.add(state)
        if done:
            state = env.reset()
    save_to_pkl(save_path, state_buffer)
    return state_buffer

def robust_learn(env_id, rob_model, total_steps, train_attack_mag, attack_steps, lr, stb_path, src_path, tgt_path, exist_stb=False, log_name=None, batch_size=32, robust_factor=1):
    env_name = env_name_set[env_id]
    env_params = env_params_set[env_id]
    if "NoFrameskip" not in env_name:
        env = make_atari_cart(env_name)
    else:
        env = make_atari(env_name)
        env = wrap_deepmind(env, **env_params)
        env = wrap_pytorch(env) 
    
    if log_name !=None:
        writer = SummaryWriter(log_name)   

    rob_model = model_setup(env_name, env, False, None, True, True, 1)   
    rob_model.features.load_state_dict(torch.load(src_path))

    if exist_stb:
        state_buffer = load_from_pkl(stb_path)
    else:
        state_buffer = generate_stb(rob_model, env, stb_path)
    
    stable_model = copy.deepcopy(rob_model)
    optimizer = optim.Adam(rob_model.parameters(), lr=lr)
    for steps in range(total_steps):
        raw_observations = state_buffer.sample(batch_size).to(torch.float32)
        raw_observations /= 255
        raw_observations.requires_grad_(True)
        true_q_vals = stable_model(raw_observations)
        action_label = torch.argmax(true_q_vals,dim=1,keepdim=True)
        q_vals = rob_model(raw_observations)
        output_p = F.softmax(q_vals,dim=1)
        target_vals = torch.gather(output_p,1,action_label)
        target_vals.backward(torch.ones_like(target_vals))
        obs_step = torch.sign(raw_observations.grad)*train_attack_mag
        best_adv_obs = torch.clamp(raw_observations-obs_step,max=1,min=0)

        output_adv = rob_model(best_adv_obs)
        output_raw = rob_model(raw_observations)
        criterion = torch.nn.CrossEntropyLoss()
        loss_adv = criterion(output_adv,action_label)
        loss_raw = criterion(output_raw,action_label)
        final_loss = robust_factor*loss_adv + (1-robust_factor)*loss_raw
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()
        if log_name !=None:
            writer.add_scalar('loss_adv', loss_adv.detach().cpu().numpy(), global_step=steps)
            writer.add_scalar('loss_raw', loss_raw.detach().cpu().numpy(), global_step=steps)
            writer.add_scalar('final_loss', final_loss.detach().cpu().numpy(), global_step=steps)
    torch.save(rob_model.features.state_dict(),tgt_path)
    if log_name!=None:
        writer.close()    
    return 



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
    logger.log('model loaded from' + model_path)
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

                