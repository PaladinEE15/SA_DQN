import sys
sys.path.append("./common")
sys.path.append("./auto_LiRPA")
from auto_LiRPA import BoundedModule, BoundedTensor, CrossEntropyWrapper
from auto_LiRPA.perturbations import PerturbationLpNorm
from statebuffer import StateBuffer, save_to_pkl, load_from_pkl
from argparser import argparser
import numpy as np
from read_config import load_config_direct
from attacks import attack,myattack
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
from tensorboardX import SummaryWriter
import copy

UINTS=[np.uint8, np.uint16, np.uint32, np.uint64]
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

env_params_set=[{"frame_stack": False, "color_image": False, "central_crop": True,"restrict_actions": True},{"frame_stack": False, "color_image": False, "central_crop": True,"crop_shift": 0},{"frame_stack": False, "color_image": False, "central_crop": True,"crop_shift": 10, "restrict_actions": 4},{"frame_stack": False, "color_image": False, "central_crop": True, "crop_shift": 20, "restrict_actions": True},{"frame_stack": False, "color_image": False, "central_crop": True,"crop_shift": 0},{"frame_stack": False, "color_image": False, "central_crop": True,"crop_shift": 0},{"frame_stack": False, "color_image": False, "central_crop": True,"crop_shift": 0},{"frame_stack": False, "color_image": False, "central_crop": True,"crop_shift": 0}]
env_name_set = ["BankHeistNoFrameskip-v4","FreewayNoFrameskip-v4","PongNoFrameskip-v4","RoadRunnerNoFrameskip-v4","BreakoutNoFrameskip-v4","IceHockeyNoFrameskip-v4","RobotankNoFrameskip-v4","SpaceInvadersNoFrameskip-v4"]

class BestP(torch.nn.Module):
    def __init__(self,q_net):
        super(BestP,self).__init__()
        self.q_net = q_net

    def forward(self, the_input):
        data = the_input[0]
        info = the_input[1]
        y_out = self.q_net(data)
        y_out_sum = torch.mean(y_out,dim=1,keepdim = True)
        target_val = torch.gather(y_out,1,info)

        p_out = target_val-y_out_sum
        
        return p_out    

def get_exp_module(bounded_module):
    for _, node in bounded_module.named_modules():
        # Find the Exp neuron in computational graph
        if isinstance(node, BoundExp):
            return node
    return None


def get_logits_lower_bound(model, state, eps, C, IBP=False):
    ptb = PerturbationLpNorm(norm=np.inf, eps=eps, x_L=0, x_U=1)
    bnd_state = BoundedTensor(state, ptb)
    #pred = model(bnd_state, method_opt="forward")
    if IBP:
        logits_clb, _ = model.compute_bounds(x=bnd_state, C=C, IBP=True, method=None)
    else:
        logits_clb, _ = model.compute_bounds(x=bnd_state, IBP=False, C=C, method="backward", bound_upper=False)
    return logits_clb

model_src_path = ['models/BankHeist-natural.model','models/Freeway-natural.model','models/Pong-natural.model','models/RoadRunner-natural.model','models/Breakout-natural.pth','models/IceHockey-natural.pth','models/Robotank-natural.pth','models/SpaceInvaders-natural.pth']
model_abbr = ['BH','FW','PO','RR','BO','IH','RT','SI']
model_config_path=['config/BankHeist_nat.json','config/Freeway_nat.json','config/Pong_nat.json','config/RoadRunner_nat.json','config/Breakout_nat.json','config/IceHockey_nat.json','config/Robotank_nat.json','config/SpaceInvaders_nat.json']

def rob_train_and_test(env_id, train_attack_mag=1/255, lr=0.0003, batch_size=32, total_steps=40000, robust_factor_and_steps=[[1,1]],model_id=['1'],is_cov=False):
    for idx, paraset in enumerate(robust_factor_and_steps):
        robust_factor = paraset[0]
        atk_steps = paraset[1]
        print('rob factor:',robust_factor,"atk_steps:",atk_steps)
        start_time = time.time()
        robust_learn(env_id=env_id, total_steps=total_steps, train_attack_mag=train_attack_mag, lr=lr, stb_path='buffers/'+model_abbr[env_id]+'_buffer', src_path=model_src_path[env_id], tgt_path='rob_models/rob_'+model_abbr[env_id]+model_id[idx]+'.pkl', log_name='Logs/'+model_abbr[env_id], batch_size=batch_size, robust_factor=robust_factor,atk_steps=atk_steps,is_cov=is_cov)       
        end_time = time.time()
        print('train time:', end_time-start_time)
        '''
        print('raw performance:')
        test_performance(target_model_config={'model_path':'rob_models/rob_'+model_abbr[env_id]+model_id[idx]+'.pkl'}, config_path=model_config_path[env_id], is_attack=False, episode_num=5, max_frames=10000, compare_action=False, log_path=None)
        print('atk performance:')
        test_performance(target_model_config={'model_path':'rob_models/rob_'+model_abbr[env_id]+model_id[idx]+'.pkl'}, config_path=model_config_path[env_id], is_attack=True, episode_num=5, max_frames=10000, compare_action=False, log_path=None)
        '''


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

def robust_learn_new(env_id, total_steps, train_attack_mag, lr, stb_path, src_path, tgt_path, log_name=None, batch_size=32, robust_factor=1):
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

    if os.path.exists(stb_path+'.pkl'):
        print('use existing buffer')
        state_buffer = load_from_pkl(stb_path)
    else:
        print('generate new buffer')
        state_buffer = generate_stb(rob_model, env, stb_path)
    
    stable_model = copy.deepcopy(rob_model)
    optimizer = optim.Adam(rob_model.parameters(), lr=lr)
    for steps in range(total_steps):
        factor = steps/total_steps
        raw_observations = state_buffer.sample(batch_size).to(torch.float).clone()
        raw_observations /= 255
        raw_observations.requires_grad_(True)
        true_q_vals = stable_model(raw_observations)
        action_label = torch.argmax(true_q_vals,dim=1).clone().detach()
        
        if steps == 0:
            action_label_cpu = action_label.cpu()
            model_loss = BoundedModule(CrossEntropyWrapper(rob_model.features), (raw_observations, action_label_cpu),
                               bound_opts={'relu': "same-slope", 'loss_fusion': True}, device='cuda')
        bnd_state = BoundedTensor(raw_observations, PerturbationLpNorm(norm=np.inf, eps=train_attack_mag))
        
        optimizer.zero_grad()
        _, ub = model_loss(method_opt="compute_bounds", x=[bnd_state,action_label], IBP=True, C=None, final_node_name=None, no_replicas=True)
        exp_module = get_exp_module(model_loss)
        max_input = exp_module.max_input
        robust_loss = torch.mean(torch.log(ub) + max_input)
        robust_loss.backward()
        optimizer.step()
        if steps%100 == 0:
            print('loss:',robust_loss.detach().cpu().numpy())
        if log_name !=None:
            writer.add_scalar('robust_loss', robust_loss.detach().cpu().numpy(), global_step=steps)
    torch.save(rob_model.features.state_dict(),tgt_path)
    if log_name!=None:
        writer.close()    
    return 


def robust_learn(env_id, total_steps, train_attack_mag, lr, stb_path, src_path, tgt_path, is_cov=False, log_name=None, batch_size=32, robust_factor=1, atk_steps = 1):
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

    if os.path.exists(stb_path+'.pkl'):
        print('use existing buffer')
        state_buffer = load_from_pkl(stb_path)
    else:
        print('generate new buffer')
        state_buffer = generate_stb(rob_model, env, stb_path)
    
    stable_model = copy.deepcopy(rob_model)
    optimizer = optim.Adam(rob_model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    for steps in range(total_steps):
        raw_observations = state_buffer.sample(batch_size).to(torch.float).clone()
        raw_observations /= 255
        raw_observations.requires_grad_(True)
        true_q_vals = stable_model(raw_observations)
        action_label = torch.argmax(true_q_vals,dim=1).clone().detach()
        if is_cov:
            if steps == 0:
                rob_model = BoundedModule(rob_model.features,raw_observations,device='cuda')
            c = torch.eye(stable_model.num_actions).type_as(raw_observations)[action_label].unsqueeze(1) - torch.eye(stable_model.num_actions).type_as(raw_observations).unsqueeze(0)
            I = (~(action_label.data.unsqueeze(1) == torch.arange(stable_model.num_actions).type_as(action_label.data).unsqueeze(0)))
            c = (c[I].view(raw_observations.size(0), stable_model.num_actions-1, stable_model.num_actions)).cuda()
            bnd_state = BoundedTensor(raw_observations, PerturbationLpNorm(norm=np.inf, eps=train_attack_mag))
            pred = rob_model(bnd_state)
            lb, _ = rob_model.compute_bounds(IBP=True, C=c, bound_upper=False)
            #lb, _ = rob_model.compute_bounds(IBP=False, method='backward', C=c, bound_upper=False)
            reg_loss, _ = torch.min(lb,dim=1)
            reg_loss = - torch.clamp(reg_loss, max=1)
            reg_loss = reg_loss.mean()
            output_raw = rob_model(raw_observations)
            loss_raw = criterion(output_raw,action_label)
            optimizer.zero_grad()
            total_loss = reg_loss*robust_factor + loss_raw*(1-robust_factor)
            reg_loss.backward()
            optimizer.step()
            if log_name !=None:
                writer.add_scalar('reg_loss', reg_loss.detach().cpu().numpy(), global_step=steps)
        else:
            mid_observations = Variable(raw_observations.data,requires_grad=True)
            step_size = train_attack_mag/atk_steps
            
            for atk_mini_step in range(atk_steps):
                q_vals = rob_model(mid_observations)
                #use maximize tools
                #output_p = F.softmax(q_vals,dim=1)
                #target_vals = torch.gather(output_p,1,action_label)
                rob_model.features.zero_grad()
                losses = criterion(q_vals,action_label)
                losses.backward(torch.ones_like(losses))
                mid_observations = mid_observations + torch.sign(mid_observations.grad)*step_size
                mid_observations = torch.clamp(mid_observations,max=1,min=0)
                mid_observations = Variable(mid_observations.data,requires_grad=True)
            eta = torch.clamp(mid_observations - raw_observations,max=train_attack_mag,min=-train_attack_mag)
            mid_observations = raw_observations + eta

            output_adv = rob_model(mid_observations)
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
    if is_cov:
        torch.save(rob_model.state_dict(),tgt_path)
    else:
        torch.save(rob_model.features.state_dict(),tgt_path)
    if log_name!=None:
        writer.close()    
    return 



def test_performance(target_model_config, config_path, atk_type, atk_sth=1/255, atk_steps=10, is_attack=False, episode_num=5, max_frames=10000, compare_action=False, log_path=None, extra_model_config=None):
    config = load_config_direct(config_path)
    training_config = config['training_config']
    test_config = config['test_config']
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
    raw_pst = np.zeros(episode_num)
    for episode_idx in range(episode_num):
        episode_reward = 0
        same_raw_cnt = 0
        for frame_idx in range(max_frames):
            state_tensor = torch.from_numpy(np.ascontiguousarray(state)).unsqueeze(0).cuda().to(torch.float32)
            if dtype in UINTS:
                state_tensor /= 255
            
            if compare_action: #compare the actions of two agents
                corrupted_state = myattack(model, state_tensor, atk_type, atk_sth, atk_steps)
                action = model.act(corrupted_state)[0]
                action_raw = extra_model.act(state_tensor)[0]
                if action_raw == action:
                    same_raw_cnt+=1
            elif is_attack: #single model but also attack
                corrupted_state = myattack(model, state_tensor, atk_type, atk_sth, atk_steps)
                action = model.act(corrupted_state)[0]
            else: #raw version
                action = model.act(state_tensor)[0]

            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward 

            if done:
                #logger.log('done with reward {} and steps {}'.format(episode_reward, frame_idx))
                state = env.reset()
                reward_matrix[episode_idx] = episode_reward
                raw_pst[episode_idx] = same_raw_cnt/(frame_idx+1)
                break
    
    logger.log('reward mean {} and std {}'.format(reward_matrix.mean(), reward_matrix.std()))
    if compare_action:
        logger.log('raw_pst mean {} and std {}'.format(raw_pst.mean(), raw_pst.std()))
    return

                