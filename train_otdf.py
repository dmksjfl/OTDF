import numpy as np
import torch
import gym
import argparse
import os
import random
import math
import time
import copy
from pathlib import Path
import yaml


from algo.OTDF import OTDF
import algo.utils as utils
from envs.env_utils import call_terminal_func
from envs.common import call_env
from tensorboardX import SummaryWriter

import d4rl


def eval_policy(policy, env, eval_episodes=10, eval_cnt=None):
    eval_env = env

    avg_reward = 0.
    for episode_idx in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            next_state, reward, done, _ = eval_env.step(action)

            avg_reward += reward
            state = next_state
    avg_reward /= eval_episodes

    print("[{}] Evaluation over {} episodes: {}".format(eval_cnt, eval_episodes, avg_reward))

    return avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./logs")
    parser.add_argument("--policy", default="OTDF", help='policy to use, support OTDF')
    parser.add_argument("--env", default="halfcheetah")
    parser.add_argument("--seed", default=0, type=int)            
    parser.add_argument("--save-model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--metric", default='cosine', type=str)     # metric used in optimal transport
    parser.add_argument('--srctype', default='medium', type=str)
    parser.add_argument("--tartype", default='medium', type=str)
    parser.add_argument("--steps", default=1e6, type=int)
    parser.add_argument("--weight", action="store_true")
    parser.add_argument("--proportion", default=0.8, type=float)
    parser.add_argument("--noreg", action="store_true")
    parser.add_argument("--reg_weight", default=0.5, type=float)
    
    args = parser.parse_args()

    with open(f"{str(Path(__file__).parent.absolute())}/config/{args.policy.lower()}/{args.env.replace('-', '_')}.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("------------------------------------------------------------")
    print("Policy: {}, Env: {}, Seed: {}".format(args.policy, args.env, args.seed))
    print("------------------------------------------------------------")
    
    outdir = args.dir + '/' + args.policy.lower() + '/' + args.env + '-srcdatatype-' + args.srctype + '-tardatatype-' + args.tartype + '/r' + str(args.seed)
    writer = SummaryWriter('{}/tb'.format(outdir))
    if args.save_model and not os.path.exists("{}/models".format(outdir)):
        os.makedirs("{}/models".format(outdir))
    
    if '_' in args.env:
        args.env = args.env.replace('_', '-')
    
    # train env
    src_env_name = args.env.split('-')[0] + '-' + args.srctype + '-v2'
    src_env = gym.make(src_env_name)
    src_env.seed(args.seed)
    # test env
    tar_env = call_env(config['tar_env_config'])
    tar_env.seed(args.seed)
    # eval env
    src_eval_env = copy.deepcopy(src_env)
    src_eval_env.seed(args.seed + 100)
    tar_eval_env = copy.deepcopy(tar_env)
    tar_eval_env.seed(args.seed + 100)

    # seed all
    src_env.action_space.seed(args.seed)
    tar_env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    state_dim = src_env.observation_space.shape[0]
    action_dim = src_env.action_space.shape[0] 
    max_action = float(src_env.action_space.high[0])
    min_action = -max_action
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config['metric'] = args.metric

    weight = True if args.weight else False
    noreg = True if args.noreg else False

    config.update({
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': max_action,
        'weight': weight,
        'proportion': float(args.proportion),
        'noreg': noreg,
        'reg_weight': args.reg_weight,
    })

    if args.policy.lower() == 'otdf':
        policy = OTDF(config, device)
    else:
        raise NotImplementedError
    
    ## write logs to record training parameters
    with open(outdir + 'log.txt','w') as f:
        f.write('\n Policy: {}; Env: {}, seed: {}'.format(args.policy, args.env, args.seed))
        for item in config.items():
            f.write('\n {}'.format(item))

    src_replay_buffer = utils.OTReplayBuffer(state_dim, action_dim, device)
    tar_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)

    # load offline datasets
    src_dataset = d4rl.qlearning_dataset(src_env)
    tar_dataset = utils.call_tar_dataset(args.env, args.tartype)

    src_replay_buffer.convert_D4RL(src_dataset)
    tar_replay_buffer.convert_D4RL(tar_dataset)

    # load optimal transport cost
    src_replay_buffer.cost = utils.call_otdf_cost(args.env, args.srctype, args.tartype)

    eval_cnt = 0

    # whether to pretrain VAE
    if not noreg:
        policy.train_vae(tar_replay_buffer, config['batch_size'], writer)
    
    eval_src_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
    eval_tar_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)
    eval_cnt += 1

    for t in range(int(args.steps)):
        policy.train(src_replay_buffer, tar_replay_buffer, config['batch_size'], writer)

        if (t + 1) % config['eval_freq'] == 0:
            src_eval_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
            tar_eval_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)
            writer.add_scalar('test/source return', src_eval_return, global_step = t+1)
            writer.add_scalar('test/target return', tar_eval_return, global_step = t+1)
            eval_cnt += 1

            if args.save_model:
                policy.save('{}/models/model'.format(outdir))
    writer.close()
