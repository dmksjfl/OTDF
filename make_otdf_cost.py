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
import h5py

import algo.utils as utils
from envs.env_utils import call_terminal_func
from envs.common import call_env

import ott
import d4rl
import scipy as sp

import ot
import jax.numpy as jnp
import numpy as np
import jax

def solve_ot(
    src_data, tar_data, cost_type='cosine'
):
    src_B = src_data.shape[0]
    tgt_B = tar_data.shape[0]

    src_embs = jnp.array(src_data.reshape(src_B, -1), dtype=jnp.float16)  # (batch_size1 + batch_size2, dim)
    tgt_embs = jnp.array(tar_data.reshape(tgt_B, -1), dtype=jnp.float16)  # (batch_size1 + batch_size2, dim)

    if cost_type == 'euclidean':
        cost_fn = ott.geometry.costs.Euclidean()
    elif cost_type == 'cosine':
        cost_fn = ott.geometry.costs.Cosine()
    else:
        raise NotImplementedError

    scale_cost = 'max_cost'
    geom = ott.geometry.pointcloud.PointCloud(src_embs, tgt_embs, cost_fn=cost_fn, scale_cost=scale_cost)

    solver = ott.solvers.linear.sinkhorn.Sinkhorn(threshold=1e-9, max_iterations=100)
    prob = ott.problems.linear.linear_problem.LinearProblem(geom)
    sinkhorn_output = solver(prob)
    
    coupling_matrix = geom.transport_from_potentials(
        sinkhorn_output.f, sinkhorn_output.g
    )
    cost_matrix = cost_fn.all_pairs(src_embs, tgt_embs)
    ot_costs = jnp.einsum('ij,ij->i', coupling_matrix, cost_matrix)

    return -ot_costs



def filter_dataset(src_replay_buffer, tar_replay_buffer, cost_type='cosine'):
    src_num = src_replay_buffer.state.shape[0]
    srcdata = np.hstack([src_replay_buffer.state, src_replay_buffer.action, src_replay_buffer.next_state])

    tar_num = tar_replay_buffer.state.shape[0]
    tardata = np.hstack([tar_replay_buffer.state, tar_replay_buffer.action, tar_replay_buffer.next_state])

    cost_result = []

    batch_solve = jax.jit(solve_ot)

    iter_time = src_num // 10000 + 1

    for i in range(iter_time):
        current_time = time.time()
        Gs = batch_solve(srcdata[10000*i:10000*(i+1)], tardata)

        part_res = jax.device_get(Gs)
        part_res = part_res.tolist()

        cost_result = cost_result + part_res

        print('Have completed {} transitions'.format(10000*(i+1)))
    
    cost_result = np.array(cost_result)

    return cost_result




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./costlogs")
    parser.add_argument("--policy", default="OTDF", help='policy to use, support OTDF')
    parser.add_argument("--env", default="halfcheetah")
    parser.add_argument("--seed", default=0, type=int)            
    parser.add_argument("--save-model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--metric", default='cosine', type=str)     # metric used in optimal transport
    parser.add_argument('--srctype', default='medium', type=str)
    parser.add_argument("--tartype", default='medium', type=str)
    parser.add_argument("--steps", default=1e6, type=int)
    
    args = parser.parse_args()

    with open(f"{str(Path(__file__).parent.absolute())}/config/{args.policy.lower()}/{args.env.replace('-', '_')}.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("------------------------------------------------------------")
    print("Policy: {}, Env: {}, Seed: {}".format(args.policy, args.env, args.seed))
    print("------------------------------------------------------------")
    
    outdir = args.dir + '/' + args.env + '-srcdatatype-' + args.srctype + '-tardatatype-' + args.tartype

    if not os.path.exists(args.dir):
        os.makedirs(args.dir)
    
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

    config.update({
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': max_action,
    })


    src_replay_buffer = utils.OTReplayBuffer(state_dim, action_dim, device)
    tar_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)

    # load offline datasets
    src_dataset = d4rl.qlearning_dataset(src_env)
    tar_dataset = utils.call_tar_dataset(args.env, args.tartype)

    src_replay_buffer.convert_D4RL(src_dataset)
    tar_replay_buffer.convert_D4RL(tar_dataset)

    cost = filter_dataset(src_replay_buffer, tar_replay_buffer, config['metric'])

    print('done')
    replay_dataset = dict(
        cost           =   cost,
    )

    with h5py.File(outdir +  ".hdf5", 'w') as hfile:

        for k in replay_dataset:
            hfile.create_dataset(k, data=replay_dataset[k], compression='gzip')
