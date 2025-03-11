from typing import Dict, List, Tuple
import numpy as np
import gym

from envs.hopper        import HP_Dynamics_Wrapper
from envs.halfcheetah   import HC_Dynamics_Wrapper, HC_Speed_and_Dynamics_Wrapper
from envs.ant           import AT_Dynamics_Wrapper
from envs.walker        import WK_Dynamics_Wrapper

from gym.envs.mujoco.ant_v3            import AntEnv
from gym.envs.mujoco.half_cheetah_v3   import HalfCheetahEnv
from gym.envs.mujoco.walker2d_v3       import Walker2dEnv
from gym.envs.mujoco.hopper_v3         import HopperEnv
from gym.wrappers.time_limit           import TimeLimit


from pathlib import Path
import d4rl


def call_env(env_config: Dict) -> gym.Env:
    if 'Hopper' in env_config['env_name']:
        if env_config['env_name'] == 'Hopper-og':
            return gym.make('Hopper-v2')
        elif env_config['env_name'] == 'Hopper-morph':
            return TimeLimit(
                HopperEnv(xml_file= f"{str(Path(__file__).parent.absolute())}/assets/hopper_morph.xml"),
                1000
            )
        elif env_config['env_name'] == 'Hopper-gravity':
            return TimeLimit(
                HopperEnv(xml_file= f"{str(Path(__file__).parent.absolute())}/assets/hopper_gravity.xml"),
                1000
            )
        else:
            return HP_Dynamics_Wrapper(param_dict=env_config['param'])
    elif 'HalfCheetah' in env_config['env_name']:
        if env_config['env_name'] == 'HalfCheetah-og':
            return gym.make('HalfCheetah-v2')
        elif env_config['env_name'] == 'HalfCheetah-morph':
            return TimeLimit(
                HalfCheetahEnv(xml_file= f"{str(Path(__file__).parent.absolute())}/assets/halfcheetah_morph.xml",),
                1000
            )
        elif env_config['env_name'] == 'HalfCheetah-gravity':
            return TimeLimit(
                HalfCheetahEnv(xml_file= f"{str(Path(__file__).parent.absolute())}/assets/halfcheetah_gravity.xml",),
                1000
            )
        elif env_config['env_name'] == 'HalfCheetah-speed-dyna':
            return HC_Speed_and_Dynamics_Wrapper(param_dict=env_config['param'])
        else:
            return HC_Dynamics_Wrapper(param_dict=env_config['param'])
    elif 'Ant' in env_config['env_name']:
        if env_config['env_name'] == 'Ant-og':
            return gym.make('Ant-v3')
        elif env_config['env_name'] == 'Ant-morph':
            return TimeLimit(
                AntEnv(xml_file= f"{str(Path(__file__).parent.absolute())}/assets/ant_morph.xml",),
                1000
            )
        elif env_config['env_name'] == 'Ant-gravity':
            return TimeLimit(
                AntEnv(xml_file= f"{str(Path(__file__).parent.absolute())}/assets/ant_gravity.xml",),
                1000
            )
        else:
            return AT_Dynamics_Wrapper(param_dict=env_config['param'])
    elif 'Walker' in env_config['env_name']:
        if env_config['env_name'] == 'Walker-og':
            return gym.make('Walker2d-v2')
        elif env_config['env_name'] == 'Walker-morph':
            return TimeLimit(
                Walker2dEnv(xml_file= f"{str(Path(__file__).parent.absolute())}/assets/walker_morph.xml",),
                1000
            )
        elif env_config['env_name'] == 'Walker-gravity':
            return TimeLimit(
                Walker2dEnv(xml_file= f"{str(Path(__file__).parent.absolute())}/assets/walker_gravity.xml",),
                1000
            )
        else:
            return WK_Dynamics_Wrapper(param_dict=env_config['param'])
    else:
        raise NotImplementedError


