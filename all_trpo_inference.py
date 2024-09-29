from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from chainerrl import policies
from chainerrl import misc
from chainerrl import links
from chainerrl import experiments
from chainerrl.agents import PPO
from chainerrl.agents import a3c
import chainerrl
import os
import norms
import ppo_adversary
from trpo_adversary import TRPO_Adversary
from builtins import *  # NOQA
from future import standard_library

standard_library.install_aliases()  # NOQA
import argparse
import chainer
from chainer import functions as F
from chainer import links as L
import gym
import gym.wrappers
from gym import spaces 
import os
import re  # 添加这一行来导入re模块
import numpy as np
import cupy as cp
from operator import add
import copy
import os
import matplotlib.pyplot as plt
import datetime
import matplotlib.colors as mcolors
filename =''
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
# os.environ['DISPLAY'] = ':0'
class A3CFFGaussian(chainer.Chain, a3c.A3CModel):
    """An example of A3C feedforward Gaussian policy."""

    def __init__(self, obs_size, action_space,
                 n_hidden_layers=3, n_hidden_channels=64,
                 bound_mean=True):
        assert bound_mean in [False, True]
        super().__init__()
        hidden_sizes = (n_hidden_channels,) * n_hidden_layers
        # hidden_sizes = (128, 64) Run4
        with self.init_scope():
            self.pi = policies.FCGaussianPolicyWithStateIndependentCovariance(
                obs_size, action_space.low.size,
                n_hidden_layers, n_hidden_channels,
                var_type='diagonal', nonlinearity=F.tanh,
                bound_mean=bound_mean,
                min_action=action_space.low, max_action=action_space.high,
                mean_wscale=1e-2)
            self.v = links.MLP(obs_size, 1, hidden_sizes=hidden_sizes)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


def main(rollout,env_id,LOAD,Budget):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--env_id', type=str, default='Hopper')
    parser.add_argument('--seed', type=int, default=0,  # default 0
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--n_episodes', type=int, default=20)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--num-envs', type=int, default=1,
                        help='Number of envs run in parallel.')
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--rollout', type=str, default='Nominal',
                        choices=('Nominal', 'Random', 'MAS', 'LAS'))
    parser.add_argument('--start_atk', type=int, default='1')
    parser.add_argument('--clip', type=bool, default=True,
                        help='If set to False, actions will be projected '
                             'based on unclipped nominal action ')
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1)#3
    parser.add_argument('--save_experiments', type=bool, default=False)
    parser.add_argument('--horizon', type=int, default=3)#10
    parser.add_argument('--budget', type=float, default=0)#1
    parser.add_argument('--s', type=str, default='l2')
    parser.add_argument('--t', type=str, default='l2')
    parser.add_argument('--simple_txt', type=str,required=True,default='HP_simple_record_Dy.txt')
    parser.add_argument('--interval', type=int,required=False,default=10)
    parser.add_argument('--prob', type=float,required=False,default=0.4)
    parser.add_argument('--switch_type', type=str,required=True,default='period')
    args = parser.parse_args()
    
    args.env_id=env_id
    global filename
    filename = args.simple_txt
    if args.env_id == 'LL':  # 0.3-0.5
        env_name = 'LunarLanderContinuous-v2'
        load = 'PPOLunarLanderContinuous-v2_Run1'
        load = 'results/20230324T131211.577075/2000000_finish'

        # load = 'results/20230406T160615.993315/2000000_finish'
        # load = 'results/20230407T020235.078803/best'
        # load = 'results/20230406T231609.203255/2000000_finish'

        # load = 'results/20230419T120108.184419/2000000_finish'
        # load = 'results/20230419T120108.184419/best'
        # load = 'results/20230419T155355.704430/best'
        # load = 'results/20230419T155345.845003/2000000_finish'

        # load = 'results/20230420T163440.323189/2000000_finish'
        # load = 'results/20230421T173132.491072/best'
        # load = 'results/20230421T170348.347539/best'
        # load = 'results/20230423T032647.822895/2000000_finish'
        # load = 'results/20230423T030731.305553/2000000_finish'
        # load = 'results/20230422T234029.951891/2000000_finish'
        # load = 'results/20230422T234023.265669/2000000_finish'
        # load = 'results/20230422T232740.217900/2000000_finish'
        #
        # load = 'results/20230424T213243.112830/2000000_finish'
        # load = 'results/20230425T000833.571764/2000000_finish'
        # load = 'results/20230426T064117.924307/2000000_finish'
        # load = 'results/20230425T220527.596357/2000000_finish'#
        # load = 'results/20230425T220538.423991/2000000_finish'
        # load = 'results/20230426T005409.827797/2000000_finish'# robust
        # load = 'results/20230426T005652.993513/2000000_finish'
        # load = 'results/20230426T034656.600460/2000000_finish'
        # load = 'results/20230426T035115.439220/2000000_finish'
        # load = 'results/20230426T062932.089476/2000000_finish'
        # load = 'results/20230426T064117.924307/2000000_finish'
        # load = 'results/20230503T161356.833998/2000000_finish'
        # load = 'results/20230503T190708.762944/2000000_finish'
        # load = 'results/20230503T224554.561263/2000000_finish'#
        # load = 'results/20230504T102246.035676/2000000_finish'
        # load = 'results/20230504T140406.421514/2000000_finish'
        # load = 'results/20230504T141546.732022/2000000_finish'
        # load = 'results/20230504T181447.288518/2000000_finish'#
        load = 'results/20230619T210634.799766/2000000_finish'
        load = 'results/20230620T005217.965164/2000000_finish'
        load = 'results/20230620T043532.704355/2000000_finish'
        load = 'results/20230619T134617.169313/2000000_finish'
        load = 'results/20230625T135404.836810/2000000_finish'
        load = 'results/20230625T135409.125264/2000000_finish'
        load = 'results/20230625T171843.504968/2000000_finish'
        load = 'results/20230625T173119.542761/2000000_finish'
        load = 'results/20230625T205545.492106/2000000_finish'
        load = 'results/20230625T211115.077022/2000000_finish'#
        # load = 'results/20230625T212825.213955/2000000_finish'
        # load = 'results/20230626T002338.120135/2000000_finish'
        # load = 'results/20230626T003214.701659/2000000_finish'
        # load = 'results/20230626T004454.082465/2000000_finish'
        # load = 'results/20230626T010746.275912/2000000_finish'
        # load = 'results/20230626T035820.651890/2000000_finish'
        # load = 'results/20230626T040236.582451/2000000_finish'
        # load = 'results/20230626T041612.550966/2000000_finish'
        # load =  'results/20230626T044455.477058/2000000_finish'
        load ='results/20230703T114335.972471/2000000_finish'
        load = 'results/20230703T114335.972471/2000000_finish'
        # load = 'results/20230705T142224.821499/2000000_finish'
        load = 'results/20230706T162009.844292/2000000_finish'
        load = 'results/20230706T102554.438535/2000000_finish'
        load = 'results/20230705T142224.821499/2000000_finish'
        n_hidden_channels = 128
        n_hidden_layers = 3
        args.seed = 0

    elif args.env_id == 'BW':
        env_name = 'BipedalWalker-v3'
        load = 'results/20230408T215341.630946/2000000_finish'
        # load = 'results/20230409T050240.733872/2000000_finish'
        # load = 'results/20230409T075637.754980/2000000_finish'
        # load = 'results/20230409T014659.094667/best'

        load = 'results/20230408T215336.090566/2000000_finish'

        # load = 'results/20230420T204740.826277/2000000_finish'
        # load = 'results/20230421T013349.702554/2000000_finish'
        # load = 'results/20230421T065624.692024/2000000_finish'
        load = 'results/20230422T012439.435227/2000000_finish'
        # load = 'results/20230421T180248.380276/2000000_finish'
        # load = 'results/20230421T065624.692024/2000000_finish'
        # load = 'results/20230421T111749.349082/best'
        load = 'results/20230503T194739.272686/2000000_finish'
        load = 'results/20230503T194759.395007/2000000_finish'
        load = 'results/20230504T050516.573907/2000000_finish'
        load = 'results/20230504T210413.739779/2000000_finish'
        load = 'results/20230624T155300.452827/2000000_finish'
        load = 'results/20230624T182349.794974/2000000_finish'

        # load = 'results/20230624T205827.781912/2000000_finish'
        load = 'results/20230704T162231.402309/2000000_finish'
        load = 'results/20230705T004421.894623/2000000_finish'
        load = 'results/20230704T201932.653225/2000000_finish'
        load = 'results/20230917T023305.142518/2000000_finish' #train_robust_ppo.py --env BipedalWalker-v3 --attack_budget 1.5 --attack_counter 50 --constrained_distance 0.03 --seed 0
        
        # load="results/20231020T194123.677053/20000_finish"#/home/admin641/xc/SpationAttack_chainer/train_ppo.py
        # load="results/20231020T184117.027081/20000_finish"#/home/admin641/xc/SpationAttack_chainer/train_robust_ppo.py
        # load="results/20231022T185205.355612/2000000_finish"#train_robust_ppo_multi --env BipedalWalker-v3 --attack_budget 1.5
        load="results/20231031T175404.459005ppo2000000/2000000_finish"
        load="results/20231112T110207.977456/2000000_finish"#worst11.13
        load="results/20231113T144501.290829ZGF0.01_1/2000000_finish"#ZGFrobust_adv_ppo,和上面相比只少了worst
        load="results/20231113T151814.738492worst0.01_1/2000000_finish"#worst11.13晚
        load="results/20231114T102637.058703WORST0.01/2000000_finish"#11.15
        load="results/20231114T161528.053780worst0.01/2000000_finish"
        load="results/20231113T144501.290829ZGF0.01_1/2000000_finish"
        # load="results/20231114T175614.629170ZGF0.02/2000000_finish"
        # load="results/20231114T232727.292944zgf0.02/2000000_finish"
        # load="results/20231114T232814.712947WORST0.02/2000000_finish"
        # load="results/20231115T043614.600683worst0.02/2000000_finish"
        # load="results/20231115T054949.441033ZGF0.03/2000000_finish"
        # load="results/20231115T113601.378952zgf0.03/2000000_finish"
        # load="results/20231115T105825.871039WORST0.03/2000000_finish"
        # load="results/20231115T165950.762717worst0.03/2000000_finish"
        # load='/home/admin641/xc/SpationAttack_chainerMulti/results/20231125T232809.225558robust_adv_ppo_ZGF/2000000_finish'
        # load='results/20231126T205737.835113/2000000_finish'
        # load='results/20231126T202053.354882/2000000_finish'
        # load='results/20231126T205737.835113/2000000_finish'
        n_hidden_layers = 3
        # print(load)
        args.seed = 0


    elif args.env_id == 'Hopper':
        env_name = 'Hopper-v2'

        load = 'results/20230626T142819.205989/2000000_finish'
        load = 'results/20230626T174738.574281/2000000_finish'
        load = 'results/20230626T142819.205989/2000000_finish'
        load = 'results/20230626T111127.445556/2000000_finish'
        load = 'results/20230626T111123.103041/2000000_finish'
        load = 'results/20230703T114324.758687/2000000_finish'
        load = 'results/20230706T131848.421433/2000000_finish'
        # load = 'results/20230706T104800.274396/2000000_finish'
        load = 'results/20230703T114324.758687/2000000_finish'
        
        args.seed = 0

        n_hidden_channels = 64
        n_hidden_layers = 3

    elif args.env_id == 'Walker':  # 0.5
        env_name = 'Walker2d-v2'
        load = 'walker2D/3204/1600000_finish'
        load = 'results/20230327T155454.926663/2000000_finish'
        # load = 'runs/PPO/PPO_Walker2d_v2_Run1'

        # load = 'results/20230406T160654.973561/2000000_finish'
        load = 'results/20230407T015451.888336/best'  #
        # load = 'results/20230407T015451.888336/2000000_finish'

        # load = 'results/20230406T230733.578074/2000000_finish'
        # load = 'results/20230420T204829.053429/2000000_finish'
        # load = 'results/20230421T134307.609052/2000000_finish'
        # load = 'results/20230421T134307.609052/best'
        # load = 'results/20230421T225746.197957/best'
        # load = 'results/20230421T225746.197957/best'
        # load = 'results/20230421T171108.249941/2000000_finish'
        # load = 'results/20230421T170908.735048/2000000_finish'
        # load = 'results/20230423T144932.206148/2000000_finish'
        load = 'results/20230423T144943.480899/2000000_finish'
        load = 'results/20230423T144932.206148/2000000_finish'
        load = 'results/20230619T015959.734657/2000000_finish'
        load = 'results/20230619T211503.452359/2000000_finish'
        load = 'results/20230620T010031.442318/2000000_finish'
        load = 'results/20230624T215002.820686/2000000_finish'
        load = 'results/20230703T114331.718954/2000000_finish'
        # load = 'results/20230704T201924.317719/2000000_finish'
        load = 'results/20230707T040003.658450/2000000_finish'
        load='results/20231128T220643.952068copyenv/2000000_finish'
        load='results/20231128T220545.950494robustppozgf/2000000_finish'
        # load='results/20231129T122338.764756copyenv2/2000000_finish'
        # args.seed=8
        n_hidden_channels = 64
        n_hidden_layers = 3

    elif args.env_id == 'HalfCheetah':
        env_name = 'HalfCheetah-v2'
        load = 'runs/PPO/HalfCheetah-v2/PPO_HalfCheetah_v2_Run1'
        load = 'results/20230331T203623.394706/2000000_finish'
        load = 'results/20230407T171705.356527/2000000_finish'
        load = 'results/20230409T131735.177810/2000000_finish'
        # load = 'halfcheetah/2798/2500000_finish'

        # load = 'results/20230407T020704.598370/best'
        # load = 'results/20230406T160650.022690/2000000_finish'
        # load = 'results/20230407T133038.266239/2000000_finish'
        # load = 'results/20230407T160532.927492/2000000_finish'
        # load = 'results/20230407T200458.715073/2000000_finish'
        # load = 'results/20230408T013812.228081/2000000_finish'
        # load = 'results/20230420T162904.327850/2000000_finish'
        # load = 'results/20230421T005702.856692/2000000_finish'
        # load = 'results/20230420T204952.395464/2000000_finish'
        load = 'results/20230422T233956.372344/2000000_finish'
        load = 'results/20230423T032529.971510/2000000_finish'
        # load = 'results/20230421T164626.161159/2000000_finish'
        # load = 'results/20230421T050423.495175/2000000_finish'
        # load = 'results/20230423T070803.993425/2000000_finish'
        load = 'results/20230424T170247.813942/2000000_finish'
        # load = 'results/20230426T064649.627044/2000000_finish'
        load = 'results/20230426T035529.942201/2000000_finish'  # 0.3 26623
        # load = 'results/20230426T005959.835611/2000000_finish'
        # load = 'results/20230425T220521.876656/2000000_finish'
        # load = 'results/20230426T064957.744583/2000000_finish'
        load = 'results/20230503T161354.789922/2000000_finish'
        # load = 'results/20230503T190936.012621/2000000_finish'

        load = 'results/20230503T231314.883907/2000000_finish'
        load = 'results/20230503T231420.511629/2000000_finish'
        load = 'results/20230504T102250.527788/2000000_finish'
        load = 'results/20230504T102332.472804/2000000_finish'
        load = 'results/20230504T183044.183635/2000000_finish'
        load = 'results/20230504T183129.938693/2000000_finish'  #

        load = 'results/20230624T214927.825077/2000000_finish'
        load = 'results/20230625T001932.624435/2000000_finish'
        load = 'results/20230625T024207.544340/2000000_finish'#seems robust
        n_hidden_channels = 64
        n_hidden_layers = 3
        args.seed = 99

    elif args.env_id == 'IP':  # 0.3
        env_name = 'InvertedPendulum-v2'
        load = 'results/20230406T094017.272058/2000000_finish'
        # load = 'results/20230406T160431.910426/2000000_finish'

        load = 'results/20230406T230626.443922/2000000_finish'  # best
        # load = 'results/20230407T015702.916144/2000000_finish'
        # load = 'results/20230421T004932.906046/2000000_finish'
        load = 'results/20230421T171754.738074/2000000_finish'
        load = 'results/20230421T231736.537643/2000000_finish'

        load = 'results/20230503T194719.904283/2000000_finish'
        load = 'results/20230503T194729.708760/2000000_finish'
        load = 'results/20230504T000503.232345/2000000_finish'
        load = 'results/20230504T000611.966240/best'
        load = 'results/20230504T000744.495158/best'
        load = 'results/20230618T232200.563531/2000000_finish'
        load = 'results/20230619T061455.016989/2000000_finish'
        load = 'results/20230620T000442.220720/2000000_finish'
        load = 'results/20230703T114346.436930/2000000_finish'
        load = 'results/20230706T102609.972771/2000000_finish'
    elif args.env_id == 'Swimmer':
        env_name = 'Swimmer-v2'
        load = 'results/20230409T040733.281852/2000000_finish'

        # load = 'results/20230407T193107.052127/2000000_finish'
        load = 'results/20230408T010606.844249/2000000_finish'
    elif args.env_id == 'Humanoid':
        env_name = 'Humanoid-v2'
        load = 'results/20230406T160816.712208/2000000_finish'
        # load = 'results/20230406T160822.234852/2000000_finish'

    else:
        print('No model found')
    
    args.rollout=rollout
    load=LOAD
    args.budget=Budget
    

    def clip_action_filter(a):
        return np.clip(a, action_space.low, action_space.high)

    misc.set_random_seed(args.seed, gpus=(args.gpu,))
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs

    def make_env(env_name):
        env = gym.make(env_name)
        # Use different random seeds for train and test envs
        # process_seed = int(process_seeds[process_idx])
        # env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env.seed(args.seed)
        # Cast observations to float32 because our model uses float32
        env = chainerrl.wrappers.CastObservationToFloat32(env)
        if args.monitor:
            env = gym.wrappers.Monitor(env, args.outdir, force=True)
        if isinstance(env.action_space, spaces.Box):
            misc.env_modifiers.make_action_filtered(env, clip_action_filter)
        if args.render:
            env = chainerrl.wrappers.Render(env)
        return env

    env = make_env(env_name)
    # env_sim1=copy.deepcopy(env)
    # env_sim2=copy.deepcopy(env)
    # dict={'one':env_sim1,'two':env_sim2}
    
    
    spy_env = make_env(env_name)
    timestep_limit = env.spec.max_episode_steps
    obs_space = env.observation_space
    action_space = env.action_space
    

    # Normalize observations based on their empirical mean and variance
    obs_normalizer = chainerrl.links.EmpiricalNormalization(
        obs_space.low.size, clip_threshold=5)
    winit = chainerrl.initializers.Orthogonal(1.)
    winit_last = chainerrl.initializers.Orthogonal(1e-2)
    action_size = action_space.low.size

    # Switch policy types accordingly to action space types

    policy = chainer.Sequential(
        L.Linear(None, 64, initialW=winit),
        F.tanh,
        L.Linear(None, 64, initialW=winit),
        F.tanh,
        L.Linear(None, action_size, initialW=winit_last),
        chainerrl.policies.GaussianHeadWithStateIndependentCovariance(
            action_size=action_size,
            var_type='diagonal',
            var_func=lambda x: F.exp(2 * x),  # Parameterize log std
            var_param_init=0,  # log std = 0 => std = 1
        ),
    )
    vf = chainer.Sequential(
        L.Linear(None, 64, initialW=winit),
        F.tanh,
        L.Linear(None, 64, initialW=winit),
        F.tanh,
        L.Linear(None, 1, initialW=winit),
    )

    # For Mujoco Envs
    #################################################################
    if args.env_id == 'Hopper' or args.env_id == 'Walker' or args.env_id == 'HalfCheetah':

        if args.env_id == 'Hopper':
            policy = chainer.Sequential(
                L.Linear(None, 64, initialW=winit),
                F.tanh,
                L.Linear(None, 64, initialW=winit),
                F.tanh,
                L.Linear(None, action_size, initialW=winit_last),
                chainerrl.policies.GaussianHeadWithStateIndependentCovariance(
                    action_size=action_size,
                    var_type='diagonal',
                    var_func=lambda x: F.exp(2 * x),  # Parameterize log std
                    var_param_init=0,  # log std = 0 => std = 1
                ),
            )
            vf = chainer.Sequential(
                L.Linear(None, 64, initialW=winit),
                F.tanh,
                L.Linear(None, 64, initialW=winit),
                F.tanh,
                L.Linear(None, 1, initialW=winit),
            )


        elif args.env_id == 'Walker' or args.env_id == 'HalfCheetah':
            policy = chainer.Sequential(
                L.Linear(None, 64, initialW=winit),
                F.tanh,
                L.Linear(None, 64, initialW=winit),
                F.tanh,
                L.Linear(None, action_size, initialW=winit_last),
                chainerrl.policies.GaussianHeadWithStateIndependentCovariance(
                    action_size=action_size,
                    var_type='diagonal',
                    var_func=lambda x: F.exp(2 * x),  # Parameterize log std
                    var_param_init=0,  # log std = 0 => std = 1
                ),
            )

            vf = chainer.Sequential(
                L.Linear(None, 128, initialW=winit),
                F.tanh,
                L.Linear(None, 64, initialW=winit),
                F.tanh,
                L.Linear(None, 1, initialW=winit),
            )

    model = chainerrl.links.Branched(policy, vf)

    opt = chainer.optimizers.Adam(alpha=3e-4, eps=1e-5)
    opt.setup(model)

    agent = TRPO_Adversary(model, policy, vf,opt,
                          obs_normalizer=obs_normalizer,
                          gpu=args.gpu, update_interval=2048,
                          minibatch_size=64, epochs=10,
                          clip_eps_vf=None, entropy_coef=0.0,
                          standardize_advantages=True,
                          )

    spy = TRPO_Adversary(model, policy, vf,opt,
                        obs_normalizer=obs_normalizer,
                        gpu=args.gpu, update_interval=2048,
                        minibatch_size=64, epochs=10,
                        clip_eps_vf=None, entropy_coef=0.0,
                        standardize_advantages=True,
                        )
    print(load)
    agent.load(load)
    spy.load(load)
    agent.model= chainerrl.links.Branched(agent.policy, agent.vf)
    spy.model= chainerrl.links.Branched(spy.policy, spy.vf)

    def compute_grad(action, means, std_devs):
        # compute analytical gradient
        coeff = -(action - means) / ((np.power(std_devs, 3) * (np.sqrt(2 * np.pi))))
        power = -(np.power((action - means), 2)) / (2 * np.power(std_devs, 2))
        exp = np.exp(power)
        grad_a = coeff * exp
        return grad_a

    def get_attack_seeds(rounds, switch_type, interval=10, prob=0.1):
        assert switch_type == 'period' or switch_type == 'prob'
        attack_seeds = np.zeros(rounds, dtype=float)
        
        if switch_type == 'period':
            assert interval > 0 and interval <= rounds
            attack_times = np.arange(0, rounds, 2*interval)
            for n in range(len(attack_times)-1):
                attack_seeds[attack_times[n]:attack_times[n]+interval] = 1.0#0为不攻击
                
        elif switch_type == 'prob':
            assert prob >= 0.0 and prob <= 1.0
            switch_times = np.arange(0, rounds, interval)
            cache_num = 1.0  # 初始激活状态

            for n in switch_times:
                if np.random.uniform() <= prob:
                    cache_num = 1 - cache_num  # 切换激活状态
                end_num = min(n + interval, rounds)
                attack_seeds[n:end_num] = cache_num
                
        return attack_seeds

    epsilon = args.epsilon
    lr = args.lr
    budget = args.budget

    if args.rollout == 'Nominal':
        print('Running nominal inference')
        total=0
        for i in range(args.n_episodes):
            done = False
            obs = env.reset()
            R = 0
            t = 0
            while not done and t < timestep_limit:
                # env.render()
                action, action_dist, vs_pred = agent.act_forward(obs)
                obs, r, done, _ = env.step(action)
                if args.clip is True: 
                    action = np.clip(action, -1, 1)
                # print(obs,action)
                R += r
                t += 1
            agent.stop_episode()
            print(R,',')
            total+=R
        # print(total/args.n_episodes)
        return total/args.n_episodes

    elif args.rollout == 'Random':
        print('Running random attacks')
        total=0
        for i in range(args.n_episodes):
            done = False
            obs = env.reset()
            R = 0
            t = 0
            while not done and t < timestep_limit:
                # env.render()
                action, action_dist, vs_pred = agent.act_forward(obs)
                if t % 1 == 0:
                    delta = norms.random_delta(n_dim=action_space.low.size, budget=args.budget)
                    action = action + delta
                obs, r, done, _ = env.step(action)
                R += r
                t += 1
            agent.stop_episode()
            # print('test episode:', i, 'R:', R)
            print(R,',')
            total+=R
        return total/args.n_episodes

    elif args.rollout == 'MAS':
        print('Running MAS')
        total = 0
        # args.n_episodes=40
        for i in range(args.n_episodes):#十条轨迹
            total_policy_distance = 0
            obs = env.reset()
            done = False
            R = 0
            t = 0
            while not done and t < timestep_limit:#进行一条轨迹的采样之前需要判断是否结束或超时
                if t < args.start_atk:
                    # print(t)
                    # env.render()
                    action, action_dist, vs_pred = agent.act_forward(obs)
                    if args.clip is True:
                        action = np.clip(action, action_space.low, action_space.high)
                    obs, r, done, _ = env.step(action)
                    R += r
                    t += 1
                else:
                    # env.render()
                    action, action_dist, vs_pred = spy.act_forward(obs)
                    if args.clip is True:
                        action = np.clip(action, action_space.low, action_space.high)
                    means = []
                    std_devs = []
                    for k in range(len(action_dist.mean.data[0])):
                        means.append(cp.asnumpy(action_dist.mean[0][k].data))
                        var = np.exp(cp.asnumpy(action_dist.ln_var[0][k].data))
                        std_devs.append(np.sqrt(var))

                    grad_a = compute_grad(action, means, std_devs)
                    adv_action = action - (lr * grad_a)

                    grad_a = compute_grad(adv_action, means, std_devs)
                    adv_action_new = adv_action - (lr * grad_a)

                    counter = 0
                    while np.absolute(adv_action - adv_action_new).any() > epsilon and counter < 25:
                        # print('Optimizing')
                        adv_action = adv_action_new
                        grad_a = compute_grad(adv_action, means, std_devs)
                        adv_action_new = adv_action - (lr * grad_a)
                        counter += 1

                    delta = adv_action_new - action
                    if args.s == 'l2':
                        proj_spatial_delta = norms.l2_spatial_project(delta, budget)
                    elif args.s == 'l1':
                        proj_spatial_delta = norms.l1_spatial_project2(delta, budget)

                    proj_action = action + proj_spatial_delta

                    proj_action = np.clip(proj_action, action_space.low, action_space.high)

                    distance = action_dist.prob(cp.array(action)) - action_dist.prob(cp.array(proj_action))#计算两个动作的概率之差
                    # print(action)
                    # print(action_dist)
                    # print(action_dist.prob(cp.array(action)))
                    # print(F.exp(action_dist.log_prob(cp.array(action))))
                    total_policy_distance += distance
                    obs, r, done, _ = env.step(proj_action)#环境执行扰动后的动作
                    R += r
                    t += 1
            total += R
            # print('test episode:', i, 'R:', R)
            print(R,',')
            # print('test episode:', i, 'Policy Distance:', total_policy_distance / t)

            agent.stop_episode()
        # print(total)
        return total/args.n_episodes
    elif args.rollout == 'MAS_Dy':
        print('Running MAS_Dy')
        total = 0
        # args.n_episodes=40
        for i in range(args.n_episodes):#十条轨迹
            total_policy_distance = 0
            obs = env.reset()
            done = False
            R = 0
            t = 0
            attack_seed_list=get_attack_seeds(timestep_limit, args.switch_type, args.interval, args.prob)#0为不攻击
            while not done and t < timestep_limit:#进行一条轨迹的采样之前需要判断是否结束或超时
                if t < args.start_atk or attack_seed_list[t]==0:
                    # print(t)
                    # env.render()
                    action, action_dist, vs_pred = agent.act_forward(obs)
                    if args.clip is True:
                        action = np.clip(action, action_space.low, action_space.high)
                    obs, r, done, _ = env.step(action)
                    R += r
                    t += 1
                else:
                    # env.render()
                    action, action_dist, vs_pred = spy.act_forward(obs)
                    if args.clip is True:
                        action = np.clip(action, action_space.low, action_space.high)
                    means = []
                    std_devs = []
                    for k in range(len(action_dist.mean.data[0])):
                        means.append(cp.asnumpy(action_dist.mean[0][k].data))
                        var = np.exp(cp.asnumpy(action_dist.ln_var[0][k].data))
                        std_devs.append(np.sqrt(var))

                    grad_a = compute_grad(action, means, std_devs)
                    adv_action = action - (lr * grad_a)

                    grad_a = compute_grad(adv_action, means, std_devs)
                    adv_action_new = adv_action - (lr * grad_a)

                    counter = 0
                    while np.absolute(adv_action - adv_action_new).any() > epsilon and counter < 25:
                        # print('Optimizing')
                        adv_action = adv_action_new
                        grad_a = compute_grad(adv_action, means, std_devs)
                        adv_action_new = adv_action - (lr * grad_a)
                        counter += 1

                    delta = adv_action_new - action
                    if args.s == 'l2':
                        proj_spatial_delta = norms.l2_spatial_project(delta, budget)
                    elif args.s == 'l1':
                        proj_spatial_delta = norms.l1_spatial_project2(delta, budget)

                    proj_action = action + proj_spatial_delta

                    proj_action = np.clip(proj_action, action_space.low, action_space.high)

                    distance = action_dist.prob(cp.array(action)) - action_dist.prob(cp.array(proj_action))#计算两个动作的概率之差
                    # print(action)
                    # print(action_dist)
                    # print(action_dist.prob(cp.array(action)))
                    # print(F.exp(action_dist.log_prob(cp.array(action))))
                    total_policy_distance += distance
                    obs, r, done, _ = env.step(proj_action)#环境执行扰动后的动作
                    R += r
                    t += 1
            total += R
            # print('test episode:', i, 'R:', R)
            print(R,',')
            # print('test episode:', i, 'Policy Distance:', total_policy_distance / t)

            agent.stop_episode()
        # print(total)
        return total/args.n_episodes

    elif args.rollout == 'LAS':
        print('Running LAS')
        global_budget = args.budget
        epsilon = args.epsilon
        total_LAS=0
        # args.n_episodes=20
        for i in range(args.n_episodes):
            # print('Episode:', i)
            exp_v = []
            total_R = []#最后的长度是timestep_limit：1600
            states = []
            actions = []
            adv_actions = []

            obs = env.reset()
            done = False
            R = 0
            t = 0

            while not done and t < timestep_limit:
                # -----------virtual planning loop begins here-------------------------#
                v_actions = []
                v_values = []
                v_states = []
                spy_rewards = []
                deltas = []

                spy_done = False
                spy_R = 0
                spy_t = 0
                spy_env.seed(args.seed)
                spy_obs = spy_env.reset()

                # bring virtual environment up to state s_t
                for l in range(t):
                    # spy_env.render()
                    spy_obs, spy_r, spy_done, _ = spy_env.step(adv_actions[l])#在每个时间步下，首先将spy环境依次执行扰动后的动作（标准环境都已经执行过了），达到标准环境现在的状态
                spy_done = False
                # spy_env=copy.deepcopy(env)

                if t % args.horizon == 0:#时间窗口10次进行周期，为0，则重置时间窗口和预算，每个时间步t窗口和预算减一，args.horizon为定值10，horizon将10-1循环
                    horizon = args.horizon
                    global_budget = args.budget

                while not spy_done and spy_t < horizon:#一个时间步最多执行10次迭代，horizon最小1
                    # spy_env.render()

                    # forcing spy to act on real observation
                    if spy_t == 0:
                        action, action_dist, vs_pred = spy.act_forward(obs)#也就是某一个时间步下标准智能体的状态，输出这个状态下的动作和价值：forcing spy to act on real observation
                    else:
                        action, action_dist, vs_pred = spy.act_forward(spy_obs)#第一个spy_obs达到标准环境现在的状态

                    if args.clip is True:
                        action = np.clip(action, -1, 1)

                    means = []
                    std_devs = []
                    dims = len(action_dist.mean.data[0])
                    for j in range(dims):
                        means.append(cp.asnumpy(action_dist.mean[0][j].data))
                        var = np.exp(cp.asnumpy(action_dist.ln_var[0][j].data))
                        std_devs.append(np.sqrt(var))

                    grad_a = compute_grad(action, means, std_devs)
                    adv_action = action - (lr * grad_a)

                    grad_a = compute_grad(adv_action, means, std_devs)
                    adv_action_new = adv_action - (lr * grad_a)

                    counter = 0
                    while np.absolute(adv_action - adv_action_new).any() > epsilon and counter < 25:
                        # print('Optimizing')
                        adv_action = adv_action_new
                        grad_a = compute_grad(adv_action, means, std_devs)
                        adv_action_new = adv_action - (lr * grad_a)
                        counter += 1

                    v_actions.append(list(map(float, action)))
                    delta = adv_action_new - action
                    deltas.append(list(map(float, delta)))

                    spy_obs, spy_r, spy_done, _ = spy_env.step(action)
                    spy_R += spy_r
                    spy_t += 1

                    v_values.append(float(vs_pred.data[0][0]))
                    spy_rewards.append(spy_R)
                    v_states.append(list(map(float, spy_obs)))

                # -----------virtual planning loop ends here-------------------------#
                #以下行数是在[10,1],假定horizon=10
                if args.s == 'l1':#Compute ||δt,k||`p for each element in Aadv
                    delta_norms = [norms.l1_spatial_norm(delta) for delta in deltas]#delta_norms 维度（10，1），deltas维度（10，4）
                elif args.s == 'l2':
                    delta_norms = [norms.l2_spatial_norm(delta) for delta in deltas]#delta_norms[0]= sqrt(|x[0][0]|^2 + |x[0][1]|^2 + ... + |x[0][3]|^2)

                if args.t == 'l1':#Project sequence of ||δt,k||`p in Aadv on to ball of size B to obtain look-ahead sequence of budgets [bt,k, bt,k+1 . . . bt,k+H ]
                    temporal_deltas = norms.l1_time_project2(delta_norms, global_budget)#temporal_deltas维度（10，1）
                elif args.t == 'l2':
                    temporal_deltas = norms.l2_time_project(delta_norms, global_budget)

                spatial_deltas = []
                for a in range(len(temporal_deltas)):
                    if args.s == 'l1':#Project each δt,k in Aadv on to look-ahead sequence of budgets computed in the previous step to get sequence [δ′ t,k δ′ t,k+1 . . . δ′ t,k+H ]
                        spatial_deltas.append(
                            list(map(float, norms.l1_spatial_project2(deltas[a], temporal_deltas[a]))))
                    elif args.s == 'l2':
                        spatial_deltas.append(list(map(float, norms.l2_spatial_project(deltas[a], temporal_deltas[a]))))#spatial_deltas维度（10，4）

                proj_actions = list(map(add, v_actions[0], spatial_deltas[0]))#a0（第一个标准动作）扰动后的动作维度（1，4）
                proj_actions = np.clip(proj_actions, -1, 1)
                # env_sim=copy.deepcopy(env)
                
                obs, r, done, _ = env.step(proj_actions)#env是正常的环境，并且执行第一个动作扰动后的动作，获得下一个状态obs，进行下一个时间步进行交互
                # obs_sim, r, done, _ = env_sim.step(proj_actions)
                # obs_sim1, r, done, _ = dict['one'].step(proj_actions)#env是正常的环境，并且执行第一个动作扰动后的动作，获得下一个状态obs，进行下一个时间步进行交互
                
                R += r#当前轨迹的R=扰动后的奖励累计值
                t += 1
                # env_sim=copy.deepcopy(env)
                total_R.append(R)#total_R列表，存储当前的轨迹到该时间步的累计奖励值，以下差不多
                exp_v.append(float(vs_pred[0][0].data))
                states.append(list(map(float, obs)))#标准环境下每个时间步的状态
                actions.append(list(map(float, (np.clip(v_actions[0], -1, 1)))))#标准环境下每个时间步正常执行的动作
                adv_actions.append(list(map(float, np.clip(proj_actions, -1, 1))))#标准环境下每个时间步执行的扰动动作


                # adaptive reduce budget and planning horiozn here
                horizon = horizon - 1
                if args.t == 'l1':
                    used = temporal_deltas[0]
                    global_budget = norms.reducebudget_l1(used, global_budget)
                elif args.t == 'l2':
                    used = temporal_deltas[0]
                    global_budget = norms.reducebudget_l2(used, global_budget)

            # print('test episode:', i, 'R:', R)
            print(R,',')
            agent.stop_episode()
            total_LAS+=R
        # print('test episode_total:', i, 'R:', total_LAS)
        return total_LAS/args.n_episodes
    
            


if __name__ == '__main__':
    # choices=('Nominal', 'Random', 'MAS', 'LAS')
    choices=('Nominal', 'MAS_Dy','MAS','LAS')
    env_id='Swimmer'

    all_switch_type=['period','prob']#0是阶段攻击
    root_folder = 'SwimmerTRPO/final5/'

# 获取根目录下所有文件夹
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir() ]
    
    # 获取所有可用颜色
    all_colors = list(mcolors.TABLEAU_COLORS.values())

# 初始化一个颜色字典，用于存储每个文件夹的颜色
    folder_colors = {}
    for k in range(1,2):
        rollout=choices[k]
        average_rewards = {}
        for folder in subfolders:
            match = re.match(r'.*/(\d{8}T\d{6}.\d+)WD_(\w+)/best', folder)
            match = re.match(r'.*/(\d+T\d+\.\d+)WD_(\w+)/best', folder)
            match = re.match(r'.*/\d+T\d+\.\d+WD_(\w+)/best', folder)
            pattern = r'[A-Za-z_]+([A-Za-z_0-9]*)$'
            match = re.search(pattern, folder)
            if match:
                folder_type = match.group(0)

                variable_name = f'average_rewards_{folder_type}'
                average_rewards[variable_name] = []


                b_start=0.0
                b_end=2.01
                b_tick=0.1
                budget_range=np.arange(b_start,b_end,b_tick)#预算范围
                for i in budget_range:
                    Budget=i
                    if k==3:
                        Budget=i*3
                    print(Budget)
                # print("*****"*80)
          
                    average_rewards[variable_name].append(main(rollout, env_id, folder+'/best', Budget))
                folder_colors[folder_type] = all_colors[len(folder_colors) % len(all_colors)]
                plt.plot(budget_range, average_rewards[variable_name], marker='d', color=folder_colors[folder_type], label=folder_type)
                now = datetime.datetime.now()
                formatted_time = now.strftime("%-y-%-m-%-d_%H:%M")
                # filename='SwimmerTRPO/final4/test_W_T/simple.txt'
                with open(filename, 'a') as file:
                    for number in average_rewards[variable_name]:
                        # 将每个数字转换为字符串，并追加一个逗号和换行符
                        file.write(f"{number},\n")
                    file.write(root_folder+ formatted_time+" average_rewards_ "+folder_type+rollout+ ' '+env_id+'.png,\n')
            
                        # 添加横轴纵轴标签
        plt.xlabel('Budget')
        plt.ylabel('Rewards')

        # 设置横轴刻度
        plt.xticks(np.arange(b_start,b_end,b_tick))

        # 添加图例
        plt.legend()
        now = datetime.datetime.now()
        formatted_time = now.strftime("%-y-%-m-%-d_%H:%M")
        # 添加图名
        plt.title(formatted_time+ ' in '+rollout+ ' '+env_id)

        # 显示图形
        plt.grid(True)
        plt.savefig(root_folder+ formatted_time+rollout+ ' '+env_id+'.png')
        plt.show()
        plt.close()
        print(root_folder+ formatted_time+""+rollout+ ' '+env_id+'.png')

        


