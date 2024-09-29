"""A training script of PPO on OpenAI Gym Mujoco environments.

This script follows the settings of https://arxiv.org/abs/1709.06560 as much
as possible.
"""
import functools

import gym.spaces
from copy import deepcopy
from chainerrl import policies, replay_buffer
from chainerrl import misc
from chainerrl import links
# from chainerrl import experiments
from my_experiments import experiments
from agent.robust_adv_ppo_myworsttwostepsR import PPO
from chainerrl.agents import a3c
import chainerrl
from builtins import *  # NOQA
from future import standard_library
import torch.nn as nn
standard_library.install_aliases()  # NOQA
import argparse
import chainer
from chainer import functions as F
from chainer import links as L
import gym
import gym.wrappers
import numpy as np
from torch_utils import *
import torch.optim as optim
from auto_LiRPA import BoundedModule
from auto_LiRPA.eps_scheduler import LinearScheduler
from auto_LiRPA.bounded_tensor import BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
HIDDEN_SIZES = (64, 64)
ACTIVATION = nn.Tanh
STD = 2**0.5
def partialclass(cls, *args, **kwds):

    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)
    return NewCls

ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky": nn.LeakyReLU,
    "leaky0.05": partialclass(nn.LeakyReLU, negative_slope=0.05),
    "leaky0.1": partialclass(nn.LeakyReLU, negative_slope=0.1),
    "hardtanh": nn.Hardtanh,
}

def concat_obs_and_action(obs, action):
    """Concat observation and action to feed the critic."""
    return F.concat((obs, action), axis=-1)

def activation_with_name(name):
    return ACTIVATIONS[name]



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
def initialize_weights(mod, initialization_type, scale=STD):
    '''
    Weight initializer for the models.
    Inputs: A model, Returns: none, initializes the parameters
    '''
    for p in mod.parameters():
        if initialization_type == "normal":
            p.data.normal_(0.01)
        elif initialization_type == "xavier":
            if len(p.data.shape) >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                p.data.zero_()
        elif initialization_type == "orthogonal":
            if len(p.data.shape) >= 2:
                orthogonal_init(p.data, gain=scale)
            else:
                p.data.zero_()
        else:
            raise ValueError("Need a valid initialization key")
        
class ValueDenseNet(nn.Module):
    '''
    An example value network, with support for arbitrarily many
    fully connected hidden layers (by default 2 * 128-neuron layers),
    maps a state of size (state_dim) -> a scalar value.
    '''
    def __init__(self, state_dim, init=None, hidden_sizes=(64, 64), activation=None):
        '''
        Initializes the value network.
        Inputs:
        - state_dim, the input dimension of the network (i.e dimension of state)
        - hidden_sizes, an iterable of integers, each of which represents the size
        of a hidden layer in the neural network.
        Returns: Initialized Value network
        '''
        super().__init__()
        if isinstance(activation, str):
            self.activation = activation_with_name(activation)()
        else:
            # Default to tanh.
            self.activation = ACTIVATION()
        self.affine_layers = nn.ModuleList()

        prev = state_dim
        for h in hidden_sizes:
            l = nn.Linear(prev, h)
            if init is not None:
                initialize_weights(l, init)
            self.affine_layers.append(l)
            prev = h

        self.final = nn.Linear(prev, 1)
        if init is not None:
            initialize_weights(self.final, init, scale=1.0)

    def initialize(self, init="orthogonal"):
        for l in self.affine_layers:
            initialize_weights(l, init)
        initialize_weights(self.final, init, scale=1.0)

    def forward(self, x):
        '''
        Performs inference using the value network.
        Inputs:
        - x, the state passed in from the agent
        Returns:
        - The scalar (float) value of that state, as estimated by the net
        '''
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        value = self.final(x)
        return value

    def get_value(self, x):
        return self(x)

    def reset(self):
        return

    # MLP does not maintain history.
    def pause_history(self):
        return

    def continue_history(self):
        return

def main():
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to use, set to -1 if no GPU.')
    parser.add_argument('--env', type=str, default='Hopper-v2',#Hopper-v2,BipedalWalker-v3
                        help='OpenAI Gym MuJoCo env to perform algorithm on.')
    parser.add_argument('--num-envs', type=int, default=1,
                        help='Number of envs run in parallel.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--outdir', type=str, default='results_ppo_HP_pararm',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--steps', type=int, default=2.5 * 10 ** 6,
                        help='Total number of timesteps to train the agent.')
    parser.add_argument('--eval-interval', type=int, default=100000,
                        help='Interval in timesteps between evaluations.')
    parser.add_argument('--eval-n-runs', type=int, default=100,
                        help='Number of episodes run for each evaluation.')
    parser.add_argument('--render', action='store_true',
                        help='Render env states in a GUI window.')
    parser.add_argument('--demo', action='store_true',
                        help='Just run evaluation, not training.')
    parser.add_argument('--load-pretrained', action='store_true',
                        default=False)
    parser.add_argument('--load', type=str, default='',
                        help='Directory to load agent from.')
    parser.add_argument('--logger-level', type=int, default=logging.INFO,
                        help='Level of the root logger.')
    parser.add_argument('--monitor', action='store_true',
                        help='Wrap env with gym.wrappers.Monitor.')
    parser.add_argument('--log-interval', type=int, default=1000,
                        help='Interval in timesteps between outputting log'
                             ' messages during training')
    parser.add_argument('--update-interval', type=int, default=2048,
                        help='Interval in timesteps between model updates.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to update model for per PPO'
                             ' iteration.')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Minibatch size')
    parser.add_argument('--run_id', type=str, default='_Run1')

    parser.add_argument('--q_weight_init', type=float, default=0.8)
    parser.add_argument('--attack_budget', type=float, default=0.5)
    parser.add_argument('--constrained_distance', type=float, default=0.01)
    parser.add_argument('--attack_counter', type=int, default=25)
    parser.add_argument('--attack_learning_rate', type=float, default=1)
    parser.add_argument('--attack_norms', type=str, default='l2')
    parser.add_argument('--attack_epsilon', type=float, default=0.1)
    parser.add_argument('--initial_exploration_steps', type=int, default=100000)
    parser.add_argument('--memory_capacity', type=int, default=300000)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--q_func_update_interval', type=int, default=3)

    args = parser.parse_args()

    logging.basicConfig(level=args.logger_level)

    # Set a random seed used in ChainerRL
    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    args.outdir = experiments.prepare_output_dir(args, args.outdir)

    def make_env(process_idx, test):
        env = gym.make(args.env)
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[process_idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = chainerrl.wrappers.CastObservationToFloat32(env)
        if args.monitor:
            env = chainerrl.wrappers.Monitor(env, args.outdir)
        if args.render:
            env = chainerrl.wrappers.Render(env)
        return env

    def make_batch_env(test):
        return chainerrl.envs.MultiprocessVectorEnv(
            [functools.partial(make_env, idx, test)
             for idx, env in enumerate(range(args.num_envs))])

    # Only for getting timesteps, and obs-action spaces
    sample_env = gym.make(args.env)
    timestep_limit = sample_env.spec.max_episode_steps
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space
    print('Observation space:', obs_space)
    print('Action space:', action_space)

    assert isinstance(action_space, gym.spaces.Box)

    winit = chainerrl.initializers.Orthogonal(1.)
    winit_last = chainerrl.initializers.Orthogonal(1e-2)
    # Normalize observations based on their empirical mean and variance
    obs_normalizer = chainerrl.links.EmpiricalNormalization(
        obs_space.low.size, clip_threshold=5)

    value_hidden = 64
    obs_size = obs_space.low.size
    action_size = action_space.low.size

    def build_network(value_hidden):
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
            L.Linear(None, value_hidden, initialW=winit),
            F.tanh,
            L.Linear(None, 64, initialW=winit),
            F.tanh,
            L.Linear(None, 1, initialW=winit),
        )
        return policy, vf

    # For Mujoco Envs
    #################################################################
    if args.env == 'Hopper-v2' or args.env == 'Walker2d-v2' or args.env == 'HalfCheetah-v2':

        if args.env == 'Hopper-v2':
            value_hidden = 64

        elif args.env == 'Walker2d-v2' or args.env == 'HalfCheetah-v2':
            value_hidden = 128

    policy, vf = build_network(value_hidden)
    model = chainerrl.links.Branched(policy, vf)

    opt = chainer.optimizers.Adam(3e-4, eps=1e-5)
    opt.setup(model)

    def make_q_func_with_optimizer():
        winit = chainer.initializers.GlorotUniform()
        q_func = chainer.Sequential(
            concat_obs_and_action,
            L.Linear(None, 256, initialW=winit),
            F.relu,
            L.Linear(None, 256, initialW=winit),
            F.relu,
            L.Linear(None, 1, initialW=winit),
        )
        q_func_optimizer = chainer.optimizers.Adam(3e-4).setup(q_func)
        return q_func, q_func_optimizer

    q_func, q_func_optimizer = make_q_func_with_optimizer()
    worstq_model = ValueDenseNet(obs_size + action_size, 'orthogonal', activation='tanh')
    worstq_opt = optim.Adam(worstq_model.parameters(), lr=0.0004, eps=1e-5) 
    target_worstq_model = deepcopy(worstq_model)
    robust_eps_scheduler = LinearScheduler(0.05, "start=1,length=732")
    rbuf = replay_buffer.ReplayBuffer(10 ** 6)


    lam = lambda f: 1-f/976#学习率更新方式
    Q_SCHEDULER = optim.lr_scheduler.LambdaLR(worstq_opt , lr_lambda=lam)#s#动态更新价值网络学习率
    # print(action_space.low)
    agent = PPO(
        model,
        opt,
        rbuf,
        q_func,
        q_func_optimizer,
        worstq_model,
        worstq_opt,
        target_worstq_model,
        robust_eps_scheduler,
        Q_SCHEDULER,
        args.steps,
        q_weight_init=args.q_weight_init,
        obs_normalizer=obs_normalizer,
        gpu=args.gpu,
        update_interval=args.update_interval,
        minibatch_size=args.batch_size,
        epochs=args.epochs,
        entropy_coef=0,
        standardize_advantages=True,
        gamma=0.995,
        lambd=0.97,
        attack_budget=args.attack_budget,
        constrained_distance=args.constrained_distance,
        counter=args.attack_counter,
        attack_learning_rate=args.attack_learning_rate,
        attack_norms=args.attack_norms,
        attack_epsilon=args.attack_epsilon,
        action_low=action_space.low[0],
        action_high=action_space.high[0],
        initial_exploration_steps=args.initial_exploration_steps,
        q_func_update_interval=args.q_func_update_interval
    )

    if args.load or args.load_pretrained:
        # either load or load_pretrained must be false
        assert not args.load or not args.load_pretrained
        if args.load:
            agent.load(args.load)
        else:
            agent.load(misc.download_model(
                "PPO", args.env,
                model_type="final")[0])

    if args.demo:
        env = make_batch_env(True)
        eval_stats = experiments.eval_performance(
            env=env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=timestep_limit)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        experiments.train_agent_batch_with_evaluation(
            agent=agent,
            # env=make_batch_env(False),
            # eval_env=make_batch_env(True),
            env=make_env(0,False),
            eval_env=make_env(0,True),
            outdir=args.outdir,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            max_episode_len=timestep_limit,
            save_best_so_far_agent=False,
        )
    # agent.save('runs/PPO/' + 'PPO_' + args.env + args.run_id + str(args.seed))


if __name__ == '__main__':
    main()
