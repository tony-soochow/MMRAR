import collections
import itertools
from logging import getLogger
import random
from chainerrl.replay_buffer import ReplayUpdater, batch_experiences
import chainer
from chainer import cuda
import chainer.functions as F
import torch.nn.functional as F2
import numpy as np
import norms
import chainerrl
from chainerrl import agent
from chainerrl.agents.ppo import _compute_explained_variance
from agent.robust_adv_ppo_myworsttwostepsR import _make_dataset
from chainerrl.agents.ppo import _make_dataset_recurrent
from chainerrl.agents.ppo import _yield_subset_of_sequences_with_fixed_number_of_items  # NOQA
from chainerrl.misc.batch_states import batch_states
from chainerrl.misc.copy_param import synchronize_parameters
import cupy as cp
import math
import copy
import cupy as cp
from tqdm import tqdm
from torch_utils import *
def _get_ordered_params(link):
    """Get a list of parameters sorted by parameter names."""
    name_param_pairs = list(link.namedparams())
    ordered_name_param_pairs = sorted(name_param_pairs, key=lambda x: x[0])
    return [x[1] for x in ordered_name_param_pairs]


def _flatten_and_concat_variables(vs):
    """Flatten and concat variables to make a single flat vector variable."""
    return F.concat([F.flatten(v) for v in vs], axis=0)


def _as_ndarray(x):
    """chainer.Variable or ndarray -> ndarray."""
    if isinstance(x, chainer.Variable):
        return x.array
    else:
        return x


def _flatten_and_concat_ndarrays(vs):
    """Flatten and concat variables to make a single flat vector ndarray."""
    xp = chainer.cuda.get_array_module(vs[0])
    return xp.concatenate([_as_ndarray(v).ravel() for v in vs], axis=0)


def _split_and_reshape_to_ndarrays(flat_v, sizes, shapes):
    """Split and reshape a single flat vector to make a list of ndarrays."""
    xp = chainer.cuda.get_array_module(flat_v)
    sections = np.cumsum(sizes)
    vs = xp.split(flat_v, sections)
    return [v.reshape(shape) for v, shape in zip(vs, shapes)]


def _replace_params_data(params, new_params_data):
    """Replace data of params with new data."""
    for param, new_param_data in zip(params, new_params_data):
        assert param.shape == new_param_data.shape
        param.array[:] = new_param_data


def _hessian_vector_product(flat_grads, params, vec):
    """Compute hessian vector product efficiently by backprop."""
    grads = chainer.grad([F.sum(flat_grads * vec)], params)
    assert all(grad is not None for grad in grads),\
        "The Hessian-vector product contains None."
    grads_data = [grad.array for grad in grads]
    return _flatten_and_concat_ndarrays(grads_data)


def _mean_or_nan(xs):
    """Return its mean a non-empty sequence, numpy.nan for a empty one."""
    return np.mean(xs) if xs else np.nan


def _find_old_style_function(outputs):
    """Find old-style functions in the computational graph."""
    found = []
    for v in outputs:
        assert isinstance(v, (chainer.Variable, chainer.variable.VariableNode))
        if v.creator is None:
            continue
        if isinstance(v.creator, chainer.Function):
            found.append(v.creator)
        else:
            assert isinstance(v.creator, chainer.FunctionNode)
        found.extend(_find_old_style_function(v.creator.inputs))
    return found

def compute_grad(action, means, std_devs):
    # compute analytical gradient

    coeff = -(action - means) / ((np.power(std_devs, 3) * (np.sqrt(2 * np.pi))))
    power = -(np.power((action - means), 2)) / (2 * np.power(std_devs, 2))
    exp = np.exp(power)
    grad_a = coeff * exp
    return grad_a


def find_min_point(action, means, std_devs, lr, clip, budget, s, epsilon, total_counter, action_low, action_hihgh):
    if clip is True:
        action = cp.clip(action, action_low, action_hihgh)

    grad_a = compute_grad(action, means, std_devs)
    adv_action = action - (lr * grad_a)

    grad_a = compute_grad(adv_action, means, std_devs)
    adv_action_new = adv_action - (lr * grad_a)

    counter = 0
    while np.absolute(adv_action - adv_action_new).any() > epsilon and counter < total_counter:
        # print('Optimizing')
        adv_action = adv_action_new
        grad_a = compute_grad(adv_action, means, std_devs)
        adv_action_new = adv_action - (lr * grad_a)
        counter += 1

    delta = adv_action_new - action
    if s == 'l2':
        proj_spatial_delta = norms.l2_spatial_project(delta, budget)
    elif s == 'l1':
        proj_spatial_delta = norms.l1_spatial_project2(delta, budget)

    proj_action = action + proj_spatial_delta
    proj_action = cp.clip(proj_action, action_low, action_hihgh)

    return proj_action


def adversarl_attack(action, mean, var, attack_learning_rate, attack_budget, attack_norms, attack_epsilon, counter,
                     action_low, action_high):
    means = cp.array(mean.data)
    std_devs = cp.array(cp.sqrt(cp.exp(var.data)))

    min_action = find_min_point(action, means, std_devs, attack_learning_rate, True, attack_budget,
                                attack_norms, attack_epsilon, counter, action_low, action_high)

    return min_action


def find_min_point1(action, means, std_devs, lr, clip, budget, s, epsilon, total_counter, action_low, action_hihgh):
    if clip is True:
        action = np.clip(action, action_low, action_hihgh)

    grad_a = compute_grad(action, means, std_devs)
    adv_action = action - (lr * grad_a)

    grad_a = compute_grad(adv_action, means, std_devs)
    adv_action_new = adv_action - (lr * grad_a)

    counter = 0
    while np.absolute(adv_action - adv_action_new).any() > epsilon and counter < total_counter:
        # print('Optimizing')
        adv_action = adv_action_new
        grad_a = compute_grad(adv_action, means, std_devs)
        adv_action_new = adv_action - (lr * grad_a)
        counter += 1

    delta = adv_action_new - action
    if s == 'l2':
        proj_spatial_delta = norms.l2_spatial_project(delta, budget)
    elif s == 'l1':
        proj_spatial_delta = norms.l1_spatial_project2(delta, budget)

    proj_action = action + proj_spatial_delta
    proj_action = np.clip(proj_action, action_low, action_hihgh)

    return proj_action


def adversarl_attack1(action, mean, var, attack_learning_rate, attack_budget, attack_norms, attack_epsilon, counter,
                      action_low, action_high):
    means = np.array(mean.data.get())
    std_devs = np.array(np.sqrt(np.exp(var.data.get())))

    min_action = find_min_point1(action, means, std_devs, attack_learning_rate, True, attack_budget,
                                 attack_norms, attack_epsilon, counter, action_low, action_high)

    return min_action

def worst_action_pgd(q_net, policy_net, states, eps=0.0005, maxiter=100):
    with ch.no_grad():
        # action_ub, action_lb = policy_net(states)
        action_distrib, _ = policy_net(states)
        action = chainer.cuda.to_cpu(action_distrib.sample().array)
        action = ch.tensor(action, dtype=ch.float32)
        action_ub, action_lb=action+eps,action-eps
    # print(action_means)
    # var_actions = Variable(action_means.clone().to(device), requires_grad=True)

    var_actions = action.requires_grad_()
    step_eps = (action_ub - action_lb) / maxiter
    states = ch.tensor(states, dtype=ch.float32)
    for i in range(maxiter):
        worst_q = q_net(ch.cat((states, var_actions), dim=1))
        worst_q.backward(ch.ones_like(worst_q))
        grad = var_actions.grad.data  
        var_actions.data += step_eps * ch.sign(grad)#减修改成了加
        var_actions = ch.max(var_actions, action_lb)
        var_actions = ch.min(var_actions, action_ub)
        var_actions = var_actions.detach().requires_grad_() 
    q_net.zero_grad()
    return var_actions.detach()


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
class TRPO(agent.AttributeSavingMixin, agent.Agent):
    """Trust Region Policy Optimization.

    A given stochastic policy is optimized by the TRPO algorithm. A given
    value function is also trained to predict by the TD(lambda) algorithm and
    used for Generalized Advantage Estimation (GAE).

    Since the policy is optimized via the conjugate gradient method and line
    search while the value function is optimized via SGD, these two models
    should be separate.

    Since TRPO requires second-order derivatives to compute Hessian-vector
    products, Chainer v3.0.0 or newer is required. In addition, your policy
    must contain only functions that support second-order derivatives.

    See https://arxiv.org/abs/1502.05477 for TRPO.
    See https://arxiv.org/abs/1506.02438 for GAE.

    Args:
        policy (Policy): Stochastic policy. Its forward computation must
            contain only functions that support second-order derivatives.
        vf (ValueFunction): Value function.
        vf_optimizer (chainer.Optimizer): Optimizer for the value function.
        obs_normalizer (chainerrl.links.EmpiricalNormalization or None):
            If set to chainerrl.links.EmpiricalNormalization, it is used to
            normalize observations based on the empirical mean and standard
            deviation of observations. These statistics are updated after
            computing advantages and target values and before updating the
            policy and the value function.
        gamma (float): Discount factor [0, 1]
        lambd (float): Lambda-return factor [0, 1]
        phi (callable): Feature extractor function
        entropy_coef (float): Weight coefficient for entropoy bonus [0, inf)
        update_interval (int): Interval steps of TRPO iterations. Every after
            this amount of steps, this agent updates the policy and the value
            function using data from these steps.
        vf_epochs (int): Number of epochs for which the value function is
            trained on each TRPO iteration.
        vf_batch_size (int): Batch size of SGD for the value function.
        standardize_advantages (bool): Use standardized advantages on updates
        line_search_max_backtrack (int): Maximum number of backtracking in line
            search to tune step sizes of policy updates.
        conjugate_gradient_max_iter (int): Maximum number of iterations in
            the conjugate gradient method.
        conjugate_gradient_damping (float): Damping factor used in the
            conjugate gradient method.
        act_deterministically (bool): If set to True, choose most probable
            actions in the act method instead of sampling from distributions.
        value_stats_window (int): Window size used to compute statistics
            of value predictions.
        entropy_stats_window (int): Window size used to compute statistics
            of entropy of action distributions.
        kl_stats_window (int): Window size used to compute statistics
            of KL divergence between old and new policies.
        policy_step_size_stats_window (int): Window size used to compute
            statistics of step sizes of policy updates.

    Statistics:
        average_value: Average of value predictions on non-terminal states.
            It's updated before the value function is updated.
        average_entropy: Average of entropy of action distributions on
            non-terminal states. It's updated on act_and_train.
        average_kl: Average of KL divergence between old and new policies.
            It's updated after the policy is updated.
        average_policy_step_size: Average of step sizes of policy updates
            It's updated after the policy is updated.
    """

    saved_attributes = ['policy', 'vf', 'vf_optimizer', 'obs_normalizer']

    def __init__(self,
                 replay_buffer,
                 q_func,
                 q_func_optimizer,
                 worstq_model,
                 worstq_opt,
                 target_worstq_model,
                 robust_eps_scheduler,
                 Q_SCHEDULER,
                 steps,
                 policy,
                 vf,
                 vf_optimizer,
                 obs_normalizer=None,
                 gpu=None,
                 gamma=0.99,
                 lambd=0.95,
                 phi=lambda x: x,
                 entropy_coef=0.01,
                 update_interval=2048,
                 max_kl=0.01,
                 vf_epochs=3,
                 vf_batch_size=64,
                 standardize_advantages=True,
                 batch_states=batch_states,
                 recurrent=False,
                 max_recurrent_sequence_len=None,
                 line_search_max_backtrack=10,
                 conjugate_gradient_max_iter=10,
                 conjugate_gradient_damping=1e-2,
                 act_deterministically=False,
                 value_stats_window=1000,
                 entropy_stats_window=1000,
                 kl_stats_window=100,
                 policy_step_size_stats_window=100,
                 minibatch_size=64,
                 attack_budget=1,
                 constrained_distance=0.03,
                 counter=0,
                 attack_learning_rate=1,
                 attack_epsilon=0.1,
                 attack_norms='l2',
                 initial_exploration_steps=3000,
                 action_low=-1.0,
                 action_high=1.0,
                 q_func_update_interval=1,
                 value_loss_stats_window=100,
                 logger=getLogger(__name__),
                 ):

        self.policy = policy
        self.vf = vf
        self.q_func = q_func
        self.q_func_optimizer = q_func_optimizer
        assert policy.xp is vf.xp, 'policy and vf must be on the same device'
        if recurrent:
            self.model = chainerrl.links.StatelessRecurrentBranched(policy, vf)
        else:
            self.model = chainerrl.links.Branched(policy, vf)
        if policy.xp is not np:
            if hasattr(policy, 'device'):
                # Link.device is available only from chainer v6
                self.model.to_device(policy.device)
            else:
                self.model.to_gpu(device=policy._device_id)
        self.vf_optimizer = vf_optimizer
        self.obs_normalizer = obs_normalizer
        self.gamma = gamma
        self.lambd = lambd
        self.phi = phi
        self.entropy_coef = entropy_coef
        self.update_interval = update_interval
        self.max_kl = max_kl
        self.vf_epochs = vf_epochs
        self.vf_batch_size = vf_batch_size
        self.standardize_advantages = standardize_advantages
        self.batch_states = batch_states
        self.recurrent = recurrent
        self.max_recurrent_sequence_len = max_recurrent_sequence_len
        self.line_search_max_backtrack = line_search_max_backtrack
        self.conjugate_gradient_max_iter = conjugate_gradient_max_iter
        self.conjugate_gradient_damping = conjugate_gradient_damping
        self.act_deterministically = act_deterministically
        
        self.attack_learning_rate = attack_learning_rate
        self.attack_epsilon = attack_epsilon
        self.attack_norms = attack_norms
        self.counter = counter
        self.attack_budget = attack_budget
        self.constrained_facor = constrained_distance
        self.action_low = action_low
        self.action_high = action_high
        
        self.logger = logger

        self.value_record = collections.deque(maxlen=value_stats_window)
        self.entropy_record = collections.deque(maxlen=entropy_stats_window)
        self.kl_record = collections.deque(maxlen=kl_stats_window)
        self.policy_step_size_record = collections.deque(
            maxlen=policy_step_size_stats_window)
        self.explained_variance = np.nan

        assert self.policy.xp is self.vf.xp,\
            'policy and vf should be in the same device.'
        if self.obs_normalizer is not None:
            assert self.policy.xp is self.obs_normalizer.xp,\
                'policy and obs_normalizer should be in the same device.'
        self.xp = self.policy.xp
        self.last_state = None
        self.last_action = None
        self.batch_last_obs = None
        self.batch_last_act= None
        self.batch_last_env = None

        # Contains episodes used for next update iteration
        self.memory = []
        # Contains transitions of the last episode not moved to self.memory yet
        self.last_episode = []
        self.worstq=worstq_model
        self.worstq_opt=worstq_opt
        self.target_worstq=target_worstq_model
        self.worstq_eps_scheduler=robust_eps_scheduler
        self.Q_SCHEDULER=Q_SCHEDULER
        self.steps=steps
        self.replay_buffer = replay_buffer
        self.minibatch_size = minibatch_size
        self.replay_updater = ReplayUpdater(
            replay_buffer=replay_buffer,
            update_func=self.update,
            batchsize=minibatch_size,
            n_times_update=1,
            replay_start_size=10000,
            update_interval=q_func_update_interval,
            episodic_update=False,
        )
        
        if gpu is not None and gpu >= 0:
            cuda.get_device_from_id(gpu).use()
            self.model.to_gpu(device=gpu)
            self.q_func.to_gpu(device=gpu)

            if self.obs_normalizer is not None:
                self.obs_normalizer.to_gpu(device=gpu)
        
        # Batch versions of last_episode, last_state, and last_action
        self.batch_last_episode = None
        self.batch_last_state = None
        self.batch_last_action = None

        # Recurrent states of the model
        self.train_recurrent_states = None
        self.train_prev_recurrent_states = None
        self.test_recurrent_states = None
        self.q_func_loss_record = collections.deque(maxlen=value_loss_stats_window)
        self.value_loss_record = collections.deque(
            maxlen=value_loss_stats_window)
        self.t = 0
        self.wt=0
        self.target_q_func = copy.deepcopy(self.q_func)
        self.initial_exploration_steps = initial_exploration_steps
    def sync_target_network(self):
        """Synchronize target network with current network."""
        synchronize_parameters(
            src=self.q_func,
            dst=self.target_q_func,
            method='soft',
            tau=5e-3,
        )
 
        
    def batch_experiences(self, experiences, xp, phi, gamma, batch_states=batch_states):
        """Takes a batch of k experiences each of which contains j

        consecutive transitions and vectorizes them, where j is between 1 and n.

        Args:
            experiences: list of experiences. Each experience is a list
                containing between 1 and n dicts containing
                - state (object): State
                - action (object): Action
                - reward (float): Reward
                - is_state_terminal (bool): True iff next state is terminal
                - next_state (object): Next state
            xp : Numpy compatible matrix library: e.g. Numpy or CuPy.
            phi : Preprocessing function
            gamma: discount factor
            batch_states: function that converts a list to a batch
        Returns:
            dict of batched transitions
        """

        batch_exp = {
            'state': batch_states(
                [elem[0]['state'] for elem in experiences], xp, phi),
            'action': xp.asarray([elem[0]['action'] for elem in experiences]),
            'perturbed_action': xp.asarray([elem[0]['perturbed_action'] for elem in experiences]),
            'reward': xp.asarray([sum((gamma ** i) * exp[i]['reward']
                                      for i in range(len(exp)))
                                  for exp in experiences],
                                 dtype=np.float32),
            'next_state': batch_states(
                [elem[-1]['next_state']
                 for elem in experiences], xp, phi),
            'is_state_terminal': xp.asarray(
                [any(transition['is_state_terminal']
                     for transition in exp) for exp in experiences],
                dtype=np.float32),
            'discount': xp.asarray([(gamma ** len(elem)) for elem in experiences],
                                   dtype=np.float32)}
        if all(elem[-1]['next_action'] is not None for elem in experiences):
            batch_exp['next_action'] = xp.asarray(
                [elem[-1]['next_action'] for elem in experiences])
        return batch_exp

    def update(self, experiences, errors_out=None):
        """Update the model from experiences"""

        batch = self.batch_experiences(experiences, self.xp, self.phi, self.gamma)

        self.update_q_func(batch)
        self.sync_target_network()

    def update_q_func(self, batch):
        """Compute loss for a given Q-function."""

        batch_next_state = batch['next_state']
        batch_rewards = batch['reward']
        batch_terminal = batch['is_state_terminal']
        batch_state = batch['state']
        batch_actions = batch['action']
        batch_discount = batch['discount']

        if self.obs_normalizer:
            norm_next_states = self.obs_normalizer(batch_next_state, update=False)
        else:
            norm_next_states = batch_next_state

        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            next_action_distrib, _ = self.model(norm_next_states)
            next_actions, next_log_prob = \
                next_action_distrib.sample_with_log_prob()

            next_q = self.target_q_func(batch_next_state, next_actions)

            target_q = batch_rewards + batch_discount * \
                       (1.0 - batch_terminal) * F.flatten(next_q)

        predict_q = F.flatten(self.q_func(batch_state, batch_actions))

        loss = F.mean_squared_error(target_q, predict_q)


        # Update stats
        self.q_func_loss_record.append(float(loss.array))
        self.q_func_optimizer.update(lambda: loss)

    def _initialize_batch_variables(self, num_envs):
        self.batch_last_episode = [[] for _ in range(num_envs)]
        self.batch_last_state = [None] * num_envs
        self.batch_last_action = [None] * num_envs

    def _update_if_dataset_is_ready(self):
        dataset_size = (
            sum(len(episode) for episode in self.memory)
            + len(self.last_episode)
            + (0 if self.batch_last_episode is None else sum(
                len(episode) for episode in self.batch_last_episode)))
        if dataset_size >= self.update_interval:
            self._flush_last_episode()
            if self.recurrent:
                dataset = _make_dataset_recurrent(
                    episodes=self.memory,
                    model=self.model,
                    phi=self.phi,
                    batch_states=self.batch_states,
                    obs_normalizer=self.obs_normalizer,
                    gamma=self.gamma,
                    lambd=self.lambd,
                    max_recurrent_sequence_len=self.max_recurrent_sequence_len,
                )
                self._update_recurrent(dataset)
            else:
                penalty_term = False
                if self.t > self.initial_exploration_steps:
                    penalty_term = True
                dataset = _make_dataset(
                    episodes=self.memory,
                    model=self.model,
                    bounded_model=self.q_func,
                    worstq_model=self.worstq,
                    phi=self.phi,
                    batch_states=self.batch_states,
                    obs_normalizer=self.obs_normalizer,
                    gamma=self.gamma,
                    lambd=self.lambd,
                    attack_learning_rate=self.attack_learning_rate,
                    attack_budget=self.attack_budget,
                    attack_norms=self.attack_norms,
                    attack_epsilon=self.attack_epsilon,
                    constrained_facor=self.constrained_facor,
                    counter=self.counter,
                    penalty_term=penalty_term,
                    action_low=self.action_low,
                    action_high=self.action_high
                )
                assert len(dataset) == dataset_size
                self._update(dataset)
            self.explained_variance = _compute_explained_variance(
                list(itertools.chain.from_iterable(self.memory)))
            self.memory = []

    def _flush_last_episode(self):
        if self.last_episode:
            self.memory.append(self.last_episode)
            self.last_episode = []
        if self.batch_last_episode:
            for i, episode in enumerate(self.batch_last_episode):
                if episode:
                    self.memory.append(episode)
                    self.batch_last_episode[i] = []

    def _update(self, dataset):
        """Update both the policy and the value function."""

        if self.obs_normalizer:
            self._update_obs_normalizer(dataset)
        self._update_worstq(dataset)
        self._update_policy(dataset)
        self._update_vf(dataset)

    def _update_recurrent(self, dataset):
        """Update both the policy and the value function."""

        flat_dataset = list(itertools.chain.from_iterable(dataset))
        if self.obs_normalizer:
            self._update_obs_normalizer(flat_dataset)

        self._update_policy_recurrent(dataset)
        self._update_vf_recurrent(dataset)

    def _update_vf_recurrent(self, dataset):

        for epoch in range(self.vf_epochs):
            random.shuffle(dataset)
            for minibatch in _yield_subset_of_sequences_with_fixed_number_of_items(  # NOQA
                    dataset, self.vf_batch_size):
                self._update_vf_once_recurrent(minibatch)

    def _update_vf_once_recurrent(self, episodes):

        xp = self.model.xp
        flat_transitions = list(itertools.chain.from_iterable(episodes))

        # Prepare data for a recurrent model
        seqs_states = []
        for ep in episodes:
            states = self.batch_states(
                [transition['state'] for transition in ep], xp, self.phi)
            if self.obs_normalizer:
                states = self.obs_normalizer(states, update=False)
            seqs_states.append(states)

        flat_vs_teacher = xp.array(
            [[transition['v_teacher']] for transition in flat_transitions],
            dtype=np.float32)

        with chainer.using_config('train', False),\
                chainer.no_backprop_mode():
            vf_rs = self.vf.concatenate_recurrent_states(
                [ep[0]['recurrent_state'][1] for ep in episodes])

        flat_vs_pred, _ = self.vf.n_step_forward(
            seqs_states, recurrent_state=vf_rs, output_mode='concat')

        vf_loss = F.mean_squared_error(flat_vs_pred, flat_vs_teacher)
        self.vf_optimizer.update(lambda: vf_loss)

    def _update_obs_normalizer(self, dataset):
        assert self.obs_normalizer
        states = batch_states(
            [b['state'] for b in dataset], self.obs_normalizer.xp, self.phi)
        self.obs_normalizer.experience(states)
        
        
    def _update_worstq(self, dataset):
        """Update both the policy and the value function."""

        if self.obs_normalizer:
            self._update_obs_normalizer(dataset)

        xp = self.model.xp

        assert 'state' in dataset[0]
        assert 'v_teacher' in dataset[0]
        states = self.batch_states(
            [b['state'] for b in dataset], xp, self.phi)
        next_states = self.batch_states(
            [b['next_state'] for b in dataset], xp, self.phi)
        if self.obs_normalizer:
            states = self.obs_normalizer(states, update=False)
            next_states = self.obs_normalizer(next_states, update=False)
        actions = xp.array([b['action'] for b in dataset])
        not_dones = xp.array([b['nonterminal'] for b in dataset])
        rewards = xp.array([b['regret_reward'] for b in dataset])
        parmas={"GAMMA":0.99,'Q_EPOCHS':10,'NUM_MINIBATCHES':32,'TAU':0.001}
        q_loss=self.worst_q_step(states,actions,next_states,not_dones,rewards,self.worstq,self.target_worstq,self.model,self.worstq_opt,parmas,self.worstq_eps_scheduler)
        self.Q_SCHEDULER.step()


    def worst_q_step(self,all_states, actions, next_states, not_dones, rewards, q_net, target_q_net, policy_net, q_opt,     
                params, eps_scheduler, should_tqdm=False, should_cuda=False):
        
        '''
        Take an optimizer step training the worst-q function
        parameterized by a neural network
        Inputs:
        - all_states, the states at each timestep
        - actions, the actions taking at each timestep
        - next_states, the next states after taking actions
        - not dones, N * T array with 0s at final steps and 1s everywhere else
        - rewards, the rewards gained at each timestep
        - q_net, worst-case q neural network
        - q_opt, the optimizer for q_net
        - target_q_net, the target q_net
        - params, dictionary of parameters
        Returns:
        - Loss of the q_net regression problem
        '''
        current_eps = eps_scheduler.get_eps()

        r = range(params['Q_EPOCHS']) if not should_tqdm else \
                                tqdm(range(params['Q_EPOCHS']))
        all_states = ch.tensor(all_states, dtype=ch.float32)
        actions = ch.tensor(actions, dtype=ch.float32)
        rewards = ch.tensor(rewards, dtype=ch.float32)
        not_dones = ch.tensor(not_dones, dtype=ch.float32)
        for i in r:
            # Create minibatches with shuffuling
            state_indices = np.arange(rewards.nelement())
            np.random.shuffle(state_indices)
            splits = np.array_split(state_indices, params['NUM_MINIBATCHES'])

            assert shape_equal_cmp(rewards, not_dones)

            # Minibatch SGD
            for selected in splits:
                q_opt.zero_grad()

                def sel(*args):
                    return [v[selected] for v in args]

                def to_cuda(*args):
                    return [v.cuda() for v in args]

                # Get a minibatch (64).
                tup = sel(actions, rewards, not_dones, next_states, all_states)
                mask = ch.tensor(True)

                if should_cuda: tup = to_cuda(*tup)
                sel_acts, sel_rews, sel_not_dones, sel_next_states, sel_states = tup

                # Worst q prediction of current network given the states.
                curr_q = q_net(ch.cat((sel_states, sel_acts), dim=1)).squeeze(-1)
                # worst_actions = worst_action_pgd(q_net, policy_net, sel_next_states, eps=0.01, maxiter=50)
                next_action_distrib, _ = policy_net(sel_next_states)
                sel_next_action = chainer.cuda.to_cpu(next_action_distrib.sample().array)
                sel_next_action = ch.tensor(sel_next_action, dtype=ch.float32)
                sel_next_states = ch.tensor(sel_next_states, dtype=ch.float32)
                expected_q = sel_rews + params['GAMMA'] * sel_not_dones * target_q_net(ch.cat((sel_next_states, sel_next_action), dim=1)).squeeze(-1)
                '''
                print('curr_q', curr_q.mean())
                print('expected_q', expected_q.mean())
                '''
                q_loss = F2.mse_loss(curr_q, expected_q)
                q_loss.backward()
                q_opt.step()
                soft_update(target_q_net, q_net, params['TAU'])

            # print(f'q_loss={q_loss.item():8.5f}')

        return q_loss
    

    def _update_vf(self, dataset):
        """Update the value function using a given dataset.

        The value function is updated via SGD to minimize TD(lambda) errors.
        """

        xp = self.vf.xp

        assert 'state' in dataset[0]
        assert 'v_teacher' in dataset[0]

        dataset_iter = chainer.iterators.SerialIterator(
            dataset, self.vf_batch_size)

        while dataset_iter.epoch < self.vf_epochs:
            batch = dataset_iter.__next__()
            states = batch_states([b['state'] for b in batch], xp, self.phi)
            if self.obs_normalizer:
                states = self.obs_normalizer(states, update=False)
            vs_teacher = xp.array(
                [b['v_teacher'] for b in batch], dtype=xp.float32)
            vs_pred = self.vf(states)
            vf_loss = F.mean_squared_error(vs_pred, vs_teacher[..., None])
            self.vf_optimizer.update(lambda: vf_loss)

    def _compute_gain(self, log_prob, log_prob_old, entropy, advs):
        """Compute a gain to maximize."""
        prob_ratio = F.exp(log_prob - log_prob_old)
        mean_entropy = F.mean(entropy)
        surrogate_gain = F.mean(prob_ratio * advs)
        return surrogate_gain + self.entropy_coef * mean_entropy

    def _update_policy(self, dataset):
        """Update the policy using a given dataset.

        The policy is updated via CG and line search.
        """

        assert 'state' in dataset[0]
        assert 'action' in dataset[0]
        assert 'adv' in dataset[0]

        # Use full-batch
        xp = self.policy.xp
        states = batch_states([b['state'] for b in dataset], xp, self.phi)
        env_states = ch.tensor(states, dtype=ch.float32)
        if self.obs_normalizer:
            states = self.obs_normalizer(states, update=False)
        actions = xp.array([b['action'] for b in dataset])
        actions_tensor = ch.tensor(actions, dtype=ch.float32)
        
        advs = xp.array([b['adv'] for b in dataset], dtype=np.float32)
        if self.standardize_advantages:
            mean_advs = xp.mean(advs)
            std_advs = xp.std(advs)
            advs = (advs - mean_advs) / (std_advs + 1e-8)
        
        worst=self.worstq(ch.cat((env_states,actions_tensor), dim=1)).squeeze(-1)
        q_weight = math.pow((0.8/self.steps) * float(self.wt), 3)
        worst_array = cp.asarray(worst.detach().cpu())
        mean_worst = xp.mean(worst_array)
        std_worst = xp.std(worst_array)
        worst = (worst_array - mean_worst) / (std_worst + 1e-8)
        worst=worst*q_weight
        # worst=worst*q_weight
        # worst=worst.detach()
        # worst=worst.cpu().numpy().astype(cp.float32)
        advs-=cp.array(worst,dtype=cp.float64)
        

        # Recompute action distributions for batch backprop
        action_distrib = self.policy(states)

        log_prob_old = xp.array(
            [transition['log_prob'] for transition in dataset],
            dtype=np.float32)

        gain = self._compute_gain(
            log_prob=action_distrib.log_prob(actions),
            log_prob_old=log_prob_old,
            entropy=action_distrib.entropy,
            advs=advs)

        # Distribution to compute KL div against
        action_distrib_old = action_distrib.copy()

        full_step = self._compute_kl_constrained_step(
            action_distrib=action_distrib,
            action_distrib_old=action_distrib_old,
            gain=gain)

        self._line_search(
            full_step=full_step,
            dataset=dataset,
            advs=advs,
            action_distrib_old=action_distrib_old,
            gain=gain)

    def _update_policy_recurrent(self, dataset):
        """Update the policy using a given dataset.

        The policy is updated via CG and line search.
        """

        xp = self.model.xp
        flat_transitions = list(itertools.chain.from_iterable(dataset))

        # Prepare data for a recurrent model
        seqs_states = []
        for ep in dataset:
            states = self.batch_states(
                [transition['state'] for transition in ep], xp, self.phi)
            if self.obs_normalizer:
                states = self.obs_normalizer(states, update=False)
            seqs_states.append(states)

        flat_actions = xp.array(
            [transition['action'] for transition in flat_transitions])
        flat_advs = xp.array(
            [transition['adv'] for transition in flat_transitions],
            dtype=np.float32)

        if self.standardize_advantages:
            mean_advs = xp.mean(flat_advs)
            std_advs = xp.std(flat_advs)
            flat_advs = (flat_advs - mean_advs) / (std_advs + 1e-8)

        with chainer.using_config('train', False),\
                chainer.no_backprop_mode():
            policy_rs = self.policy.concatenate_recurrent_states(
                [ep[0]['recurrent_state'][0] for ep in dataset])

        flat_distribs, _ = self.policy.n_step_forward(
            seqs_states, recurrent_state=policy_rs, output_mode='concat')

        log_prob_old = xp.array(
            [transition['log_prob'] for transition in flat_transitions],
            dtype=np.float32)

        gain = self._compute_gain(
            log_prob=flat_distribs.log_prob(flat_actions),
            log_prob_old=log_prob_old,
            entropy=flat_distribs.entropy,
            advs=flat_advs)

        # Distribution to compute KL div against
        action_distrib_old = flat_distribs.copy()

        full_step = self._compute_kl_constrained_step(
            action_distrib=flat_distribs,
            action_distrib_old=action_distrib_old,
            gain=gain)

        self._line_search(
            full_step=full_step,
            dataset=dataset,
            advs=flat_advs,
            action_distrib_old=action_distrib_old,
            gain=gain)

    def _compute_kl_constrained_step(self, action_distrib, action_distrib_old,
                                     gain):
        """Compute a step of policy parameters with a KL constraint."""
        policy_params = _get_ordered_params(self.policy)
        kl = F.mean(action_distrib_old.kl(action_distrib))

        # Check if kl computation fully supports double backprop
        old_style_funcs = _find_old_style_function([kl])
        if old_style_funcs:
            raise RuntimeError("""\
Old-style functions (chainer.Function) are used to compute KL divergence.
Since TRPO requires second-order derivative of KL divergence, its computation
should be done with new-style functions (chainer.FunctionNode) only.

Found old-style functions: {}""".format(old_style_funcs))

        kl_grads = chainer.grad([kl], policy_params,
                                enable_double_backprop=True)
        assert all(g is not None for g in kl_grads), "\
The gradient contains None. The policy may have unused parameters."
        flat_kl_grads = _flatten_and_concat_variables(kl_grads)

        def fisher_vector_product_func(vec):
            fvp = _hessian_vector_product(flat_kl_grads, policy_params, vec)
            return fvp + self.conjugate_gradient_damping * vec

        gain_grads = chainer.grad([gain], policy_params)
        assert all(g is not None for g in kl_grads), "\
The gradient contains None. The policy may have unused parameters."
        flat_gain_grads = _flatten_and_concat_ndarrays(gain_grads)
        step_direction = chainerrl.misc.conjugate_gradient(
            fisher_vector_product_func, flat_gain_grads,
            max_iter=self.conjugate_gradient_max_iter,
        )

        # We want a step size that satisfies KL(old|new) < max_kl.
        # Let d = alpha * step_direction be the actual parameter updates.
        # The second-order approximation of KL divergence is:
        #   KL(old|new) = 1/2 d^T I d + O(||d||^3),
        # where I is a Fisher information matrix.
        # Substitute d = alpha * step_direction and solve KL(old|new) = max_kl
        # for alpha to get the step size that tightly satisfies the constraint.

        dId = float(step_direction.dot(
            fisher_vector_product_func(step_direction)))
        scale = (2.0 * self.max_kl / (dId + 1e-8)) ** 0.5
        return scale * step_direction

    def _line_search(self, full_step, dataset, advs, action_distrib_old, gain):
        """Do line search for a safe step size."""
        xp = self.policy.xp
        policy_params = _get_ordered_params(self.policy)
        policy_params_sizes = [param.size for param in policy_params]
        policy_params_shapes = [param.shape for param in policy_params]
        step_size = 1.0
        flat_params = _flatten_and_concat_ndarrays(policy_params)

        if self.recurrent:
            seqs_states = []
            for ep in dataset:
                states = self.batch_states(
                    [transition['state'] for transition in ep], xp, self.phi)
                if self.obs_normalizer:
                    states = self.obs_normalizer(states, update=False)
                seqs_states.append(states)
            with chainer.using_config('train', False),\
                    chainer.no_backprop_mode():
                policy_rs = self.policy.concatenate_recurrent_states(
                    [ep[0]['recurrent_state'][0] for ep in dataset])

            def evaluate_current_policy():
                distrib, _ = self.policy.n_step_forward(
                    seqs_states, recurrent_state=policy_rs,
                    output_mode='concat')
                return distrib
        else:
            states = self.batch_states(
                [transition['state'] for transition in dataset], xp, self.phi)
            if self.obs_normalizer:
                states = self.obs_normalizer(states, update=False)

            def evaluate_current_policy():
                return self.policy(states)

        flat_transitions = (list(itertools.chain.from_iterable(dataset))
                            if self.recurrent else dataset)
        actions = xp.array(
            [transition['action'] for transition in flat_transitions])
        log_prob_old = xp.array(
            [transition['log_prob'] for transition in flat_transitions],
            dtype=np.float32)

        for i in range(self.line_search_max_backtrack + 1):
            self.logger.info(
                'Line search iteration: %s step size: %s', i, step_size)
            new_flat_params = flat_params + step_size * full_step
            new_params = _split_and_reshape_to_ndarrays(
                new_flat_params,
                sizes=policy_params_sizes,
                shapes=policy_params_shapes,
            )
            _replace_params_data(policy_params, new_params)
            with chainer.using_config('train', False),\
                    chainer.no_backprop_mode():
                new_action_distrib = evaluate_current_policy()
                new_gain = self._compute_gain(
                    log_prob=new_action_distrib.log_prob(actions),
                    log_prob_old=log_prob_old,
                    entropy=new_action_distrib.entropy,
                    advs=advs)
                new_kl = F.mean(action_distrib_old.kl(new_action_distrib))

            improve = new_gain.array - gain.array
            self.logger.info(
                'Surrogate objective improve: %s', float(improve))
            self.logger.info('KL divergence: %s', float(new_kl.array))
            if not xp.isfinite(new_gain.array):
                self.logger.info(
                    "Surrogate objective is not finite. Bakctracking...")
            elif not xp.isfinite(new_kl.array):
                self.logger.info(
                    "KL divergence is not finite. Bakctracking...")
            elif improve < 0:
                self.logger.info(
                    "Surrogate objective didn't improve. Bakctracking...")
            elif float(new_kl.array) > self.max_kl:
                self.logger.info(
                    "KL divergence exceeds max_kl. Bakctracking...")
            else:
                self.kl_record.append(float(new_kl.array))
                self.policy_step_size_record.append(step_size)
                break
            step_size *= 0.5
        else:
            self.logger.info("\
Line search coundn't find a good step size. The policy was not updated.")
            self.policy_step_size_record.append(0.)
            _replace_params_data(
                policy_params,
                _split_and_reshape_to_ndarrays(
                    flat_params,
                    sizes=policy_params_sizes,
                    shapes=policy_params_shapes),
            )

    def act_and_train(self, obs, reward,batch_env):
        
        self.wt+=1
        
        if self.last_state is not None:
            transition = {
                'state': self.last_state,
                'action': self.last_action,
                'env':self.batch_last_env, 
                'reward': reward,
                'next_state': obs,
                'nonterminal': 1.0,
            }
            if self.recurrent:
                transition['recurrent_state'] =\
                    self.model.get_recurrent_state_at(
                        self.train_prev_recurrent_states,
                        0, unwrap_variable=True)
                self.train_prev_recurrent_states = None
                transition['next_recurrent_state'] =\
                    self.model.get_recurrent_state_at(
                        self.train_recurrent_states, 0, unwrap_variable=True)
            self.last_episode.append(transition)

        self._update_if_dataset_is_ready()
        
        self.t += 1
        if self.batch_last_obs is not None:
            assert self.batch_last_act is not None
            # Add a transition to the replay buffer
            self.replay_buffer.append(
                state=self.batch_last_obs,
                action=self.batch_last_act,
                perturbed_action=self.batch_last_per_act,
                reward=reward,
                next_state=obs,
                next_action=None,
                is_state_terminal=False,
                env_id=0,
            )
        self.replay_updater.update_if_necessary(self.t)

        xp = self.xp
        b_state = self.batch_states([obs], xp, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        # action_distrib will be recomputed when computing gradients
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            if self.recurrent:
                assert self.train_prev_recurrent_states is None
                self.train_prev_recurrent_states = self.train_recurrent_states
                (action_distrib, value), self.train_recurrent_states =\
                    self.model(b_state, self.train_prev_recurrent_states)
            else:
                action_distrib, value = self.model(b_state)
            action = chainer.cuda.to_cpu(action_distrib.sample().array)[0]
            self.entropy_record.append(float(action_distrib.entropy.array))
            self.value_record.append(float(value.array))
        

        self.last_state = obs
        self.last_action = action
        self.batch_last_env = batch_env

        min_action = adversarl_attack1(action, action_distrib.mean, action_distrib.ln_var,
                                self.attack_learning_rate, self.attack_budget, self.attack_norms,
                                self.attack_epsilon, self.counter,
                                self.action_low, self.action_high)
        self.batch_last_obs = obs
        self.batch_last_act = action
        self.batch_last_per_act = min_action
        
        return min_action

    def act(self, obs):
        xp = self.xp
        b_state = self.batch_states([obs], xp, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            if self.recurrent:
                action_distrib, self.test_recurrent_states =\
                    self.policy(b_state, self.test_recurrent_states)
            else:
                action_distrib = self.policy(b_state)
            if self.act_deterministically:
                action = chainer.cuda.to_cpu(
                    action_distrib.most_probable.array)[0]
            else:
                action = chainer.cuda.to_cpu(
                    action_distrib.sample().array)[0]

        return action

    def stop_rbf(self, obs, reward, done=False):

        assert self.batch_last_obs is not None
        self.replay_buffer.append(
            state=self.batch_last_obs,
            action=self.batch_last_act,
            perturbed_action=self.batch_last_per_act,
            reward=reward,
            next_state=obs,
            next_action=None,
            is_state_terminal=done,
            env_id=0,
            )
        self.batch_last_obs=None
        self.batch_last_act=None
        self.replay_buffer.stop_current_episode(env_id=0)

    
    def stop_episode_and_train(self, state, reward, done=False):

        assert self.last_state is not None
        transition = {
            'state': self.last_state,
            'action': self.last_action,
            'env':self.batch_last_env,
            'reward': reward,
            'next_state': state,
            'nonterminal': 0.0 if done else 1.0,
        }
        if self.recurrent:
            transition['recurrent_state'] = self.model.get_recurrent_state_at(
                self.train_prev_recurrent_states, 0, unwrap_variable=True)
            self.train_prev_recurrent_states = None
            transition['next_recurrent_state'] =\
                self.model.get_recurrent_state_at(
                    self.train_recurrent_states, 0, unwrap_variable=True)
            self.train_recurrent_states = None
        self.last_episode.append(transition)

        self.last_state = None
        self.last_action = None
        self.batch_last_env = None

        self._flush_last_episode()
        self.stop_episode()

        self._update_if_dataset_is_ready()

    def stop_episode(self):
        self.test_recurrent_states = None

    def batch_act(self, batch_obs):
        xp = self.xp
        b_state = self.batch_states(batch_obs, xp, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            if self.recurrent:
                (action_distrib, _), self.test_recurrent_states = self.model(
                    b_state, self.test_recurrent_states)
            else:
                action_distrib, _ = self.model(b_state)
            if self.act_deterministically:
                action = chainer.cuda.to_cpu(
                    action_distrib.most_probable.array)
            else:
                action = chainer.cuda.to_cpu(action_distrib.sample().array)

        return action

    def batch_act_and_train(self, batch_obs):
        xp = self.xp
        b_state = self.batch_states(batch_obs, xp, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        num_envs = len(batch_obs)
        if self.batch_last_episode is None:
            self._initialize_batch_variables(num_envs)
        assert len(self.batch_last_episode) == num_envs
        assert len(self.batch_last_state) == num_envs
        assert len(self.batch_last_action) == num_envs

        # action_distrib will be recomputed when computing gradients
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            if self.recurrent:
                assert self.train_prev_recurrent_states is None
                self.train_prev_recurrent_states = self.train_recurrent_states
                (action_distrib, batch_value), self.train_recurrent_states =\
                    self.model(b_state, self.train_prev_recurrent_states)
            else:
                action_distrib, batch_value = self.model(b_state)
            batch_action = chainer.cuda.to_cpu(action_distrib.sample().array)
            self.entropy_record.extend(
                chainer.cuda.to_cpu(action_distrib.entropy.array))
            self.value_record.extend(chainer.cuda.to_cpu((batch_value.array)))

        self.batch_last_state = list(batch_obs)
        self.batch_last_action = list(batch_action)

        return batch_action

    def batch_observe(self, batch_obs, batch_reward, batch_done, batch_reset):
        if self.recurrent:
            # Reset recurrent states when episodes end
            indices_that_ended = [
                i for i, (done, reset)
                in enumerate(zip(batch_done, batch_reset)) if done or reset]
            if indices_that_ended:
                self.test_recurrent_states =\
                    self.model.mask_recurrent_state_at(
                        self.test_recurrent_states, indices_that_ended)

    def batch_observe_and_train(self, batch_obs, batch_reward,
                                batch_done, batch_reset):

        for i, (state, action, reward, next_state, done, reset) in enumerate(zip(  # NOQA
            self.batch_last_state,
            self.batch_last_action,
            batch_reward,
            batch_obs,
            batch_done,
            batch_reset,
        )):
            if state is not None:
                assert action is not None
                transition = {
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state,
                    'nonterminal': 0.0 if done else 1.0,
                }
                if self.recurrent:
                    transition['recurrent_state'] =\
                        self.model.get_recurrent_state_at(
                            self.train_prev_recurrent_states,
                            i, unwrap_variable=True)
                    transition['next_recurrent_state'] =\
                        self.model.get_recurrent_state_at(
                            self.train_recurrent_states,
                            i, unwrap_variable=True)
                self.batch_last_episode[i].append(transition)
            if done or reset:
                assert self.batch_last_episode[i]
                self.memory.append(self.batch_last_episode[i])
                self.batch_last_episode[i] = []
            self.batch_last_state[i] = None
            self.batch_last_action[i] = None

        self.train_prev_recurrent_states = None

        if self.recurrent:
            # Reset recurrent states when episodes end
            indices_that_ended = [
                i for i, (done, reset)
                in enumerate(zip(batch_done, batch_reset)) if done or reset]
            if indices_that_ended:
                self.train_recurrent_states =\
                    self.model.mask_recurrent_state_at(
                        self.train_recurrent_states, indices_that_ended)

        self._update_if_dataset_is_ready()

    def get_statistics(self):
        return [
            ('average_value', _mean_or_nan(self.value_record)),
            ('average_entropy', _mean_or_nan(self.entropy_record)),
            ('average_kl', _mean_or_nan(self.kl_record)),
            ('average_policy_step_size',_mean_or_nan(self.policy_step_size_record)),
            ('explained_variance', self.explained_variance),
        ]