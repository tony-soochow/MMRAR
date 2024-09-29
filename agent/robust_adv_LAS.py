import collections
import itertools
import random

import chainer
from chainer import cuda
import chainer.functions as F
import numpy as np

from chainerrl import agent
from chainerrl.misc.batch_states import batch_states
from chainerrl.misc.copy_param import synchronize_parameters
from chainerrl.replay_buffer import ReplayUpdater, batch_experiences
import copy

import norms
import cupy as cp


def _mean_or_nan(xs):
    """Return its mean a non-empty sequence, numpy.nan for a empty one."""
    return np.mean(xs) if xs else np.nan


def _elementwise_clip(x, x_min, x_max):
    """Elementwise clipping

    Note: chainer.functions.clip supports clipping to constant intervals
    """
    return F.minimum(F.maximum(x, x_min), x_max)


def _add_advantage_and_value_target_to_episode(episode, gamma, lambd):
    """Add advantage and value target values to an episode."""
    adv = 0.0
    for transition in reversed(episode):
        td_err = (
                transition['penalized_reward']
                + (gamma * transition['nonterminal'] * transition['next_v_pred'])
                - transition['v_pred']
        )
        adv = td_err + gamma * lambd * adv
        transition['adv'] = adv
        transition['v_teacher'] = adv + transition['v_pred']


def _add_advantage_and_value_target_to_episodes(episodes, gamma, lambd):
    """Add advantage and value target values to a list of episodes."""
    for episode in episodes:
        _add_advantage_and_value_target_to_episode(
            episode, gamma=gamma, lambd=lambd)


def _add_log_prob_and_value_to_episodes_recurrent(
        episodes,
        model,
        phi,
        batch_states,
        obs_normalizer,
):
    xp = model.xp

    # Prepare data for a recurrent model
    seqs_states = []
    seqs_next_states = []
    for ep in episodes:
        states = batch_states(
            [transition['state'] for transition in ep], xp, phi)
        next_states = batch_states(
            [transition['next_state'] for transition in ep], xp, phi)
        if obs_normalizer:
            states = obs_normalizer(states, update=False)
            next_states = obs_normalizer(next_states, update=False)
        seqs_states.append(states)
        seqs_next_states.append(next_states)

    flat_transitions = list(itertools.chain.from_iterable(episodes))

    # Predict values using a recurrent model
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        rs = model.concatenate_recurrent_states(
            [ep[0]['recurrent_state'] for ep in episodes])
        next_rs = model.concatenate_recurrent_states(
            [ep[0]['next_recurrent_state'] for ep in episodes])
        assert len(rs) == len(next_rs)

        (flat_distribs, flat_vs), _ = model.n_step_forward(
            seqs_states, recurrent_state=rs, output_mode='concat')
        (_, flat_next_vs), _ = model.n_step_forward(
            seqs_next_states, recurrent_state=next_rs, output_mode='concat')

        flat_actions = xp.array([b['action'] for b in flat_transitions])
        flat_log_probs = flat_distribs.log_prob(flat_actions)
        flat_log_probs = chainer.cuda.to_cpu(flat_log_probs.array)
        flat_vs = chainer.cuda.to_cpu(flat_vs.array)
        flat_next_vs = chainer.cuda.to_cpu(flat_next_vs.array)

    # Add predicted values to transitions
    for transition, log_prob, v, next_v in zip(flat_transitions,
                                               flat_log_probs,
                                               flat_vs,
                                               flat_next_vs):
        transition['log_prob'] = float(log_prob)
        transition['v_pred'] = float(v)
        transition['next_v_pred'] = float(next_v)


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

def find_min_point3(dataset,obs_normalizer,s,bounded_model,lr, clip, budget,epsilon,action_low, action_high,
                   xp,phi, model,batch_states,horizon,total_counter=50):

    states = batch_states([b['state'] for b in dataset], xp, phi)
    if obs_normalizer:
        states = obs_normalizer(states, update=False)
    envs=list([b['env'] for b in dataset])
    distribs, _ = model(states)
    actions = xp.array([b['action'] for b in dataset])
    Prewards=cp.zeros(len(dataset),dtype=np.float32)
    next_obss= [None] * len(dataset)
    horizon=2
    
    for i in range(len(dataset)):
        env=envs[i]
        current_horizon = horizon
        current_budget = budget
        done=False
        while not done and current_horizon>0:
            spy_t=0
            deltas=[]
            v_actions = []
            spy_done=False
            spy_env=copy.deepcopy(env)
            env_states=[]
            while not spy_done and spy_t < current_horizon:
                if current_horizon==horizon and spy_t==0:
                    env_states.append(dataset[i]['state'])
                    next_obss[i]=dataset[i]['state']
                    action=actions[i]
                    distrib=distribs[i]
                else:
                    env_states.append(spy_obs)
                    b_state = batch_states([spy_obs], xp, phi)
                    if obs_normalizer:
                        b_state = obs_normalizer(b_state, update=False)
                    with chainer.using_config('train', False), chainer.no_backprop_mode():
                        new_distrib, _ = model(b_state)
                        distrib=new_distrib[0]
                        action = chainer.cuda.to_cpu(distrib.sample().data)
                        action=cp.asarray(action)
                    
                mean=distrib.mean
                var=distrib.ln_var
                mean = cp.array(mean.data)
                std_dev = cp.array(cp.sqrt(cp.exp(var.data)))
                if clip is True:
                    action = cp.clip(action, action_low, action_high)
                    
                grad_a = compute_grad(action, mean, std_dev)
                adv_action = action - (lr * grad_a)

                grad_a = compute_grad(adv_action, mean, std_dev)
                adv_action_new = adv_action - (lr * grad_a)
                counter = 0
                while np.absolute(adv_action - adv_action_new).any() > epsilon and counter < total_counter:
                    # print('Optimizing')
                    adv_action = adv_action_new
                    grad_a = compute_grad(adv_action, mean, std_dev)
                    adv_action_new = adv_action - (lr * grad_a)
                    counter += 1
                    
                v_actions.append(list(map(float, action)))
                delta = adv_action_new - action
                deltas.append(list(map(float, delta)))
                spy_obs, _, spy_done, _ = spy_env.step(action.get())#get()返回一个与原始变量（GPU内存）相同数据的 NumPy 数据，该数据位于主机内存中
                spy_t += 1
                
            if s == 'l2':
                proj_spatial_delta = norms.l2_spatial_project(deltas, current_budget/len(deltas))
            elif s == 'l1':
                proj_spatial_delta = norms.l1_spatial_project2(deltas, current_budget/len(deltas))

            proj_action = v_actions + proj_spatial_delta
            proj_action = cp.clip(proj_action, action_low, action_high)
            Q_env_states = batch_states(env_states, xp, phi)
            v_actions=cp.array(v_actions,dtype=np.float32)
            proj_action=cp.array(proj_action,dtype=np.float32)

            original_reward = F.flatten(bounded_model(Q_env_states, v_actions))
            perturbed_rewrad = F.flatten(bounded_model(Q_env_states,proj_action))#此处不应该用剪枝后的动作
            loss_reward=F.relu(original_reward-perturbed_rewrad)
            if cp.all(loss_reward.data == 0):
                loss_reward+=0.1
            
            temporal_deltas=(loss_reward[0]/ cp.sum(loss_reward.array) * current_budget).array
                
            if s == 'l1':#Project each δt,k in Aadv on to look-ahead sequence of budgets computed in the previous step to get sequence [δ′ t,k δ′ t,k+1 . . . δ′ t,k+H ]
                spatial_delta=(
                    list(map(float, norms.l1_spatial_project2(deltas[0], temporal_deltas.item()))))
            elif s== 'l2':
                spatial_delta=(list(map(float, norms.l2_spatial_project(deltas[0], temporal_deltas.item()))))#spatial_deltas维度（10，4）
            proj_action_0 = v_actions[0] + cp.array(spatial_delta,dtype=np.float32)
            proj_action_0 = np.clip(proj_action_0, -1, 1)
            if current_horizon>1:
                spy_obs, r, done, _ = env.step(proj_action_0.get())
                Prewards[i]+=r
                next_obss[i]=spy_obs
            else:
                next_obss[i]=batch_states(next_obss[i], xp, phi)
                next_state=cp.reshape(next_obss[i], (1, -1))
                next_adv=cp.reshape(proj_action_0, (1, -1))
                a=(F.flatten(bounded_model(next_state,next_adv))).array.item()
                Prewards[i]+=a
            current_horizon = current_horizon - 1
            current_budget = current_budget-temporal_deltas.item()

    return chainer.Variable(Prewards)
def _add_log_prob_and_value_to_episodes(
        episodes,
        model,
        bounded_model,
        phi,
        batch_states,
        obs_normalizer,
        attack_learning_rate,
        attack_budget,
        attack_norms,
        attack_epsilon,
        constrained_facor,
        counter,
        penalty_term,
        action_low,
        action_high
):
    dataset = list(itertools.chain.from_iterable(episodes))
    xp = model.xp

    # Compute v_pred and next_v_pred
    states = batch_states([b['state'] for b in dataset], xp, phi)
    next_states = batch_states([b['next_state'] for b in dataset], xp, phi)

    if obs_normalizer:
        states = obs_normalizer(states, update=False)
        next_states = obs_normalizer(next_states, update=False)

    with chainer.using_config('train', False), chainer.no_backprop_mode():
        distribs, vs_pred = model(states)
        _, next_vs_pred = model(next_states)

        actions = xp.array([b['action'] for b in dataset])
        rewards = np.array([b['reward'] for b in dataset])

        env_states = batch_states([b["state"] for b in dataset], xp, phi)

        """Calculate Adversarial Attack"""
        horizon=2
        perturbed_rewrad=find_min_point3(dataset,obs_normalizer,attack_norms,bounded_model,attack_learning_rate, True, attack_budget,attack_epsilon,action_low, action_high,
                   xp,phi, model,batch_states,horizon,total_counter=50)
        
        
        
       

        original_reward = F.flatten(bounded_model(env_states, actions))
    
        # # 1-alpha)
        """robst method"""
        modified_reward = (original_reward - F.absolute(constrained_facor  * original_reward))
        transition_distance = F.relu(modified_reward - perturbed_rewrad)

        # policy_distance = F.maximum(policy_distance, xp.array(self.constrained_distance, dtype=cp.float32)).array
        constraint_distance = (transition_distance.array.get()) ** 2

        u = 1 + np.sqrt(1 / constraint_distance)

        penalty = np.nan_to_num(constraint_distance * u)

        penalized_rewards = rewards
        if penalty_term:
            penalized_rewards = rewards - penalty

        next_vs_pred = chainer.cuda.to_cpu(next_vs_pred.array.ravel())
        log_probs = chainer.cuda.to_cpu(distribs.log_prob(actions).array)
        vs_pred = chainer.cuda.to_cpu(vs_pred.array.ravel())

    for transition, log_prob, v_pred, next_v_pred, penalized_reward in zip(dataset,
                                                                           log_probs,
                                                                           vs_pred,
                                                                           next_vs_pred,
                                                                           penalized_rewards):
        transition['log_prob'] = log_prob
        transition['v_pred'] = v_pred
        transition['next_v_pred'] = next_v_pred
        transition['penalized_reward'] = penalized_reward


def _limit_sequence_length(sequences, max_len):
    assert max_len > 0
    new_sequences = []
    for sequence in sequences:
        while len(sequence) > max_len:
            new_sequences.append(
                sequence[:max_len])
            sequence = sequence[max_len:]
        assert 0 < len(sequence) <= max_len
        new_sequences.append(sequence)
    return new_sequences


def _yield_subset_of_sequences_with_fixed_number_of_items(
        sequences, n_items):
    assert n_items > 0
    stack = list(reversed(sequences))
    while stack:
        subset = []
        count = 0
        while count < n_items:
            sequence = stack.pop()
            subset.append(sequence)
            count += len(sequence)
        if count > n_items:
            # Split last sequence
            sequence_to_split = subset[-1]
            n_exceeds = count - n_items
            assert n_exceeds > 0
            subset[-1] = sequence_to_split[:-n_exceeds]
            stack.append(sequence_to_split[-n_exceeds:])
        assert sum(len(seq) for seq in subset) == n_items
        yield subset


def _compute_explained_variance(transitions):
    """Compute 1 - Var[return - v]/Var[return].

    This function computes the fraction of variance that value predictions can
    explain about returns.
    """
    t = np.array([tr['v_teacher'] for tr in transitions])
    y = np.array([tr['v_pred'] for tr in transitions])
    vart = np.var(t)
    if vart == 0:
        return np.nan
    else:
        return float(1 - np.var(t - y) / vart)


def _make_dataset_recurrent(
        episodes, model, phi, batch_states, obs_normalizer,
        gamma, lambd, max_recurrent_sequence_len):
    """Make a list of sequences with necessary information."""

    _add_log_prob_and_value_to_episodes_recurrent(
        episodes=episodes,
        model=model,
        phi=phi,
        batch_states=batch_states,
        obs_normalizer=obs_normalizer,
    )

    _add_advantage_and_value_target_to_episodes(
        episodes, gamma=gamma, lambd=lambd)

    if max_recurrent_sequence_len is not None:
        dataset = _limit_sequence_length(
            episodes, max_recurrent_sequence_len)
    else:
        dataset = list(episodes)

    return dataset


def _make_dataset(
        episodes, model, bounded_model, phi, batch_states, obs_normalizer, gamma, lambd,
        attack_learning_rate,  # constrain
        attack_budget,
        attack_norms,
        attack_epsilon,
        constrained_facor,
        counter,
        penalty_term,
        action_low,
        action_high
):
    """Make a list of transitions with necessary information."""

    _add_log_prob_and_value_to_episodes(
        episodes=episodes,
        model=model,
        bounded_model=bounded_model,
        phi=phi,
        batch_states=batch_states,
        obs_normalizer=obs_normalizer,
        attack_learning_rate=attack_learning_rate,
        attack_budget=attack_budget,
        attack_norms=attack_norms,
        attack_epsilon=attack_epsilon,
        constrained_facor=constrained_facor,
        counter=counter,
        penalty_term=penalty_term,
        action_low=action_low,
        action_high=action_high
    )

    _add_advantage_and_value_target_to_episodes(
        episodes, gamma=gamma, lambd=lambd)

    return list(itertools.chain.from_iterable(episodes))


class PPO(agent.AttributeSavingMixin, agent.BatchAgent):
    """Proximal Policy Optimization

    See https://arxiv.org/abs/1707.06347

    Args:
        model (A3CModel): Model to train.  Recurrent models are not supported.
            state s  |->  (pi(s, _), v(s))
        optimizer (chainer.Optimizer): Optimizer used to train the model
        gpu (int): GPU device id if not None nor negative
        gamma (float): Discount factor [0, 1]
        lambd (float): Lambda-return factor [0, 1]
        phi (callable): Feature extractor function
        value_func_coef (float): Weight coefficient for loss of
            value function (0, inf)
        entropy_coef (float): Weight coefficient for entropy bonus [0, inf)
        update_interval (int): Model update interval in step
        minibatch_size (int): Minibatch size
        epochs (int): Training epochs in an update
        clip_eps (float): Epsilon for pessimistic clipping of likelihood ratio
            to update policy
        clip_eps_vf (float): Epsilon for pessimistic clipping of value
            to update value function. If it is ``None``, value function is not
            clipped on updates.
        standardize_advantages (bool): Use standardized advantages on updates
        recurrent (bool): If set to True, `model` is assumed to implement
            `chainerrl.links.StatelessRecurrent` and update in a recurrent
            manner.
        max_recurrent_sequence_len (int): Maximum length of consecutive
            sequences of transitions in a minibatch for updatig the model.
            This value is used only when `recurrent` is True. A smaller value
            will encourage a minibatch to contain more and shorter sequences.
        act_deterministically (bool): If set to True, choose most probable
            actions in the act method instead of sampling from distributions.
        value_stats_window (int): Window size used to compute statistics
            of value predictions.
        entropy_stats_window (int): Window size used to compute statistics
            of entropy of action distributions.
        value_loss_stats_window (int): Window size used to compute statistics
            of loss values regarding the value function.
        policy_loss_stats_window (int): Window size used to compute statistics
            of loss values regarding the policy.

    Statistics:
        average_value: Average of value predictions on non-terminal states.
            It's updated on (batch_)act_and_train.
        average_entropy: Average of entropy of action distributions on
            non-terminal states. It's updated on (batch_)act_and_train.
        average_value_loss: Average of losses regarding the value function.
            It's updated after the model is updated.
        average_policy_loss: Average of losses regarding the policy.
            It's updated after the model is updated.
        n_updates: Number of model updates so far.
        explained_variance: Explained variance computed from the last batch.
    """

    saved_attributes = ['model', 'optimizer', 'obs_normalizer']

    def __init__(self,
                 model,
                 optimizer,
                 replay_buffer,  # for dynamics model
                 q_func,
                 q_func_optimizer,
                 obs_normalizer=None,
                 gpu=None,
                 gamma=0.99,
                 lambd=0.95,
                 phi=lambda x: x,
                 value_func_coef=1.0,
                 entropy_coef=0.01,
                 update_interval=2048,
                 minibatch_size=64,
                 epochs=10,
                 clip_eps=0.2,
                 clip_eps_vf=None,
                 standardize_advantages=True,
                 batch_states=batch_states,
                 recurrent=False,
                 max_recurrent_sequence_len=None,
                 act_deterministically=False,
                 value_stats_window=1000,
                 entropy_stats_window=1000,
                 value_loss_stats_window=100,
                 policy_loss_stats_window=100,
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
                 ):
        self.model = model
        self.optimizer = optimizer
        self.obs_normalizer = obs_normalizer
        self.q_func = q_func
        self.q_func_optimizer = q_func_optimizer

        self.replay_buffer = replay_buffer
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

        self.gamma = gamma
        self.lambd = lambd
        self.phi = phi
        self.value_func_coef = value_func_coef
        self.entropy_coef = entropy_coef
        self.update_interval = update_interval
        self.minibatch_size = minibatch_size
        self.epochs = epochs
        self.clip_eps = clip_eps
        self.clip_eps_vf = clip_eps_vf
        self.standardize_advantages = standardize_advantages
        self.batch_states = batch_states
        self.recurrent = recurrent
        self.max_recurrent_sequence_len = max_recurrent_sequence_len
        self.act_deterministically = act_deterministically

        self.xp = self.model.xp

        # Contains episodes used for next update iteration
        self.memory = []

        # Contains transitions of the last episode not moved to self.memory yet
        self.last_episode = []
        self.last_state = None
        self.last_action = None

        # Batch versions of last_episode, last_state, and last_action
        self.batch_last_episode = None
        self.batch_last_state = None
        self.batch_last_action = None
        
        self.batch_last_env = None

        # Recurrent states of the model
        self.train_recurrent_states = None
        self.train_prev_recurrent_states = None
        self.test_recurrent_states = None

        self.value_record = collections.deque(maxlen=value_stats_window)
        self.entropy_record = collections.deque(maxlen=entropy_stats_window)
        self.value_loss_record = collections.deque(
            maxlen=value_loss_stats_window)
        self.policy_loss_record = collections.deque(
            maxlen=policy_loss_stats_window)
        self.explained_variance = np.nan

        self.q_func_loss_record = collections.deque(maxlen=value_loss_stats_window)

        self.t = 0
        self.attack_learning_rate = attack_learning_rate
        self.attack_epsilon = attack_epsilon
        self.attack_norms = attack_norms
        self.counter = counter
        self.attack_budget = attack_budget
        self.constrained_facor = constrained_distance
        self.action_low = action_low
        self.action_high = action_high
        self.explained_variance = np.nan
        self.initial_exploration_steps = initial_exploration_steps

        self.target_q_func = copy.deepcopy(self.q_func)

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
                
        self.batch_last_env = None
        

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

    def _update_obs_normalizer(self, dataset):
        assert self.obs_normalizer
        states = self.batch_states(
            [b['state'] for b in dataset], self.obs_normalizer.xp, self.phi)
        self.obs_normalizer.experience(states)

    def _update(self, dataset):
        """Update both the policy and the value function."""

        if self.obs_normalizer:
            self._update_obs_normalizer(dataset)

        xp = self.model.xp

        assert 'state' in dataset[0]
        assert 'v_teacher' in dataset[0]

        dataset_iter = chainer.iterators.SerialIterator(
            dataset, self.minibatch_size)

        if self.standardize_advantages:
            all_advs = xp.array([b['adv'] for b in dataset])
            mean_advs = xp.mean(all_advs)
            std_advs = xp.std(all_advs)

        while dataset_iter.epoch < self.epochs:
            batch = dataset_iter.__next__()
            states = self.batch_states(
                [b['state'] for b in batch], xp, self.phi)
            if self.obs_normalizer:
                states = self.obs_normalizer(states, update=False)
            actions = xp.array([b['action'] for b in batch])
            distribs, vs_pred = self.model(states)

            advs = xp.array([b['adv'] for b in batch], dtype=xp.float32)
            if self.standardize_advantages:
                advs = (advs - mean_advs) / (std_advs + 1e-8)

            log_probs_old = xp.array([b['log_prob']
                                      for b in batch], dtype=xp.float32)
            vs_pred_old = xp.array([b['v_pred']
                                    for b in batch], dtype=xp.float32)
            vs_teacher = xp.array([b['v_teacher']
                                   for b in batch], dtype=xp.float32)
            # Same shape as vs_pred: (batch_size, 1)
            vs_pred_old = vs_pred_old[..., None]
            vs_teacher = vs_teacher[..., None]

            self.optimizer.update(
                self._lossfun,
                distribs.entropy, vs_pred, distribs.log_prob(actions),
                vs_pred_old=vs_pred_old,
                log_probs_old=log_probs_old,
                advs=advs,
                vs_teacher=vs_teacher,
            )

    def _update_once_recurrent(
            self, episodes, mean_advs, std_advs):

        assert std_advs is None or std_advs > 0

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

        flat_actions = xp.array(
            [transition['action'] for transition in flat_transitions])
        flat_advs = xp.array(
            [transition['adv'] for transition in flat_transitions],
            dtype=np.float32)
        if self.standardize_advantages:
            flat_advs = (flat_advs - mean_advs) / (std_advs + 1e-8)
        flat_log_probs_old = xp.array(
            [transition['log_prob'] for transition in flat_transitions],
            dtype=np.float32)
        flat_vs_pred_old = xp.array(
            [[transition['v_pred']] for transition in flat_transitions],
            dtype=np.float32)
        flat_vs_teacher = xp.array(
            [[transition['v_teacher']] for transition in flat_transitions],
            dtype=np.float32)

        with chainer.using_config('train', False),\
             chainer.no_backprop_mode():
            rs = self.model.concatenate_recurrent_states(
                [ep[0]['recurrent_state'] for ep in episodes])

        (flat_distribs, flat_vs_pred), _ = self.model.n_step_forward(
            seqs_states, recurrent_state=rs, output_mode='concat')
        flat_log_probs = flat_distribs.log_prob(flat_actions)
        flat_entropy = flat_distribs.entropy

        self.optimizer.update(
            self._lossfun,
            entropy=flat_entropy,
            vs_pred=flat_vs_pred,
            log_probs=flat_log_probs,
            vs_pred_old=flat_vs_pred_old,
            log_probs_old=flat_log_probs_old,
            advs=flat_advs,
            vs_teacher=flat_vs_teacher,
        )

    def _update_recurrent(self, dataset):
        """Update both the policy and the value function."""

        flat_dataset = list(itertools.chain.from_iterable(dataset))
        if self.obs_normalizer:
            self._update_obs_normalizer(flat_dataset)

        xp = self.model.xp

        assert 'state' in flat_dataset[0]
        assert 'v_teacher' in flat_dataset[0]

        if self.standardize_advantages:
            all_advs = xp.array([b['adv'] for b in flat_dataset])
            mean_advs = xp.mean(all_advs)
            std_advs = xp.std(all_advs)
        else:
            mean_advs = None
            std_advs = None

        for _ in range(self.epochs):
            random.shuffle(dataset)
            for minibatch in _yield_subset_of_sequences_with_fixed_number_of_items(  # NOQA
                    dataset, self.minibatch_size):
                self._update_once_recurrent(minibatch, mean_advs, std_advs)

    def _lossfun(self,
                 entropy, vs_pred, log_probs,
                 vs_pred_old, log_probs_old,
                 advs, vs_teacher):

        prob_ratio = F.exp(log_probs - log_probs_old)

        loss_policy = - F.mean(F.minimum(
            prob_ratio * advs,
            F.clip(prob_ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advs))

        if self.clip_eps_vf is None:
            loss_value_func = F.mean_squared_error(vs_pred, vs_teacher)
        else:
            loss_value_func = F.mean(F.maximum(
                F.square(vs_pred - vs_teacher),
                F.square(_elementwise_clip(vs_pred,
                                           vs_pred_old - self.clip_eps_vf,
                                           vs_pred_old + self.clip_eps_vf)
                         - vs_teacher)
            ))
        loss_entropy = -F.mean(entropy)

        self.value_loss_record.append(float(loss_value_func.array))
        self.policy_loss_record.append(float(loss_policy.array))

        loss = (
                loss_policy
                + self.value_func_coef * loss_value_func
                + self.entropy_coef * loss_entropy
        )

        return loss

    def act_and_train(self, obs, reward):

        if self.last_state is not None:
            transition = {
                'state': self.last_state,
                'action': self.last_action,
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

        return action

    def act(self, obs):
        xp = self.xp
        b_state = self.batch_states([obs], xp, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            if self.recurrent:
                (action_distrib, _), self.test_recurrent_states =\
                    self.model(b_state, self.test_recurrent_states)
            else:
                action_distrib, _ = self.model(b_state)
            if self.act_deterministically:
                action = chainer.cuda.to_cpu(
                    action_distrib.most_probable.array)[0]
            else:
                action = chainer.cuda.to_cpu(
                    action_distrib.sample().array)[0]

        return action

    def stop_episode_and_train(self, state, reward, done=False):

        assert self.last_state is not None
        transition = {
            'state': self.last_state,
            'action': self.last_action,
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

    def batch_act_and_train(self, batch_obs,batch_env):
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
        
        self.batch_last_env = batch_env

        min_actions = adversarl_attack1(batch_action, action_distrib.mean, action_distrib.ln_var,
                                        self.attack_learning_rate, self.attack_budget, self.attack_norms,
                                        self.attack_epsilon, self.counter,
                                        self.action_low, self.action_high)
        self.batch_last_obs = list(batch_obs)
        self.batch_last_act = list(batch_action)
        self.batch_last_per_act = list(min_actions)

        return min_actions

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
        for i in range(len(batch_obs)):
            self.t += 1
            if self.batch_last_obs[i] is not None:
                assert self.batch_last_act[i] is not None
                # Add a transition to the replay buffer
                self.replay_buffer.append(
                    state=self.batch_last_obs[i],
                    action=self.batch_last_act[i],
                    perturbed_action=self.batch_last_per_act[i],
                    reward=batch_reward[i],
                    next_state=batch_obs[i],
                    next_action=None,
                    is_state_terminal=batch_done[i],
                    env_id=i,
                )
                if batch_reset[i] or batch_done[i]:
                    self.batch_last_obs[i] = None
                    self.replay_buffer.stop_current_episode(env_id=i)
            self.replay_updater.update_if_necessary(self.t)

        for i, (state, action,env, reward, next_state, done, reset) in enumerate(zip(  # NOQA
            self.batch_last_state,
            self.batch_last_action,
            [self.batch_last_env],
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
                    'env':env,
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
            
            self.batch_last_env= None

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
            ('average_value_loss', _mean_or_nan(self.value_loss_record)),
            ('average_policy_loss', _mean_or_nan(self.policy_loss_record)),
            ('n_updates', self.optimizer.t),
            ('explained_variance', self.explained_variance),
            ('average_q_func_loss', _mean_or_nan(self.q_func_loss_record)),
        ]
