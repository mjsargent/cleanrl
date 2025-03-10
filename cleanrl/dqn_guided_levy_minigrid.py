# https://github.com/facebookresearch/torchbeast/blob/master/torchbeast/core/environment.py
import numpy as np
from collections import deque
import gym
import gym_minigrid
from gym import spaces
from gym_minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
import cv2
cv2.ocl.setUseOpenCL(False)

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=((shp[0] * k,)+shp[1:]), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]

def wrap_atari(env, max_episode_steps=None):
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)

    assert max_episode_steps is None

    return env

def wrap_minigrid(env, max_episode_steps=None):
    env = ...

    return env

class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to channels x weight x height
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, observation):
        return np.transpose(observation, axes=(2, 0, 1))

def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    env = ImageToPyTorch(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env

# Reference: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import argparse
from distutils.util import strtobool
import collections
import numpy as np
import gym
from gym.wrappers import TimeLimit, Monitor
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
from scipy.stats import norm
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN agent')
    # Common arguments
    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym_id', type=str, default="MiniGrid-FourRooms-v0",
                        help='the id of the gym environment')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=2,
                        help='seed of the experiment')
    parser.add_argument('--total_timesteps', type=int, default=10000000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch_deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod_mode', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture_video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb_project_name', type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    
    # Algorithm specific arguments
    parser.add_argument('--buffer_size', type=int, default=1000000,
                         help='the replay memory buffer size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--target_network_frequency', type=int, default=1000,
                        help="the timesteps it takes to update the target network")
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--batch_size', type=int, default=32,
                        help="the batch size of sample from the reply memory")
    parser.add_argument('--start_e', type=float, default=1.,
                        help="the starting epsilon for exploration")
    parser.add_argument('--end_e', type=float, default=0.02,
                        help="the ending epsilon for exploration")
    parser.add_argument('--exploration_fraction', type=float, default=0.10,
                        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument('--learning_starts', type=int, default=80000,
                        help="timestep to start learning")
    parser.add_argument('--train_frequency', type=int, default=4,
                        help="the frequency of training")
    parser.add_argument('--mu_net_coeff', type=float, default=3,
                        help="coefficent used for determining the init of mu net")
    parser.add_argument("--n_steps", type=int, default=3, 
                              help = "number of steps used in nstep return")
    parser.add_argument("--fully_observable", type=bool, default=False,
                        help="use the fully observable env wrapper")
    parser.add_argument("--clip_n", type = int, default = 100, 
                        help="maximum number of action repeats")
    parser.add_argument('--eval_frequency', type=int, default=5000,
                        help="the frequency of noiseless eval")
    # n network losses
    parser.add_argument("--obs_target", type=bool, default=False, help="use norm of the observations as a target")
    parser.add_argument("--latent_target", type=bool, default=True, help= "use latent norm as a target")
    parser.add_argument("--n_weighted", type=bool, default=False, help= "use weighted target n")
    parser.add_argument("--n_argmax",type = bool,default=False,  help= "take the argmax of latent loss as a target")
    parser.add_argument("--n_sign_changes", type=bool, default=False, help= "use sign changes to choose an n as a target")
    parser.add_argument("--value_loss", type=bool, default=True, help= "use value as a target")

    parser.add_argument("--value_conditioning", type=bool, default=False, help= "condition the action repeat on the current value estimate")
    parser.add_argument("--n_loss_weighting", type=float, default=0.5, help= "weightings of the value based and embedding based losses")
    parser.add_argument("--discount_latent_embedding", type=bool, default=1, help= "discount along the trajectories")

    
    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

# TRY NOT TO MODIFY: setup the environment
experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
        '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
if args.prod_mode:
    import wandb
    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=experiment_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"/tmp/{experiment_name}")

# TRY NOT TO MODIFY: seeding
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
env = gym.make(args.gym_id)
#env = wrap_atari(env)


if args.fully_observable:
    env = FullyObsWrapper(env)
    print("Fully Observable Obs space: ", env.observation_space)

env = ImgObsWrapper(env)
print("Obs space: ", env.observation_space)
#env = gym.wrappers.RecordEpisodeStatistics(env) # records episode reward in `info['episode']['r']`

if args.capture_video:
    env = Monitor(env, f'videos/{experiment_name}')

#env = wrap_deepmind(
#    env,
#    clip_rewards=True,
#    frame_stack=True,
#    scale=False,
#)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
env.seed(args.seed)
env.action_space.seed(args.seed)
env.observation_space.seed(args.seed)
# respect the default timelimit
assert isinstance(env.action_space, Discrete), "only discrete action space is supported"

# modified from https://github.com/seungeunrho/minimalRL/blob/master/dqn.py#
class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)

        return np.array(s_lst), np.array(a_lst), \
               np.array(r_lst), np.array(s_prime_lst), \
               np.array(done_mask_lst)

class ReplayBufferNStep():
    def __init__(self, buffer_limit, nsteps, gamma):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.nstep_buffer = [] 
        self.nsteps = nsteps
        self.gamma = gamma
    def put(self, transition):
        self.nstep_buffer.append(transition)

        if len(self.nstep_buffer ) < self.nsteps:
            return
        
        R = sum(self.nstep_buffer[i][2]*(self.gamma**i) for i in range(self.nsteps))
        state, action, _, _, _ = self.nstep_buffer.pop(0)

        self.buffer.append((state, action, R, transition[3], transition[4]))
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)

        return np.array(s_lst), np.array(a_lst), \
               np.array(r_lst), np.array(s_prime_lst), \
               np.array(done_mask_lst)

    def finish_nstep(self): 
        last_obs = self.nstep_buffer[3] if len(self.nstep_buffer) == 1 else self.nstep_buffer[-1][3]
        while len(self.nstep_buffer) > 0:
            R = sum([self.nstep_buffer[i][2]*(self.gamma**i) for i in range(len(self.nstep_buffer))])
            state, action, _, _, _= self.nstep_buffer.pop(0)

            self.buffer.append((state, action, R ,last_obs, True))

class ReplayBufferNStepVariable():
    def __init__(self, buffer_limit, gamma):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.nstep_buffer = [] 
        self.nsteps = None
        self.gamma = gamma

    def put(self, transition, n):
        self.nstep_buffer.append(transition)

        if self.nsteps == None:
            self.nsteps = n

        if len(self.nstep_buffer ) < self.nsteps:
            return
        
        # once we hit n, drain the buffer and take a new n
        while len(self.nstep_buffer) > 0:

            R = sum(self.nstep_buffer[i][2]*(self.gamma**i) for i in range(len(self.nstep_buffer)))
            state, action, _, _, _ = self.nstep_buffer.pop(0)

            self.buffer.append((state, action, R, transition[3], transition[4]))

        self.nsteps = n
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)

        return np.array(s_lst), np.array(a_lst), \
               np.array(r_lst), np.array(s_prime_lst), \
               np.array(done_mask_lst)

    def finish_nstep(self): 
        if self.nstep_buffer == []:
            return

        last_obs = self.nstep_buffer[0][3] if len(self.nstep_buffer) == 1 else self.nstep_buffer[-1][3]
        while len(self.nstep_buffer) > 0:
            R = sum([self.nstep_buffer[i][2]*(self.gamma**i) for i in range(len(self.nstep_buffer))])
            state, action, _, _, _= self.nstep_buffer.pop(0)

            self.buffer.append((state, action, R ,last_obs, True))

class ReplayBufferNStepLevy():
    def __init__(self, buffer_limit, gamma):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.nstep_buffer = [] 
        self.nsteps = None
        self.gamma = gamma

    def put(self, transition, n):
        self.nstep_buffer.append(transition)

        if self.nsteps == None:
            self.nsteps = n

        if len(self.nstep_buffer) < self.nsteps:
            return
        
        # once we hit n, drain the buffer and take a new n
        while len(self.nstep_buffer) > 0:

            len_traj = len(self.nstep_buffer)
            R = sum(self.nstep_buffer[i][2]*(self.gamma**i) for i in range(len(self.nstep_buffer)))
            all_states = [tran[0] for tran in self.nstep_buffer]
            all_next_states = [tran[3] for tran in self.nstep_buffer]
            first_state , action, _, _, _ = self.nstep_buffer.pop(0)
            # keep seperatre cache of states and next stats for each drain; do a seperate pass of the next obs
            self.buffer.append((first_state, action, R,transition[3] ,transition[4], len_traj, np.stack(all_states, axis=0), np.stack(all_next_states, axis=0)))

        self.nsteps = n

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst, len_traj_lst,  all_states_lst, all_next_states_lst  = [], [], [], [], [], [], [], []
        for transition in mini_batch:
            s, a, r, s_prime, done_mask, len_traj,  all_states, all_next_states= transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            len_traj_lst.append(len_traj)
            done_mask_lst.append(done_mask)
            all_states_lst.append(all_states) 

            all_next_states_lst.append(all_next_states) 

        return np.array(s_lst), np.array(a_lst), \
               np.array(r_lst), np.array(s_prime_lst), \
               np.array(done_mask_lst), \
               np.array(len_traj_lst), \
               np.concatenate(all_states_lst, axis=0), \
               np.concatenate(all_next_states_lst, axis=0)


    def finish_nstep(self): 
        if self.nstep_buffer == []:
            return

        last_obs = self.nstep_buffer[0][3] if len(self.nstep_buffer) == 1 else self.nstep_buffer[-1][3]
        while len(self.nstep_buffer) > 0:
            R = sum([self.nstep_buffer[i][2]*(self.gamma**i) for i in range(len(self.nstep_buffer))])

            len_traj = len(self.nstep_buffer)

            all_states = [tran[0] for tran in self.nstep_buffer]
            all_next_states = [tran[3] for tran in self.nstep_buffer]

            first_state, action, _, _, _= self.nstep_buffer.pop(0)
            self.buffer.append((first_state, action, R, last_obs ,True, len_traj, np.stack(all_states, axis=0), np.stack(all_next_states, axis=0)))


class ReplayBufferActRepeat():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, n_lst, s_prime_lst, n_nxt_lst,  done_mask_lst = [], [], [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, n, s_prime, n_nxt, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)

            n_cloned = n.clone()
            n_lst.append(n_cloned)

            s_prime_lst.append(s_prime)

            n_nxt_cloned = n_nxt.clone().detach()
            n_nxt_lst.append(n_nxt_cloned)

            done_mask_lst.append(done_mask)

        return np.array(s_lst), np.array(a_lst), \
               np.array(r_lst), n_lst, \
               np.array(s_prime_lst), n_nxt_lst, \
               np.array(done_mask_lst)


# ALGO LOGIC: initialize agent here:
class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

class QNetwork(nn.Module):
    def __init__(self, env, frames=3):
        super(QNetwork, self).__init__()
        n = env.observation_space.shape[0]
        m = env.observation_space.shape[1]
        self.linear_embedding_size = 64* ((n-1)//2 -2) * ((m-1)//2 - 2)

        self.network = nn.Sequential(
            Scale(1/255),
            nn.Conv2d(frames, 16, (2,2)),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 32,(2,2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2,2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.linear_embedding_size, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n)
        )

    def forward(self, x, device):
        x = np.swapaxes(x,1,3)

        x = torch.Tensor(x).to(device)
        n = random.randint(3, 5)

        return self.network(x), n

class LevyHead(nn.Module):
    def __init__(self):
        super(LevyHead, self).__init__()

    def forward(self, mu, scale, use_noise = True):
        batch_size = mu.shape[0] 
        noise = torch.tensor((norm.ppf(1-np.random.rand(batch_size))**-2), dtype=torch.float, device=mu.device, requires_grad=False) 
        noise = noise.unsqueeze(1)  
        scale = torch.clamp(scale, min=0, max=0.1)                                                                           
        n = mu + scale * int(use_noise)*noise 
        return n 

def layer_init(layer,std=np.sqrt(2),bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight,std)
    torch.nn.init.constant_(layer.bias,bias_const)
    return layer

class QNetworkN(nn.Module):
    def __init__(self, env, frames=3, condition=False):
        super(QNetworkN, self).__init__()
        n = env.observation_space.shape[0]
        m = env.observation_space.shape[1]
        
        self.linear_embedding_size = 64* ((n-1)//2 -2) * ((m-1)//2 - 2)
        print(self.linear_embedding_size)
        self.condition = condition
        self.latent_dim = 512 + env.action_space.n if self.condition else 512
        self.embedding = nn.Sequential(
            Scale(1/255),
            layer_init(nn.Conv2d(frames, 16, (2,2))),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            layer_init(nn.Conv2d(16, 32,(2,2))),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, (2,2))),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(self.linear_embedding_size, self.latent_dim)),
            nn.ReLU()
        )
        self.q_head =layer_init( nn.Linear(512, env.action_space.n))

        self.mu_head = nn.Sequential(nn.Linear(self.latent_dim, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 1),
                                     #nn.ReLU() 
                                     )

        self.scale_head = nn.Sequential(layer_init(nn.Linear(self.latent_dim, 64)),
                                     nn.Tanh(),
                                     layer_init(nn.Linear(64, 1)),
                                     #nn.ReLU() 
                                     )

        self.levy_head = LevyHead()

        self.n_head = layer_init(nn.Linear(self.latent_dim, 1))

    def forward(self, x, device, use_noise=True):
        x = np.swapaxes(x,1,3)

        x = torch.Tensor(x).to(device)
        if self.condition:
            z = self.embedding(x)
            q = self.q_head(z)

            z_n = z.clone().detach()
            # detach q as well?
            z_n = torch.cat([z_n,q], axis = 1)
            mu = self.mu_head(z_n)
            scale = self.scale_head(z_n)
            n = self.levy_head(mu, scale)

        else:

            z = self.embedding(x)
            z_n = z.clone().detach()

            mu = self.mu_head(z_n)
            scale = self.scale_head(z_n)
            n = self.levy_head(mu, scale, use_noise)

            q = self.q_head(z)

        return q, n, mu, scale, z_n


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope =  (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

class LevySampler(nn.Module):
    def __init__(self, mu_max, scale_max, mu_min = 1, scale_min=1e-5):
        super(LevySampler, self).__init__()
        self.mu_min = mu_min
        self.scale_min = scale_min
        self.mu_max = mu_max
        self.scale_max = scale_max

    def forward(self, mu, scale, n_prev):
        # reparam trick - use Gaussian sampling of Levy dist
        # http://www.randomservices.org/random/special/Levy.html
        # mu + scale * (cumInvNorm(1 - Uniform)^-2)

        # use nonzero elements to create a batched mask 
        batch_size = mu.shape[0]
        noise = torch.tensor((norm.ppf(1-np.random.rand(batch_size))**-2), dtype=torch.float, device=mu.device)
        noise = noise.unsqueeze(1)
        mu = torch.clamp(mu, min=self.mu_min, max=self.mu_max)
        scale = torch.clamp(scale, min=self.scale_min, max=self.scale_max)
        n = mu + scale*noise
        mask = torch.tensor([int(i > 0) for i in n_prev],device = mu.device)
        mask = mask.unsqueeze(1)
        n = n_prev*mask + n*(torch.ones_like(mask)-mask)
        return n

class QNetworkGuidedLevy(nn.Module):
    def __init__(self, env, frames=3, mu_init=None):
        super(QNetworkGuidedLevy, self).__init__()
        n = env.observation_space.shape[0]
        m = env.observation_space.shape[1]
        self.linear_embedding_size = 64* ((n-1)//2 -2) * ((m-1)//2 - 2) 
        print("Linear Embedding Size: ", self.linear_embedding_size)
        self.embedding = nn.Sequential(
            #Scale(1/255),
            nn.Conv2d(frames, 16, (2,2) ),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 32, (2,2)),
            nn.ReLU(),

            nn.Conv2d(32, 64,(2,2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.linear_embedding_size, 512),
            nn.ReLU()
        )
        self.q_head = nn.Linear(512 + 1, env.action_space.n)
        # init mu head such that initial expected mu > mu_min

        self.levy_mu_head = nn.Sequential(
                             nn.Linear(512, 64),
                             nn.Tanh(),
                             nn.Linear(64, 1),
                             nn.ReLU()
                             )
        if mu_init is not None:
            for m in self.levy_mu_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal(m.weight, mean=mu_init/512, std=0.1)
                    nn.init.constant(m.bias, 512/128)

        self.levy_scale_head = nn.Sequential(
                             nn.Linear(512, 64),
                             nn.Tanh(),
                             nn.Linear(64, 1)
                             #nn.ReLU()
                             )
        self.action_repeat_sampler = LevySampler(10, 100) # TODO add as argparse params

    # TODO concat (n, n_t) as option instead of jst n 
    # TODO choices for concat
    # n for all time steps, gradient does not flow in time but jsut from t to 0
    # tuple n, n_t
    # n - t', where t' is relative time in sub trajectory, could be better for generalisation
    # 
    def forward(self, x,n_prev, device, initiate = False):
        x = np.swapaxes(x,1,3)
        #x = x.transpose(1,3).transpose(2,3)
        # x = x.contiguous()

        x = torch.Tensor(x).to(device)
        z = self.embedding(x)
        if self.condition:
            pass 
        else:

            mu = self.levy_mu_head(z)
            scale = self.levy_scale_head(z)
            # will return sample only if n_prev is zeros
            n_out = self.action_repeat_sampler(mu, scale, n_prev)
            z = torch.cat([z, n_out], dim = 1)
            q = self.q_head(z)

        return q, z, n_out, mu, scale
 
class Sampler():
    def __init__(self, env, use_eps_greedy = True):
        self.use_eps_greedy = use_eps_greedy
        self.action_space = env.action_space
        self.on_traj = False
        self.current_traj_length = 0
        self.final_traj_length = 0 
        self.current_action = 0 

    def sample(self, network, obs, device, n, eps):
        # take in a network and observation
        # want to keep track of levy logic within sampler
        # return n logits and action 
        # need n for storing in replay buffer 

        if self.on_traj:
            # action is the current action but still need
            # to generate logits 
            action = self.current_action
            logits, z, n_out, mu, scale = network.forward(obs.reshape((1,)+obs.shape), n, device, initiate=False)
            #TODO decide if to store n locally
            self.current_traj_length = self.current_traj_length + 1
            if self.current_traj_length == self.final_traj_length:
                self.current_traj_length = 0 
                self.final_traj_length = 0
                self.on_traj = False
                n_out = torch.zeros((1,1))

        else:
            logits, z, n_out, mu, scale = network.forward(obs.reshape((1,)+obs.shape), n,  device, initiate=True)
            # the n here is the n we want to concat to future passes
            # clip, floor and store a detached version for logic
            self.final_traj_length = n.clone().detach().cpu().numpy()
            self.final_traj_length = np.floor(self.final_traj_length)
            action = torch.argmax(logits, dim=1).tolist()[0]
            if self.final_traj_length > 1:
                self.on_traj = True
                self.current_traj_length = 1


        if self.use_eps_greedy:
            rand_num = random.random()
            if rand_num < eps:
                action = self.action_space.sample() 

        return action, logits,z, n_out, mu, scale
    
rb = ReplayBufferNStepLevy(args.buffer_size,  args.gamma)
q_network = QNetworkN(env)
#q_network = nn.DataParallel(q_network)
q_network = q_network.to(device)

target_network = QNetworkN(env)
#target_network = nn.DataParallel(target_network)
target_network = target_network.to(device)

target_network.load_state_dict(q_network.state_dict())

sampler = Sampler(env)

optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
loss_fn = nn.MSELoss()
loss_fn_n = nn.L1Loss()

print(device.__repr__())
print(q_network)
print(f"Using {torch.cuda.device_count()} GPUS")

n_params = sum([p.numel() for p in q_network.parameters()])
writer.add_scalar("n_params", n_params)
print("Number of parameters:", n_params)


# TRY NOT TO MODIFY: start the game
obs = env.reset()
if args.gym_id == "MiniGrid-Empty-100x100-v0":
    env.max_steps = 5000
episode_reward = 0
n = torch.zeros((1,1))
on_levy = 0
n_tracked = []

# define scale targets
scale_tracked = []
scale_lower = 0.0
scale_higher = 0.1

eval_flag = False

for global_step in range(args.total_timesteps):
    # ALGO LOGIC: put action logic here
    epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction*args.total_timesteps, global_step)
    obs = np.array(obs)
    logits, n, mu, scale, _ = q_network.forward(obs.reshape((1,)+obs.shape), device)
    # levy action selection logic
    if on_levy == 0:
        on_levy = n.clone().detach().cpu().numpy()
        on_levy = np.floor(np.clip(on_levy,1, args.clip_n))
        
        n_tracked.append(int(on_levy))

        if random.random() < epsilon:
            current_levy_action = env.action_space.sample() 
        else:
            current_levy_action = torch.argmax(logits, dim=1).tolist()[0]
        scale_tracked.append(float(scale))#

    if on_levy > 0:
        action = current_levy_action
    else:
        action = torch.argmax(logits, dim=1).tolist()[0]

    on_levy = on_levy - 1 
    
    # take a step
    next_obs, reward, done, info = env.step(action)
    episode_reward += reward


    rb.put((obs, action, reward, next_obs, done), n)

    if global_step > args.learning_starts and global_step % args.train_frequency == 0:
        s_obs, s_actions, s_rewards, s_next_obses, s_dones,s_len_trajs, s_all_states, s_all_next_states = rb.sample(args.batch_size)
#        print(s_n)
        with torch.no_grad():
            
            # Get TD target zero indexing the output just returns the logits target_max = torch.max(target_network.forward(s_next_obses, device)[0], dim=1)[0] td_target = torch.Tensor(s_rewards).to(device) + args.gamma * target_max * (1 - torch.Tensor(s_dones).to(device))
            # get targets for n, either value based or latent embedding based             
            n_target = torch.zeros(len(s_len_trajs), requires_grad = False, device=device)
            if args.latent_target:
                all_z_old = q_network.forward(s_all_states,device)[4]
                all_targets, _,_,_, all_z_targets = target_network.forward(s_all_next_states,device)
                all_targets = torch.max(all_targets,dim=1)[0]
                len_traj_current = 0
                
                delta_z = all_z_targets-all_z_old 
                delta_z_norms = torch.norm(delta_z, 2, dim = 1)      
                value_loss_weighting = args.n_loss_weighting
                n_target = torch.zeros(len(s_len_trajs), requires_grad = False, device=device)
                
                
                for idx, len_traj in enumerate(s_len_trajs):
                    delta_z_norms_traj = delta_z_norms[len_traj_current:len_traj_current+len_traj] 
                    #TODO investigate if norm should be of nondiscounted norms or not
                    expected_z = delta_z_norms_traj.mean() 

                    # TODO check for off by one error
                    if args.discount_latent_embedding:
                        traj_discount = torch.tensor([args.gamma**i for i in range(len_traj)], requires_grad=False,device=device)
                        delta_z_norms_traj = delta_z_norms_traj * traj_discount

                    if args.n_weighted:
                        norm_factor = delta_z_norms_traj.sum() if len_traj > 0 else 1
                        weights =  delta_z_norms_traj / norm_factor if len_traj > 0 else 1
                        weighted_n = weights * torch.arange(0,len_traj,device=device)
                        n_target[idx] += (1-value_loss_weighting)*weighted_n.sum()

                    if args.n_argmax:
                        n_target[idx] += torch.argmax(expected_z - delta_z_norms_traj)  \
                            if len_traj > 1 else 0 \

                    if args.n_sign_changes:
                        z_expected_diff = expected_z - delta_z_norms_traj
                        #TODO potentually correct for symmetry
                        pos_actions =  (z_expected_diff <= 0).nonzero()
                        # safety net for rounding errors
                        if pos_actions.shape == (0,1):
                            n_target[idx] += (1-value_loss_weighting)*len_traj
                        else:
                            n_target[idx] += (1-value_loss_weighting)*pos_actions[-1].item() if len_traj > 1 else 0
                        
                    if args.value_loss:
                        sub_traj = all_targets[len_traj_current:len_traj_current+len_traj]
                        n_target[idx] += value_loss_weighting *torch.argmax(sub_traj)  \
                            if len_traj > 1 else 0 

                    len_traj_current+=len_traj
                
            else:

                all_targets = target_network.forward(s_all_next_states, device)[0] 

                # quick and dirty loop; I just can't get my head around doing this with torch indexing function rn
                len_traj_current = 0

                n_target = torch.zeros(len(s_len_trajs), requires_grad = False, device=device)

                for idx, len_traj in enumerate(s_len_trajs):
                # +1 or not to +1 - think about the floor func. if n_out = 1, on_levy = 2 therefor taking two actions on this levy therfore not +1
                    n_target[idx] = torch.argmax(all_targets[len_traj_current:len_traj_current+len_traj])  \
                        if len_traj > 1 else 0 \
                        
                    len_traj_current+=len_traj
                
        # need to write function here to get the argmax of these values for each sub trajectory
        # when storing n, does it need to be stored before being passed through
        old_val, _, old_mu, old_scale, _ = q_network.forward(s_obs, device)
        old_val = old_val.gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()

        len_traj_current = 0

        with torch.no_grad():
            scale_target = torch.where(old_mu.squeeze(1) < n_target, scale_higher, scale_lower)
        
        # calc losses
        loss = loss_fn(td_target, old_val)
        n_loss = loss_fn_n(n_target, old_mu.squeeze(1)) 
        scale_loss = loss_fn(scale_target, old_scale.squeeze(1))
        total_loss = loss + n_loss + scale_loss
        
        # logging
        if global_step % 100 == 0:
            writer.add_scalar("losses/td_loss", loss, global_step) 
            writer.add_scalar("losses/n_loss", n_loss, global_step)

            avg_n = np.mean(n_tracked)
            avg_scale = np.mean(scale_tracked)
            writer.add_scalar("charts/avg_n", avg_n, global_step)
            writer.add_scalar("charts/avg_scale", avg_scale, global_step)
            writer.add_scalar("charts/last_target_n", int(n_target[0].item()))
            n_tracked = []
            scale_tracked = []

        # optimize the model
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(list(q_network.parameters()), args.max_grad_norm)
        optimizer.step()
        
        # garbage collection
        td_target = None
        target_max = None
        loss = None
        old_val = None

        # update the target network
        if global_step % args.target_network_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())

    # give sign to eval when done with current training episode

    if global_step % args.eval_frequency == 0:
        eval_flag = True

    # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
    obs = next_obs
    if done:
        # important to note that because `EpisodicLifeEnv` wrapper is applied,
        # the real episode reward is actually the sum of episode reward of 5 lives
        print(f"global_step={global_step}, episode_reward={episode_reward}")
        writer.add_scalar("charts/episode_reward", episode_reward, global_step)
        writer.add_scalar("charts/epsilon", epsilon, global_step)
        obs, episode_reward = env.reset(), 0

        if args.gym_id == "MiniGrid-Empty-100x100-v0":
            env.max_steps = 5000
        
        n = torch.zeros((1,1))
        rb.finish_nstep()
        rb.nsteps = None
        on_levy = 0
        
        if eval_flag:
            eval_done = False 
            eval_episode_reward = 0
            n_tracked_eval = []
            mu_tracked_eval = []
            # do eval loop
            while not eval_done:
                logits, n, mu,_, _ = q_network.forward(obs.reshape((1,)+obs.shape), device, use_noise = False)
                 
                mu_tracked_eval.append(float(mu))
                # levy action selection logic
                if on_levy == 0:
                    on_levy = n.clone().detach().cpu().numpy()
                    on_levy = np.floor(np.clip(on_levy, 1, args.clip_n))
                    n_tracked_eval.append(on_levy)
                    current_levy_action = torch.argmax(logits, dim=1).tolist()[0]

                if on_levy > 0:
                    action = current_levy_action
                else:
                    action = torch.argmax(logits, dim=1).tolist()[0]

                on_levy = on_levy - 1 
                
                # take a step
                obs, reward, eval_done, info = env.step(action)
                eval_episode_reward += reward

            avg_n_eval = np.mean(n_tracked_eval)
            avg_mu_eval = np.mean(mu_tracked_eval)
            std_mu_eval = np.std(mu_tracked_eval)
            print(f"|| EVAL || global_step={global_step}, eval_episode_reward={eval_episode_reward}, avg_n={avg_n_eval},avg_mu={avg_mu_eval}, std_mu={std_mu_eval}")
            writer.add_scalar("charts/eval_episode_reward", eval_episode_reward, global_step)
            writer.add_scalar("charts/eval_avg_n", avg_n_eval, global_step)
            writer.add_scalar("charts/eval_avg_mu", avg_mu_eval, global_step)
            writer.add_scalar("charts/eval_std_mu", std_mu_eval, global_step)
            obs, episode_reward , eval_episode_reward= env.reset(), 0, 0
            eval_flag = False 
            eval_done = False
            on_levy = 0 

env.close()
writer.close()
