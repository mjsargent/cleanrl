import numpy as np
import time
import collections
import random

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

    def sample2(self, n):
        idx = np.random.choice(len(self.buffer), n, replace=False)
        minibatch = self.buffer[idx]

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

class MultiBuffer():
    def __init__(self, buffer_limit):
        self.state_buffer = collections.deque(maxlen=buffer_limit)
        self.action_buffer = collections.deque(maxlen=buffer_limit)
        self.next_state_buffer = collections.deque(maxlen=buffer_limit)
        self.reward_buffer = collections.deque(maxlen=buffer_limit)
        self.done_buffer = collections.deque(maxlen=buffer_limit)
        self.buffer_limit = buffer_limit

    def put(self, transition):
        s, a, s_, r, d = transition
        self.state_buffer.append(s)
        self.action_buffer.append(a)
        self.next_state_buffer.append(s_)
        self.reward_buffer.append(r)
        self.done_buffer.append(d)

    def sample(self, n):
        idxs = np.random.choice(self.buffer_limit, n, replace=False)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        for idx in idxs:
            s_lst.append(self.state_buffer[idx])
            a_lst.append(self.action_buffer[idx])
            r_lst.append(self.reward_buffer[idx])
            s_prime_lst.append(self.next_state_buffer[idx])
            done_mask_lst.append(self.done_buffer[idx])

        return np.array(s_lst), np.array(a_lst), \
               np.array(r_lst), np.array(s_prime_lst), \
               np.array(done_mask_lst)

 
class CyclicBuffer():
    def __init__(self, size, state_size):
        self.buffer = np.array(size, state_size, 1, state_size, 1, 1)
        self.position = 0   
        self.buffer_size = size

    def put(self, transition):
        self.buffer[self.position] = np.array(transition)
        self.position = (self.position + 1 ) % self.buffer_size
    
    def sample(self, n):
        idxs = np.random.choice(self.buffer_limit, n, replace=False)
        transitions = self.buffer[idxs]
        
        return 

replayBuffer = ReplayBuffer(1000000)
multiBuffer = MultiBuffer(1000000)
a = ([random.random() for i in range(10000)], (0), (1), ([random.random() for i in range(10000)]), (0))
[replayBuffer.put(a) for i in range(1000000)]

[multiBuffer.put(a) for i in range(1000000)]

start = time.time()
replayBuffer.sample(10000)
end = time.time()
print("normal buffer sampling: {} ".format(end - start))

start = time.time()
multiBuffer.sample(10000)
end = time.time()
print("multi buffer sampling: {} ".format(end - start))

