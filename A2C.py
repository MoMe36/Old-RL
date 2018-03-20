import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.distributions import Categorical 
from torch.autograd import Variable 

import numpy as np 


class Value(nn.Module): 

	def __init__(self, env_infos, hidden, lr): 

		nn.Module.__init__(self)
		self.env_infos = env_infos
		self.hidden = hidden 
		self.lr = lr 

		self.linears = nn.ModuleList([nn.Linear(env_infos[0], hidden[0])])
		for i in range(len(hidden)-1): 
			self.linears.append(nn.Linear(hidden[i], hidden[i+1]))
		self.out = nn.Linear(hidden[-1], 1)

		self.adam = optim.Adam(self.parameters(), lr)

	def set_to_zero(self): 

		for l in self.linears: 
			for p in l.parameters(): 
				zeros = torch.zeros_like(p.data)
				p.data = zeros

	def forward(self, x): 

		for l in self.linears:
			x = F.relu(l(x))

		estim = self.out(x)
		return estim 

	def train(self, loss): 

		self.adam.zero_grad()
		loss.backward()
		self.adam.step()

class Policy(nn.Module): 

	def __init__(self, env_infos, hidden, lr): 

		nn.Module.__init__(self)
		self.env_infos = env_infos
		self.hidden = hidden 
		self.lr = lr 

		self.linears = nn.ModuleList([nn.Linear(env_infos[0], hidden[0])])
		for i in range(len(hidden)-1): 
			self.linears.append(nn.Linear(hidden[i], hidden[i+1]))
		self.out = nn.Linear(hidden[-1], env_infos[1])

		self.adam = optim.Adam(self.parameters(), lr)

	def forward(self, x): 

		for l in self.linears: 
			x = F.relu(l(x))

		probs = F.softmax(self.out(x))
		return probs 

	def train(self, loss): 

		self.adam.zero_grad()
		loss.backward()
		self.adam.step()

	def clone(self): 

		clone_policy = Policy(self.env_infos, self.hidden, self.lr)

		for p_source, p_target in zip(self.parameters(), clone_policy.parameters()): 
			p_target.data = p_source.data

		return clone_policy
class A2C(): 

	def __init__(self, env_infos, pol_infos, val_infos): 

		self.env_infos = env_infos
		self.policy = Policy(env_infos, pol_infos[0], pol_infos[1])
		self.value = Value(env_infos, val_infos[0], val_infos[1])

		self.p_mem = []
		self.v_mem = []
		self.r_mem = []

		self.is_training = True 

		self.tau, self.gamma = 0.95,0.99

	def clear(self): 

		self.p_mem = []
		self.v_mem = []
		self.r_mem = []

	def eval(self): 
		self.is_training = False

	def think(self, x): 

		probs = self.policy(x)	
		m = Categorical(probs)
		action = m.sample()

		if self.is_training: 
			estim = self.value(x)
			self.p_mem.append(m.log_prob(action))
			self.v_mem.append([estim])

		return action.data[0]

	def observe(self, new_state, reward, done): 

		mask = 0. if done else 1. 
		next_val = self.value(new_state).data[0,0]*mask 

		self.v_mem[-1].append(next_val)
		self.r_mem.append(reward)

	def estimate_advantage(self): 

		retours, avantages = [], []
		r, a = 0,0
		for i in reversed(range(len(self.r_mem))): 
			r = self.gamma*r + self.r_mem[i]
			d = r + self.v_mem[i][1]*self.gamma - self.v_mem[i][0].data[0,0]
			a = r + self.tau*self.gamma*d 

			retours.insert(0,r)
			avantages.insert(0,a)

		avantages = np.array(avantages)
		avantages = (avantages - np.mean(avantages))/(np.std(avantages)+1e-6)
		return retours, avantages


	def train(self):

		retours, avantages = self.estimate_advantage()
		policy_loss = []

		retours = Variable(torch.Tensor(retours))
		estims = [self.v_mem[i][0] for i in range(len(self.v_mem))]
		estims = torch.cat(estims)

		value_loss = F.mse_loss(estims, retours) 
		self.value.train(value_loss)

		for a,xp in zip(avantages, self.p_mem): 
			policy_loss.append(-xp*a)
		policy_loss = torch.mean(torch.cat(policy_loss))
		self.policy.train(policy_loss)

		self.clear()





import gym 
env =gym.make('CartPole-v0')

env_infos = [4,2]
h = [10]
lr = 5e-3

player = A2C(env_infos, [h, lr], [h,lr])

epochs = 1000

mean_reward = 0. 
log = 100
for epoch in range(epochs): 
	s = env.reset()
	done = False 
	reward = 0 

	while not done: 

		s_tensor = Variable(torch.Tensor(s.tolist())).unsqueeze(0)
		action = player.think(s_tensor) 

		ns, r, done, _ = env.step(action)

		reward += r 
		ns_t = Variable(torch.Tensor(ns.tolist())).unsqueeze(0)
		player.observe(ns_t, r, done)
		s = ns 

		if done: 

			player.train()
			mean_reward += reward*1./log
			if epoch%log == 0: 
				print('Epoch {}/{} -- Reward {:.2f} '.format(epoch, epochs, mean_reward))
				mean_reward = 0 

player.eval()
import time
for i in range(10): 

	s = env.reset()
	done = False 

	while not done: 
		env.render()
		s_tensor = Variable(torch.Tensor(s.tolist())).unsqueeze(0) 
		action = player.think(s_tensor)

		ns, r, done, _ = env.step(action)
		s = ns 
		time.sleep(0.02)




