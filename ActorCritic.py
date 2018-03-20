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


class ActorCritic(): 

	def __init__(self, env_infos, policy_infos, value_infos):

		self.env_infos = env_infos
		self.value = Value(env_infos,value_infos[0], value_infos[1])
		self.policy = Policy(env_infos,policy_infos[0], policy_infos[1])

		self.r_mem = []
		self.v_mem = []
		self.p_mem = []

		self.is_training = True

		self.gamma = 0.95

	def think(self, x): 

		probs = self.policy(x) 
		m = Categorical(probs)
		action = m.sample()

		if self.is_training: 
			estim = self.value(x)
			self.v_mem.append([estim])
			self.p_mem.append(m.log_prob(action))

		return action.data[0]

	def observe(self, new_state, reward, done):

		mask = 0. if done else 1.
		next_val = self.value(new_state).data[0,0]*mask 

		self.v_mem[-1].append(next_val)
		self.r_mem.append(r)

	def discount_rewards(self): 

		result, current = [],0
		for i in reversed(range(len(self.r_mem))): 
			current = current*self.gamma + self.r_mem[i]
			result.insert(0,current)
		return result

	def train(self): 

		discounted_r = self.discount_rewards()

		policy_loss = []
		value_loss = []

		# train value 
		td_error = []
		for xp,r in zip(self.v_mem, discounted_r): 

			target = r + self.gamma*xp[1]

			td_error.append(target - xp[0].data[0,0])
			l = F.mse_loss(xp[0], Variable(torch.Tensor([target])))
			value_loss.append(l)

		value_loss = torch.mean(torch.cat(value_loss))
		self.value.train(value_loss)

		# train policy 

		for a,c in zip(self.p_mem, td_error):

			policy_loss.append(-a*c)
		policy_loss = torch.mean(torch.cat(policy_loss))
		self.policy.train(policy_loss)

		self.clear_mem()

	def clear_mem(self): 

		self.v_mem = []
		self.p_mem = []
		self.r_mem = []

	def eval(self): 

		self.is_training = False



import gym 
env =gym.make('CartPole-v0')

env_infos = [4,2]
h = [32]
lr = 5e-3

player = ActorCritic(env_infos, [h, lr], [h,lr])

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
