import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.distributions import Categorical
from torch.autograd import Variable 

import numpy as np 

class Policy(nn.Module): 
	
	def __init__(self, env_infos, hidden, lr): 

		nn.Module.__init__(self)
		self.hidden = hidden
		self.env_infos = env_infos 
		self.lr = lr 

		self.linears = nn.ModuleList()
		for i in range(len(hidden)): 
			if i == 0: 
				l = nn.Linear(env_infos[0], hidden[0])
			else:
				l = nn.Linear(hidden[i-1], hidden[i])
			self.linears.append(l)

		self.mu = nn.Linear(hidden[-1], 1)
		self.sigma = nn.Linear(hidden[-1],1)

		self.adam = optim.Adam(self.parameters(), lr)

	
	def forward(self, x): 

		for l in self.linears: 
			x = F.relu(l(x))

		mean = F.softmax(self.head(x))

		return probs 

	def update(self, loss): 

		self.adam.zero_grad()
		loss.backward()
		self.adam.step()

	def release_clone(self):

		clone = Policy(self.env_infos, self.hidden, self.lr)
		clone.load_state_dict(self.state_dict())

		return clone 

	def eval_states(self, x, actions): 

		probs = self.forward(x)
		selected = torch.gather(probs, dim = 1, index = actions)
		return selected


class Value(nn.Module): 
	
	def __init__(self, env_infos, hidden, lr): 

		nn.Module.__init__(self)
		self.hidden = hidden
		self.env_infos = env_infos 
		self.lr = lr 

		self.linears = nn.ModuleList()
		for i in range(len(hidden)): 
			if i == 0: 
				l = nn.Linear(env_infos[0], hidden[0])
			else:
				l = nn.Linear(hidden[i-1], hidden[i])
			self.linears.append(l)

		self.head = nn.Linear(hidden[-1], 1)

		self.adam = optim.Adam(self.parameters(), lr)

	
	def forward(self, x): 

		for l in self.linears: 
			x = F.relu(l(x))

		probs = F.softmax(self.head(x))

		return probs 

	def update(self, loss): 

		self.adam.zero_grad()
		loss.backward()
		self.adam.step()

class PPO(): 

	def __init__(self, env_infos, policy_param, value_param): 

		self.policy = Policy(env_infos, policy_param[0], policy_param[1])
		self.value = Value(env_infos, value_param[0], value_param[1])

		self.clone = self.policy.release_clone()

		self.actions = []
		self.visited_states = []
		self.v_mem = []
		self.rewards = []

		self.is_training = True

	def clear(self): 

		self.actions = []
		self.visited_states = []
		self.v_mem = []
		self.rewards = []

	def eval(self): 

		self.is_training = False 

	def demo(self, x): 

		probs = self.policy(x).data.numpy().reshape(-1)
		return np.argmax(probs), probs

	def think(self, x): 

		probs = self.policy(x)
		m = Categorical(probs)
		action = m.sample()

		if(self.is_training): 

			self.actions.append(action)
			self.visited_states.append(x)
			self.v_mem.append([self.value(x)])

		return action.data[0]

	def observe(self, new_state, reward, done): 

		mask = 0. if done else 1. 
		next_val = self.value(new_state).data[0]*mask

		self.v_mem[-1].append(next_val)
		self.rewards.append(reward)

	def get_advantage(self): 

		avantages, retours = [],[]
		a = 0
		r = 0 

		for i in reversed(range(len(self.rewards))): 

			r = r*0.99 + self.rewards[i]
			d = r + 0.99*self.v_mem[i][1][0] - 0.99*self.v_mem[i][0].data[0,0]
			a = r + 0.99*0.95*d 

			avantages.insert(0,a)
			retours.insert(0,r)

		avantages = torch.Tensor(avantages)
		avantages = (avantages - avantages.mean())/(avantages.std() + 1e-6)

		return avantages.view(-1,1), torch.Tensor(retours).view(-1,1)


	def train(self): 

		avantages, retours = self.get_advantage()

		# train critic
		next_estims = torch.cat([self.v_mem[i][1] for i in range(len(self.v_mem))]).view(-1,1)
		target = retours + 0.99*next_estims
		target = Variable(target)

		estims = torch.cat([self.v_mem[i][0] for i in range(len(self.v_mem))]).view(-1,1)

		value_loss = F.mse_loss(estims, target)
		self.value.update(value_loss)

		# train actor 
		states = torch.cat(self.visited_states)
		actions = torch.cat(self.actions).view(-1,1)

		old_prob = self.clone.eval_states(states, actions)
		new_prob = self.policy.eval_states(states, actions)

		ratio = new_prob/old_prob
		
		avantages = Variable(avantages)
		surr_1 = ratio*avantages
		surr_2 = torch.clamp(ratio, 0.8,1.2)*avantages
		policy_loss = -torch.mean(torch.min(surr_1,surr_2))

		self.clone = self.policy.release_clone()
		self.policy.update(policy_loss)
		
		self.clear()


import gym 
env = gym.make('CartPole-v0')

env_infos = [4,2]