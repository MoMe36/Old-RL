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

		clone_policy.load_state_dict(self.state_dict())
		# for p_source, p_target in zip(self.parameters(), clone_policy.parameters()): 
		# 	p_target.data = p_source.data

		return clone_policy

	def evaluation(self, states, actions):

		actions = actions.view(-1,1)
		probs = self.forward(states)
		selected = torch.gather(probs, 1, actions)

		# return torch.log(selected)
		return selected


class PPO(): 

	def __init__(self, env_infos, pol_infos, val_infos): 

		self.env_infos = env_infos
		self.policy = Policy(env_infos, pol_infos[0], pol_infos[1])
		self.value = Value(env_infos, val_infos[0], val_infos[1])

		self.p_mem = []
		self.v_mem = []
		self.r_mem = []

		self.chosen_action = []
		self.visited_states = []

		self.is_training = True 

		self.tau, self.gamma = 0.95,0.99

		self.policy_clone = self.policy.clone()

	def clear(self): 

		self.p_mem = []
		self.v_mem = []
		self.r_mem = []

		self.chosen_action = []
		self.visited_states = []

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

			self.chosen_action.append(action.data[0])
			self.visited_states.append(x)

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

		retours = Variable(torch.Tensor(retours))
		avantages = Variable(torch.Tensor(avantages))

		states_v = torch.cat(self.visited_states)
		actions = Variable(torch.LongTensor(self.chosen_action))

		old_probs = self.policy_clone.evaluation(states_v, actions)
		new_probs = self.policy.evaluation(states_v, actions)
		log_probs = torch.cat(self.p_mem)

		# ratio = torch.exp(new_probs - old_probs)
		ratio = (new_probs/old_probs)
		surr_1 = ratio*avantages
		surr_2 = torch.clamp(ratio, 0.8,1.2)*avantages

		loss_policy = -torch.mean(torch.min(surr_2, surr_1)*log_probs)

		# print('Loss : {}'.format(loss_policy.data[0]))
		self.policy_clone = self.policy.clone()
		self.policy.train(loss_policy)
		
		estims = [self.v_mem[i][0] for i in range(len(self.v_mem))]
		estims = torch.cat(estims)

		value_loss = F.mse_loss(estims, retours) 
		self.value.train(value_loss)

		self.clear()





import gym 
env =gym.make('CartPole-v0')

env_infos = [4,2]
h = [32]
lr = 1e-2

player = PPO(env_infos, [h, lr], [h,lr])

epochs = 500

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




