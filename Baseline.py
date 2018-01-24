import random
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.autograd import Variable 
import numpy as np 
import gym 

class Memory(): 

	def __init__(self, size): 

		self.size = size
		self.buffer = []
		self.count = 0

	def add(self, xp): 

		if len(self.buffer) < self.size: 
			self.buffer.append(xp)
		else: 
			self.buffer[self.count] = xp

		self.count = (self.count + 1)%self.size

	def clear(self): 
		self.buffer = []
		self.count = 0

	def sample(self, batch_size): 

		taille = batch_size if batch_size < len(self.buffer) else len(self.buffer)
		batch = random.sample(self.buffer, taille)
		return batch, taille


class Actor(): 

	def __init__(self, model, env_infos, lr = 5e-3, update_freq = 5, alone = False): 

		self.brain = model 
		self.env_infos = env_infos
		self.adam = optim.Adam(self.brain.parameters(), lr)
		self.update = [1,update_freq]
		self.alone = alone

	def think(self, state_tensor): 

		out = self.brain.forward(state_tensor).data.numpy()
		action = np.random.choice(np.arange(self.env_infos[1]), p = out.reshape(-1))
		return action 

	def train(self,history): 

		if self.alone:
			history = np.array(history)
			history[:,2] = self.discountReward(history[:,2])

		sH, aH, rH = [],[],[] 
		for ep in history: 
			sH.append(ep[0]); aH.append(ep[1]); rH.append(ep[2])

		indexes = np.arange(len(aH))*self.env_infos[1] + aH
		indexes_tensor = Variable(torch.Tensor(indexes.tolist()).type(torch.LongTensor))

		states_tensor = Variable(torch.Tensor(sH))
		probas = self.brain.forward(states_tensor)
		selected = probas.view(-1).index_select(0, indexes_tensor)

		weights = Variable(torch.Tensor(rH))

		loss = -torch.sum(torch.log(selected)*weights)
		loss.backward()

		self.apply()

	def apply(self): 
		self.update[0] = (self.update[0] + 1)%self.update[1]
		if self.update[0] == 0: 
			self.adam.step()
			self.adam.zero_grad()


	def discountReward(self, rewards, gamma = 0.9): 

		result = np.zeros_like(rewards)
		current = 0 
		for i in reversed(range(rewards.shape[0])): 
			current = current*gamma + rewards[i]
			result[i] = current
		return result

class Estimator(): 

	def __init__(self, td, env_infos, lr = 1e-2, batch_size = 64, memory_size = 1500):

		self.brain = td 
		self.env_infos = env_infos
		self.batch_size = batch_size
		self.memory = Memory(memory_size)
		self.adam = optim.Adam(self.brain.parameters(), lr)

	def eval(self, state_tensor): 

		out = self.brain.forward(state_tensor).data.numpy()
		return out

	def remember(self, xp): 
		self.memory.add(xp)

	def train(self): 

		batch, size = self.memory.sample(self.batch_size)
		sH, rH, nH = [],[],[]
		for b in batch: 
			sH.append(b[0])
			rH.append(b[2])
			nH.append(b[3])

		state_tensor = Variable(torch.Tensor(sH))
		state_value = self.brain.forward(state_tensor)


		next_state_tensor = Variable(torch.Tensor(nH), volatile = True)
		next_state_value = self.brain.forward(next_state_tensor)
		next_state_value.volatile = False
		targets = next_state_value.view(-1)*0.9 + Variable(torch.Tensor(rH)).view(-1)

		loss_fn = nn.MSELoss()
		loss = loss_fn(state_value, targets)
		self.adam.zero_grad()
		loss.backward()
		self.adam.step()

class Baseline(): 

	def __init__(self, player, estimator, env_infos): 

		self.actor = player
		self.estimator = estimator
		self.env_infos = env_infos

	def think(self, state_tensor): 
		val = self.actor.think(state_tensor)
		return val 

	def eval(self, state_tensor): 
		val = self.estimator.eval(state_tensor)
		return val

	def train(self, ep_history): 

		ep_history = np.array(ep_history)
		ep_history[:,2] = self.actor.discountReward(ep_history[:,2])

		sH = []
		for ep in ep_history: 
			sH.append(ep[0])

		sH_tensor = Variable(torch.Tensor(sH))
		estimation = self.eval(sH_tensor)
		ep_history[:,2] = ep_history[:,2] - estimation.reshape(-1)

		history = []
		for ep in ep_history: 
			self.estimator.remember(ep)
			history.append([ep[0], ep[1], ep[2]])
		self.estimator.train()
		self.actor.train(history)




env = gym.make('CartPole-v0')
env_infos = [4,2]
hidden = 64


p_mdl = nn.Sequential(nn.Linear(env_infos[0], hidden), nn.ReLU(), nn.Linear(hidden, env_infos[1]), nn.Softmax())
e_mdl = nn.Sequential(nn.Linear(env_infos[0], hidden), nn.Sigmoid(), nn.Linear(hidden, 1))

ac = Actor(p_mdl, env_infos)
es = Estimator(e_mdl, env_infos)
player = Baseline(ac, es, env_infos)

epochs = 5000
info, mean_reward = 100,0

for epoch in range(epochs): 

	s =env.reset()
	done, steps, reward = False, 0,0
	ep_history = []

	while not done: 

		state_tensor = Variable(torch.Tensor(s.tolist())).unsqueeze(0)
		action = player.think(state_tensor)

		ns, r, done, _ = env.step(action)
		ep_history.append([s.tolist(), action, r, ns.tolist()])

		s = ns
		reward += r 
		steps += 1 
		if done: 
			mean_reward += 1.*reward/info
			player.train(ep_history)

			if epoch%info == 0 : 
				print('It: {}/{} -- mean_reward = {:.2f} '.format(epoch, epochs, mean_reward))
				mean_reward = 0
