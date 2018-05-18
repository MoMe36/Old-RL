import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.distributions import MultivariateNormal, Normal

import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use('ggplot')


# 								PPO 
# This implementation works as follows: 
# Two networks, policy and values are created. They are separated networks. 
# Also, the clone policy network is created. It is used in the policy improvement to 
# prevent the updated policy to stray too far away from the old one.

# These two networks are then held in the PPO agent for ease of use. 
# The PPO agent keeps track of the visited states, actions taken, rewards and states values 
# At each time step, the think method is called to return an action from the policy.
# It also records the state, and the state value 

# After receiving the rewards, the observe method records the new state value, 
# the reward and a mask for terminal states


# Lastly, the train method is used to update all the networks according to the PPO algorithm 


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

		self.means = nn.Linear(hidden[-1], env_infos[1])
		self.stds = nn.Linear(hidden[-1], env_infos[1])

		self.adam = optim.Adam(self.parameters(), lr)

	
	def forward(self, x): 

		for l in self.linears: 
			x = F.tanh(l(x))

		means = F.tanh(self.means(x))
		stds = torch.exp(self.stds(x))
		return means, stds 

	def maj(self, loss): 

		self.adam.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.parameters(),10.)
		self.adam.step()

	def release_clone(self):

		clone = Policy(self.env_infos, self.hidden, self.lr)
		for source, target in zip(self.state_dict().values(), clone.state_dict().values()): 
			target.copy_(source)

		return clone 

	def eval_states(self, x, actions): 


		means, stds = self.forward(x)

		nb_examples = means.shape[0]
		log_probs = torch.zeros(means.shape[0],1)

		for i,(a,m,s) in enumerate(zip(actions, means, stds)): 

			multivar = MultivariateNormal(m,torch.diag(s))
			lp = multivar.log_prob(a)

			log_probs[i,:] = lp

		return log_probs




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

		probs = self.head(x)

		return probs 

	def maj(self, loss): 

		self.adam.zero_grad()
		loss.backward()
		self.adam.step()

class PPO(): 

	def __init__(self, env_infos, policy_param, value_param, clone_update = 2): 

		self.policy = Policy(env_infos, policy_param[0], policy_param[1])
		self.value = Value(env_infos, value_param[0], value_param[1])

		self.clone = self.policy.release_clone()

		self.actions = []
		self.visited_states = []
		self.v_mem = []
		self.rewards = []
		self.entropy = []

		self.is_training = True

		self.clone_update = clone_update
		self.counter = 0 



		self.debugger = Debug()


	def clear(self): 

		self.actions = []
		self.visited_states = []
		self.v_mem = []
		self.rewards = []
		self.entropy = []

	def eval(self): 

		self.is_training = False 

	def demo(self, x): 

		means, _ = self.policy(x)
		action = means.detach().numpy().reshape(-1)
		return np.argmax(action), action

	def demo_non_tensor(self,x): 

		x = torch.tensor(x).float().reshape(1,-1)
		return self.demo(x)

	def think(self, x): 

		means, stds = self.policy(x)
		dist = MultivariateNormal(means, torch.diag(stds.reshape(-1)))


		try: 
			action = dist.sample()
		except RuntimeError: 
			print('Entering error') #usually because of NaNs 
			self.debugger.draw(final = True)

			input('Showing policy weights')
			for p in self.policy.state_dict().values(): 
				print(p)
				input()


		if(self.is_training): 
			self.entropy.append(dist.entropy())
			self.actions.append(action)
			self.visited_states.append(x)
			self.v_mem.append([self.value(x)])

		return action.detach().numpy().reshape(-1)

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
			d = r + 0.99*self.v_mem[i][1].item() - 0.99*self.v_mem[i][0].item()
			a = d + 0.99*0.95*a 

			avantages.insert(0,a)
			retours.insert(0,r)

		avantages = torch.tensor(avantages)
		avantages = (avantages - avantages.mean())/(avantages.std() + 1e-6)

		return avantages.view(-1,1), torch.tensor(retours).view(-1,1)


	def train(self): 

		entropy_coeff = 0.5
		ent = torch.cat(self.entropy)
		avantages, retours = self.get_advantage()

		# VALUE UPDATE 

		next_estims = np.zeros((len(self.v_mem),1))
		for i in range(next_estims.shape[0]): 
			next_estims[i,0] = self.v_mem[i][1]

		next_estims = torch.tensor(next_estims).float()
		target = retours + 0.99*next_estims

		estims = torch.cat([self.v_mem[i][0] for i in range(len(self.v_mem))]).view(-1,1)

		value_loss = F.mse_loss(estims, target)
		self.value.maj(value_loss)

		# POLICY UPDATE 


		states = torch.cat(self.visited_states)
		actions = torch.cat(self.actions)

		if states.shape[0] > 1: 
			old_prob = self.clone.eval_states(states, actions)
			new_prob = self.policy.eval_states(states, actions)

			ratio = torch.exp(new_prob - old_prob)
			ratio = torch.where(ratio != ratio, torch.ones_like(ratio), ratio)
						

			surr_1 = ratio*avantages
			surr_2 = torch.clamp(ratio, 0.8,1.2)*avantages
			policy_loss = -(torch.min(surr_1,surr_2)).mean()

			if self.counter%self.clone_update == 0:
				self.clone = self.policy.release_clone()
			self.policy.maj(policy_loss)


			#-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
			#-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
			#					DEBUG
			self.debugger.receive_ratio(ratio)
			self.debugger.receive_reward(avantages.detach().numpy().reshape(-1))
			self.debugger.receive_weights1(self.policy.linears[0].weight)
			self.debugger.receive_weights2(self.policy.linears[1].weight)
			self.debugger.receive_action(actions.detach().numpy().reshape(-1))
			self.debugger.receive_probs(old_prob, new_prob)
			self.debugger.receive_entropy(ent.detach().numpy().reshape(-1))
			self.debugger.receive_surrogates(surr_1.detach().numpy().reshape(-1), surr_2.detach().numpy().reshape(-1))
			#-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
			#-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
		
		self.clear()
		self.counter = (self.counter +1)%self.clone_update

	def save(self, path): 

		torch.save(self.policy.state_dict(), path+'policy')
		torch.save(self.value.state_dict(), path+'value')

	def load(self, path): 

		self.policy.load_state_dict(torch.load(path+'policy'))
		self.value.load_state_dict(torch.load(path+'value'))

	def show_debugger(self): 

		self.debugger.draw()



class Debug(): 

	def __init__(self): 

		self.ini = False 
		self.min_ratio, self.max_ratio, self.mean_ratio = [],[],[]
		self.min_reward, self.max_reward, self.mean_reward = [],[],[]

		self.l1_mean_weight, self.l1_min_weight, self.l1_max_weight = [],[],[]
		self.l2_mean_weight, self.l2_min_weight, self.l2_max_weight = [],[],[]

		self.means_mean, self.means_min, self.means_max = [],[],[]
		self.stds_min, self.stds_max, self.stds_mean = [],[],[]

		self.old_prob_min,self.old_prob_max,self.old_prob_mean, = [],[],[]
		self.new_prob_min,self.new_prob_max,self.new_prob_mean, = [],[],[]

		self.entropy_min, self.entropy_max, self.entropy_mean = [],[],[]

		self.s1_min,self.s1_max,self.s1_mean, = [],[],[]
		self.s2_min,self.s2_max,self.s2_mean, = [],[],[]


	def receive_ratio(self, r): 

		np_r = r.detach().numpy().reshape(-1)
		self.min_ratio.append(np.min(np_r))
		self.max_ratio.append(np.max(np_r))
		self.mean_ratio.append(np.mean(np_r))

	def receive_reward(self, r): 

		my_r = r.copy()
		self.min_reward.append(np.min(my_r))
		self.max_reward.append(np.max(my_r))
		self.mean_reward.append(np.mean(my_r))

	def receive_weights1(self,l1w): 

		np_r = torch.zeros_like(l1w)
		np_r.copy_(l1w)
		np_r = np_r.detach().numpy().reshape(-1)
		self.l1_min_weight.append(np.min(np_r))
		self.l1_max_weight.append(np.max(np_r))
		self.l1_mean_weight.append(np.mean(np_r))

	def receive_weights2(self,l1w): 

		np_r = torch.zeros_like(l1w)
		np_r.copy_(l1w)
		np_r = np_r.detach().numpy().reshape(-1)
		self.l2_min_weight.append(np.min(np_r))
		self.l2_max_weight.append(np.max(np_r))
		self.l2_mean_weight.append(np.mean(np_r))

	def receive_action(self, m): 

		self.means_min.append(np.min(m))
		self.means_max.append(np.max(m))
		self.means_mean.append(np.mean(m))

	def receive_entropy(self, m): 

		self.entropy_min.append(np.min(m))
		self.entropy_max.append(np.max(m))
		self.entropy_mean.append(np.mean(m))

	def receive_probs(self, old, new): 

		np_r = torch.zeros_like(old)
		np_r.copy_(old)
		np_r = np_r.detach().numpy().reshape(-1)

		self.old_prob_min.append(np.min(np_r))
		self.old_prob_max.append(np.max(np_r))
		self.old_prob_mean.append(np.mean(np_r))

		np_r = torch.zeros_like(new)
		np_r.copy_(new)
		np_r = np_r.detach().numpy().reshape(-1)

		self.new_prob_min.append(np.min(np_r))
		self.new_prob_max.append(np.max(np_r))
		self.new_prob_mean.append(np.mean(np_r))

	def receive_surrogates(self, s1,s2): 

		self.s1_min.append(np.min(s1))
		self.s1_max.append(np.max(s1))
		self.s1_mean.append(np.mean(s1))

	
		self.s2_min.append(np.min(s2))
		self.s2_max.append(np.max(s2))
		self.s2_mean.append(np.mean(s2))


	def draw(self, final = False): 

		if not self.ini: 
			self.ini = True
			self.f, self.ax = plt.subplots(2,5)
		else: 
			for a in self.ax: 
				for aa in a: 
					aa.clear()

		size = np.arange(len(self.min_ratio))
		self.ax[0,0].plot(size, self.min_ratio, label = 'min_ratio')
		self.ax[0,0].plot(size, self.max_ratio, label = 'max_ratio')
		self.ax[0,0].plot(size, self.mean_ratio, label = 'mean_ratio')

		size = np.arange(len(self.min_reward))
		self.ax[0,1].plot(size, self.min_reward, label = 'Avantage_min')
		self.ax[0,1].plot(size, self.max_reward, label = 'Avantage_max')
		self.ax[0,1].plot(size, self.mean_reward, label = 'Avantage_mean')

		size = np.arange(len(self.l1_min_weight))
		self.ax[0,2].plot(size, self.l1_min_weight, label = 'l1_min_weight')
		self.ax[0,2].plot(size, self.l1_max_weight, label = 'l1_max_weight')
		self.ax[0,2].plot(size, self.l1_mean_weight, label = 'l1_mean_weight')

		size = np.arange(len(self.l2_min_weight))
		self.ax[0,3].plot(size, self.l2_min_weight, label = 'l2_min_weight')
		self.ax[0,3].plot(size, self.l2_max_weight, label = 'l2_max_weight')
		self.ax[0,3].plot(size, self.l2_mean_weight, label = 'l2_mean_weight')

		size = np.arange(len(self.means_min))
		self.ax[0,4].plot(size, self.means_min, label = 'Action min')
		self.ax[0,4].plot(size, self.means_max, label = 'Action max')
		self.ax[0,4].plot(size, self.means_mean, label = 'Action mean')

		size = np.arange(len(self.new_prob_mean))
		self.ax[1,0].plot(size, self.new_prob_min, label = 'New prob min')
		self.ax[1,0].plot(size, self.new_prob_max, label = 'New prob max')
		self.ax[1,0].plot(size, self.new_prob_mean, label = 'New prob mean')

		size = np.arange(len(self.new_prob_mean))
		self.ax[1,1].plot(size, self.old_prob_min, label = 'Old prob min')
		self.ax[1,1].plot(size, self.old_prob_max, label = 'Old prob max')
		self.ax[1,1].plot(size, self.old_prob_mean, label = 'Old prob mean')

		size = np.arange(len(self.entropy_mean))
		self.ax[1,4].plot(size, self.entropy_min, label = 'Entropy min')
		self.ax[1,4].plot(size, self.entropy_max, label = 'Entropy max')
		self.ax[1,4].plot(size, self.entropy_mean, label = 'Entropy mean')

		size = np.arange(len(self.s1_min))
		self.ax[1,2].plot(size, self.s1_min, label = 'Surrogate 1 min')
		self.ax[1,2].plot(size, self.s1_max, label = 'Surrogate 1 max')
		self.ax[1,2].plot(size, self.s1_mean, label = 'Surrogate 1 mean')

		size = np.arange(len(self.s2_min))
		self.ax[1,3].plot(size, self.s2_min, label = 'Surrogate 2 min')
		self.ax[1,3].plot(size, self.s2_max, label = 'Surrogate 2 max')
		self.ax[1,3].plot(size, self.s2_mean, label = 'Surrogate 2 mean')

		self.ax[0,0].legend()
		self.ax[0,1].legend()
		self.ax[0,2].legend()
		self.ax[0,3].legend()
		self.ax[0,4].legend()
		self.ax[1,0].legend()
		self.ax[1,1].legend()
		self.ax[1,3].legend()
		self.ax[1,2].legend()
		self.ax[1,4].legend()

		self.ax[0,0].set_title('Ratio')
		self.ax[0,1].set_title('Advantage')
		self.ax[0,2].set_title('Policy weight distrib')
		self.ax[0,3].set_title('Policy weight distrib')
		self.ax[0,4].set_title('Action')
		self.ax[1,0].set_title('New log probs')
		self.ax[1,1].set_title('Old log probs')
		self.ax[1,2].set_title('Surrogate 1')
		self.ax[1,3].set_title('Surrogate 2')
		self.ax[1,4].set_title('Entropy')

		plt.pause(0.1)
		if final: 
			plt.show()
