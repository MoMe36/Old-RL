import gym 
import torch
from PPO import PPO 


env = gym.make('Pendulum-v0')
env_infos = [3,1]
params_policy = [[256,128],3e-4]
params_value = [[128,128],3e-3]

agent = PPO(env_infos, params_policy, params_value)


epochs = 1000
log = 30
mean_reward = 0.
for epoch in range(1,epochs+1): 

	s = env.reset()
	done = False 
	reward = 0. 

	while not done: 

		action = agent.think(torch.tensor(s.reshape(1,-1)).float())
		ns, r, done, _ = env.step(action)

		agent.observe(torch.tensor(ns.reshape(1,-1)).float(), r ,done)

		s = ns 
		reward += r 

		if done: 

			mean_reward += reward

			agent.train()
			if epoch%log == 0: 
				print('Epoch  {} -- Reward {:.3f}'.format(epoch, mean_reward*1./log))
				mean_reward = 0.
