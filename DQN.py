import torch 
import torch.nn as nn 
from torch.autograd import Variable 
import torch.optim as optim
import numpy as np 
import robotWorld as World
import random
import matplotlib.pyplot as plt

plt.style.use('ggplot')


class Memory(): 

	def __init__(self, size): 

		self.count = 0
		self.buffer = []
		self.size = size

	def add(self, xp): 

		if len(self.buffer) < self.size: 
			self.buffer.append(xp)
		else: 
			self.buffer[self.count] = xp

		self.count = (self.count+1)%self.size

	def clear(self): 

		self.buffer = []

	def randomSample(self, batch_size): 

		taille = batch_size if batch_size < len(self.buffer) else len(self.buffer)
		sample = random.sample(self.buffer, taille)
		return sample, taille

def discountReward(r, gamma = 0.99): 

	discounted = np.zeros_like(r)
	current = 0

	for i in reversed(range(r.shape[0])): 
		current = current*gamma + r[i]
		discounted[i] = current
	return discounted


# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#						     World Creation 
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-



# robotJoints controls the number of joints in the robot 
# world is designed to be inside a [0->1], [0->1] space. Hence, robotJointsLength*robotJoints should be less than 1

env = World.World(robotJoints = 2, robotJointsLength = 0.35, 
	randomizeRobot = False, randomizeTarget = False, 
	groundHeight =0.05, targetLimits = [0.2,0.8,0.1,0.6])

observation_space = 4
action_space = env.actionSpaceSize()

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#						     Model Creation 
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-


hiddenSize = 100
model = nn.Sequential(nn.Linear(observation_space, hiddenSize), nn.ReLU(), nn.Linear(hiddenSize, action_space))
adam = optim.Adam(model.parameters(), 1e-3)

successiveActions = 3 # number of frames before choosing new action given perceptions



# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#						    Learning Loop
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

epochs = 2000
batch_size = 128
memory = Memory(2000)


epsi_max = 0.9
epsi_min = 0.1
epsi_decay = 500

informationFreq = 100
successes = 0
success_history = []

for epoch in range(epochs):

	complete = False
	steps = 0
	reward =  0
	ep_history = []

	s = env.reset()

	while not complete: 

		# Choosing action via epsilon greedy methods 

		if steps % successiveActions == 0:
			epsi = epsi_min + (epsi_max - epsi_min)*np.exp(-epoch*1./epsi_decay)
			exploration = np.random.random()

			if exploration < epsi: 
				action = env.randomAction()
			else: 
				sTensor = Variable(torch.Tensor(s).type(torch.FloatTensor), requires_grad = False)
				result = model(sTensor)
				action = np.argmax((result.data.numpy()).reshape(-1))

		# Getting observation as a result of interaction and saving transition 

		newState, r, complete, success = env.step(action)

		ep_history.append([s, action, r, newState])
		reward += r
		s = newState
		steps += 1

		if complete: 
			successes += success

			ep_history = np.array(ep_history)
			ep_history[:,2] = discountReward(ep_history[:,2])

			# Adding episodes to memory

			for ep in ep_history: 
				memory.add(ep)
			

			# Sampling mini_batch from memory

			mini_batch,trueSize = memory.randomSample(batch_size)

			sH = np.zeros((trueSize, observation_space))
			rH = np.zeros((trueSize))
			aH = np.zeros((trueSize))
			nsH = np.zeros_like(sH)

			for ex in range(trueSize): 
				sH[ex,:] = mini_batch[ex][0]
				aH[ex] = mini_batch[ex][1]
				rH[ex] = mini_batch[ex][2]
				nsH[ex,:] = mini_batch[ex][3]


			indexes = np.arange(len(mini_batch))*action_space
			for it, ind in enumerate(indexes): 
				indexes[it] += aH.reshape(-1)[it]

			# Value iteration 

			shTensor = Variable(torch.from_numpy(sH).type(torch.FloatTensor))
			qValues = model.forward(shTensor)

			selectedActions = Variable(torch.from_numpy(indexes).type(torch.LongTensor))
			qValues = qValues.view(-1).index_select(dim = 0, index = selectedActions)

			nextStateTensor = Variable(torch.from_numpy(nsH).type(torch.FloatTensor), volatile = True)
			nextQValues = model.forward(nextStateTensor).max(1)[0]
			nextQValues.volatile = False

			rewardTensor = Variable(torch.from_numpy(rH).type(torch.FloatTensor))
	
			expected = rewardTensor + 0.9*nextQValues

			loss_fn = nn.MSELoss()
			loss = loss_fn(qValues, expected)
			adam.zero_grad()
			loss.backward()
			adam.step()


			# Book keeping

			if epoch%informationFreq == 0: 

				text = 'It: {} / {} | Success {} / {} | Explo: {} '.format(epoch,epochs,successes, informationFreq, epsi)
				print(text)
				success_history.append(successes)
				successes = 0
				torch.save(model, 'catcher.robot')



# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#						   Visualisation
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

success_history = np.array(success_history)
plt.plot(range(success_history.shape[0]), success_history)
plt.title('Successes over slices of {} trials'.format(informationFreq))
plt.xlabel('Trials')
plt.ylabel('Successes')
plt.show()


env.initRender()
s = env.reset()
steps = 0
epsilon = 0.1
while True: 

	if steps%successiveActions == 0: 

		if np.random.random() < epsilon:
			action = env.randomAction()
		else:
			sTensor = Variable(torch.Tensor(s).type(torch.FloatTensor), requires_grad = False)
			result = model(sTensor)
			action = np.argmax((result.data.numpy()).reshape(-1))

	ns,r,d,_ = env.step(action)
	env.render()

	s = ns
	steps += 1 
	if d: 
		steps = 0
		s = env.reset()








			




		






